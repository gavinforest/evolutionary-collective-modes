import cvxpy as cp
import numpy as np
import polars as pl
import os
from sklearn.linear_model import LinearRegression
from cmsim_biggmatrices_double_total import cmsim_biggmatrices_double
import cobra

# ************************************************
# ================================================
# THIS IS THE POLAR VERSION 
# ================================================
# ************************************************

# Theory-related functions -----------------------------------------------------

def FBA_gene(u_g,l_g,Au,Al,S,Gu,Gl,beta,return_lagrange=False,irreversible=False,solver='SCIPY'):
    v = cp.Variable(S.shape[1])
    objective = cp.Minimize(-beta @ v)
    constraints = [S @ v == 0, Au @ v <= Gu @ u_g, Al @ v >= Gl @ l_g]
    if irreversible:
        constraints += [v >= 0]
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=solver)
    except:
        pass
    
    if objective.value is None:

        if return_lagrange:
            return 0,np.zeros(S.shape[1]),np.zeros(S.shape[0]),np.zeros(Au.shape[0]),np.zeros(Al.shape[0])
        else:
            return 0,np.zeros(S.shape[1])       
    else:

        if return_lagrange:
            return -objective.value, v.value, constraints[0].dual_value, constraints[1].dual_value, constraints[2].dual_value
        else:
            return -objective.value, v.value



def Di(lam,i):
    
    n = lam.shape[0]
    if n > 1:
        idx = list(range(n))
        idx.pop(i)
        
        D = np.zeros((n-1,lam.shape[1]))
        for (jD,jlam) in zip(list(range(n-1)),idx): 
            D[jD,:] = lam[i,:] - lam[jlam,:]
        
        return D
    else:
        return np.zeros(lam.shape[1])
        
        
def find_cm(L,G,D,fix_idx=[],lam_idx=0,solver='DAQP',equal_zero=False):
    z = cp.Variable(G.shape[1])
    objective = cp.Minimize(cp.sum_squares(L[lam_idx,:] @ G - z))
    if equal_zero:
        constraints = [D @ G @ z == 0, z >= 0]
    else:
        constraints = [D @ G @ z <= 1e-8, D @ G @ z >= -1e-8, z >= 0]
    
    for idx in fix_idx:
        constraints += [z[idx] == 0]
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=solver)
    except:
        print("FAILED ON SOLVE --------------------------------------------")
        return None,np.random.randn(G.shape[1])*0.001
        
    try:
        return -objective.value, z.value
    except:
        print("FAILED ON OBJECTIVE VALUE ----------------------------------")
        return None,np.random.randn(G.shape[1])*0.001



def chain_sample(n,Au,Al,S,Gu,Gl,beta,ug0,lg0,scale=1,solver='SCIPY',fix_idx=[]):

    if len(fix_idx) > 0:
        print('FIXED BOUNDS NOT IMPLEMENTED FOR CHAIN SAMPLE')
        return

    lam = np.zeros((n,Au.shape[0] + Al.shape[0]))

    ug = ug0
    ug[ug < 0] = 0
    lg = lg0
    lg[lg > 0] = 0
    _,_,_,SP_u,SP_l = FBA_gene(ug,lg,Au,Al,S,Gu,Gl,beta,return_lagrange=True,solver=solver)
    
    lam[0,:] = np.hstack((SP_u,SP_l))

    for i in range(1,n):
        ug = ug + (SP_u@Gu)*scale
        lg = lg - (SP_l@Gl)*scale
        _,_,_,SP_u,SP_l = FBA_gene(ug,lg,Au,Al,S,Gu,Gl,beta,return_lagrange=True,solver=solver)
        lam[i,:] = np.hstack((SP_u,SP_l))
    
    return lam

def cm_cutoff(unique,counts_argsort,p,G_block,solver='SCS',equal_zero=False):
    D_rand = Di(unique[counts_argsort[:p+1]],0)
    f,z_rand = find_cm(unique[counts_argsort[:p+1]],G_block,D_rand,lam_idx=0,solver=solver,equal_zero=equal_zero)
    if f is None:
        return z_rand,False
    if f is not None:
        return z_rand,True
            

def sim_loop(data_name,root,sims,n_col,dtype=None,debug=False):

    #data_name is the name of the data to load (e.g., 'biomass')
    #root is the path to the folder that contains all simulations
    #sims are the simulations (names of folders) to load data for
    

    df = pl.DataFrame()
    for f in sims:
            
        if os.path.isdir(root + '/' + f):
            sim_root = root + '/' + f

            if f + 'args.txt' in os.listdir(sim_root):
                data_in = pl.read_csv(sim_root+'/'+f+data_name+'.csv',columns=list(range(1,n_col)),schema_overrides=dtype)
            else:

                #Need to remove non-folders from the list
                sim_chunks = sorted(os.listdir(sim_root))
                for sc in reversed(sim_chunks):
                    if not os.path.isdir(sim_root+'/'+sc):
                        sim_chunks.remove(sc)

                data_in = folder_loop(data_name,sim_root,sim_chunks,n_col=n_col,dtype=dtype,debug=debug)
            data_in = data_in.with_columns(pl.lit(f).alias('Sim'))
            df = pl.concat((df,data_in),how='vertical')
    
    return df
    
    
def load_sim_results(root,data_names,data_n_col,sims,dtypes=None,debug=False):

    #root is the path to folder containing the simulation folders
    #data_names are the data that will be loaded (e.g., biomass)
    #sims are the sims to load data for
    #dtype is a list of dictionarys that give datatypes for the pandas dataframes
        #the length of dtypes should be the same as files
    #If debug is true, only the first three folders per simulation/file will be loaded

    if dtypes is None:
        dtypes = [{}]*len(data_names)
    
    
    return_files = []
    
    for dn,dt,dn_col in zip(data_names,dtypes,data_n_col):
        return_files += [sim_loop(dn,root,sims,n_col=dn_col,dtype=dt,debug=debug)]
    
    return return_files
    

def folder_loop(data_name,sim_root,sim_chunks,n_col,dtype=None,debug=False):

    #data_name is the name of the data to load (e.g., 'biomass')
    #sim_root is the path to the folder that contains all chunks of a simulation
    #sim_chunks are the chunks of a simulation (names of folders) to load data for

    if debug:
        sim_chunks=sim_chunks[:2]

    
    df = pl.DataFrame()
    tend = 0
    for f in sim_chunks:
        if ('mutation' in data_name or 'neutral' in data_name) and 'Time' not in df.columns:

            if 'mutation' in data_name:
                if os.path.isdir(sim_root+'/'+f):
                    data_in = pl.read_csv(sim_root+'/'+f+'/'+f+data_name+'.csv',columns=list(range(1,n_col)),schema_overrides=dtype)
                    if f == sim_chunks[0]:
                        df = data_in
                    else:
                        df = df.join(data_in,on=['Gene'],suffix='_'+f)
            elif 'neutral' in data_name:
                if os.path.isdir(sim_root+'/'+f):
                    data_in = pl.read_csv(sim_root+'/'+f+'/'+f+data_name+'.csv',columns=list(range(1,n_col)),schema_overrides=dtype)
                    if f == sim_chunks[0]:
                        df = data_in
                    else:
                        df = df.join(data_in,on=['Gene','UpperLower'],suffix='_'+f)

        else:
            if os.path.isdir(sim_root+'/'+f):
                data_in = pl.read_csv(sim_root+'/'+f+'/'+f+data_name+'.csv',columns=list(range(1,n_col)),schema_overrides=dtype)
                data_in = data_in.with_columns(pl.col('Time') + tend)

                df = pl.concat((df,data_in),how='vertical')
                tend = df.select(pl.col('Time').max()).to_numpy()[0][0]

                if f != sim_chunks[-1]:
                    df = df.remove(pl.col('Time').eq(tend)) #We don't want the end of the folder and the start of the next one to be counted twice


    if 'mutation' in data_name:
        df=df.select(pl.col('Gene'),pl.sum_horizontal(pl.exclude('Gene')))
    if 'neutral' in data_name:
        df=df.select(pl.col('Gene'),pl.col('UpperLower'),pl.sum_horizontal(pl.exclude('Gene','UpperLower')))
    
    return df
    

    
    
def load_mats(root,sims):

    #root is the path to folder containing the simulation folders
    #sims are the sims to load matrices for

    mats = {'Sim':[],'Au':[],'Al':[],'S':[],'beta':[],'Gu':[],'Gl':[],'Sigmau':[],'Sigmal':[],'uimmutable':[],'limmutable':[]}
    
    for s in sims:
        chunks=False
        chunk0=''
        for f in sorted(os.listdir(root+'/'+s)):
            if os.path.isdir(root+'/'+s+'/'+f):
                chunk0=f
                chunks=True

        if chunks: 
            sim_mats = np.load(root+'/'+s+'/'+chunk0+'/'+chunk0+'mats.npz')
        else:
            sim_mats = np.load(root+'/'+s+'/'+s+'mats.npz')
            
        mats['Sim'] += [s]
        mats['Au'] += [sim_mats['Au']]
        mats['Al'] += [sim_mats['Al']]
        mats['S'] += [sim_mats['S']]
        mats['beta'] += [sim_mats['beta']]
        mats['Gu'] += [sim_mats['Gu']]
        mats['Gl'] += [sim_mats['Gl']]
        mats['Sigmau'] += [sim_mats['Sigmau']]
        mats['Sigmal'] += [sim_mats['Sigmal']]
        mats['uimmutable'] += [sim_mats['uimmutable']]
        mats['limmutable'] += [sim_mats['limmutable']]

    return mats


def load_ecolicore_mats(path,sims,exch=True,reg=False,R_bnamepath=''):
    #path is the path to the .xml file for e_coli_core

    model = cobra.io.read_sbml_model(path+'e_coli_core.xml')
    Au_,Al_,S_,Gu_,Gl_,beta,uimmutable,limmutable = cmsim_biggmatrices_double(model,exch=exch,regulated=reg,R_bnamepath=R_bnamepath,save=False)
    S = S_.to_numpy()
    Au = Au_.to_numpy()
    Al = Al_.to_numpy()
    Gu = Gu_.to_numpy()
    Gl = Gl_.to_numpy()
    Sigmau = np.eye(Gu.shape[1])
    Sigmal = np.eye(Gl.shape[1])

    mats = {'Sim':[],'Au':[],'Al':[],'S':[],'beta':[],'Gu':[],'Gl':[],'Sigmau':[],'Sigmal':[],'uimmutable':[],'limmutable':[]}
    for s in sims:
        mats['Sim'] += [s]
        mats['Au'] += [Au]
        mats['Al'] += [Al]
        mats['S'] += [S]
        mats['beta'] += [beta]
        mats['Gu'] += [Gu]
        mats['Gl'] += [Gl]
        mats['Sigmau'] += [Sigmau]
        mats['Sigmal'] += [Sigmal]
        mats['uimmutable'] += [uimmutable]
        mats['limmutable'] += [limmutable]

    return mats


def normalize_by_max(data,x,y,style,eps=1e-12):

    #Normalize 'y' by the maximum within specific values of 'style' and 'x'
    #data is all the data
    #x is the independent variable
    #y is the data that will be normalized by its maximum
    #style is an additionally independent variable that the maximum is taken within
    #eps is a small value to avoid dividing by 0

    return data.with_columns(pl.col(y) / (pl.col(y)+eps).max().over(style,x))

def normalize_by_norm(data,x,y,style,eps=1e-8):

    #Normalize 'y' by its norm within specific values of 'style' and 'x'
    #data is all the data
    #x is the independent variable
    #y is the data that will be normalized by its maximum
    #style is an additionally independent variable that the maximum is taken within
    #Don't normalize y when it's the (absolute) maximum is less than eps

    return data.with_columns((pl.col(y)/(eps+pl.col(y).dot(pl.col(y)).sqrt())*(pl.col(y).abs().max() >= eps)
                             +pl.col(y)*(pl.col(y).abs().max() < eps)).over(style,x))
    
    
def rolling_average_difference(data,y,hue,style,window=5000):

    #Calculate the average, rolling difference in y
    #Rolling differences are taken within hue and style
    #window is measured in number of data points
    
    return data.with_columns(pl.col(y).diff().rolling_mean(window_size=window).over(hue,style)).fill_null(np.nan)






def norm_bounds(ug,lg):
    #Takes in ug and lg and returns ug and lg such that the concatenated vector 
    #has norm 1
    
    norm = np.linalg.norm(np.hstack((ug,lg)))
    
    return ug/norm,lg/norm

def polars_matmul(A,B,B_rowlabel):
    #Calculate A @ B and push through the columns that were not involved
    #Assumes there is only 1 row_label column in B and the rest is data

    AatB = pl.from_numpy(A.select(pl.col(*B.select(pl.col(B_rowlabel)))).to_numpy() @ B.select(pl.exclude(B_rowlabel)).to_numpy())

    new_columns = B.columns.copy()
    new_columns.remove(B_rowlabel)
    
    AatB.columns = new_columns
    AatB = AatB.with_columns(*A.select(pl.exclude(*B.select(pl.col(B_rowlabel)))))
    return AatB
    

def makeG(Gunp,Glnp,u,l):

    #Gunp is a numpy array of Gu
    #Glnp is a numpy array of Gl
    #u is a polars dataframe with at least a column 'Gene' that sets the order of genes/column names of Gu
    #l is a polars dataframe with at least a column 'Gene' that sets the order of genes/column names of Gl
    Gu = pl.from_numpy(Gunp,schema=(u.get_column('Gene')).to_list()).\
            with_columns(pl.Series(['%d___u' % i for i in range(Gunp.shape[0])]).alias('Constraint'))
    Gl = pl.from_numpy(Glnp,schema=(l.get_column('Gene')).to_list()).\
            with_columns(pl.Series(['%d___l' % i for i in range(Glnp.shape[0])]).alias('Constraint'))
    
    rename=False
    if Gu.columns == Gl.columns:
        print('G matrix genes renamed')
        rename=True
        Gu = pl.from_numpy(Gunp,schema=((u.get_column('Gene')+'___gu').to_list())).\
                with_columns(pl.Series(['%d___u' % i for i in range(Gunp.shape[0])]).alias('Constraint'))
        Gl = pl.from_numpy(Glnp,schema=((l.get_column('Gene')+'___gl').to_list())).\
                with_columns(pl.Series(['%d___l' % i for i in range(Glnp.shape[0])]).alias('Constraint'))

    return Gu, Gl, rename


def makeA(Aunp,Alnp,f):

    Au = pl.from_numpy(Aunp,schema=(f.get_column('Reaction')).to_list()).\
            with_columns(pl.Series(['%d___u' % i for i in range(Aunp.shape[0])]).alias('Constraint'))
    Al = pl.from_numpy(-Alnp,schema=(f.get_column('Reaction')).to_list()).\
            with_columns(pl.Series(['%d___l' % i for i in range(Alnp.shape[0])]).alias('Constraint'))

    return Au, Al





def __evo_dir_and_SP__(s,t0_dir,tf_dir,t0_SP,tf_SP,SPu,SPl,ubound,lbound):

    if t0_dir is not None and tf_dir is not None:
        #Get the direction of evolution
        diff_ubound = ubound.filter(pl.col('Time').eq(t0_dir) | pl.col('Time').eq(tf_dir)).\
                            select(pl.col('Gene'),pl.col('Bound').diff().over('Gene','Sim')).drop_nulls()
        diff_lbound = lbound.filter(pl.col('Time').eq(t0_dir) | pl.col('Time').eq(tf_dir)).\
                            select(pl.col('Gene'),pl.col('Bound').diff().over('Gene','Sim')).drop_nulls()
    
        u_normed,l_normed = norm_bounds(diff_ubound['Bound'].to_numpy(),
                                       diff_lbound['Bound'].to_numpy())
    
        diff_ubound = diff_ubound.with_columns(pl.Series(u_normed).alias('Bound'),(pl.col('Gene')+'___u').alias('Gene'))
        diff_lbound = diff_lbound.with_columns(pl.Series(l_normed).alias('Bound'),(pl.col('Gene')+'___l').alias('Gene'))

    if t0_SP is not None and tf_SP is not None:
    
        #Get the shadow prices out
        SPu_sim = SPu.filter(pl.col('Time') >= t0_SP,pl.col('Time') <= tf_SP)\
                    .pivot(values='Lambda',on='Constraint')
        SPl_sim = SPl.filter(pl.col('Time') >= t0_SP,pl.col('Time') <= tf_SP)\
                    .pivot(values='Lambda',on='Constraint')


    if t0_dir is not None and tf_dir is not None and t0_SP is not None and tf_SP is not None:
        return diff_ubound,diff_lbound,SPu_sim,SPl_sim
    elif t0_dir is not None and tf_dir is not None:
        return diff_ubound,diff_lbound
    elif t0_SP is not None and tf_SP is not None:
        return SPu_sim,SPl_sim
    else:
        return None

def __parse_time_inputs__(t0_dir,tf_dir,t0_SP,tf_SP,ubound):
    
    if t0_dir is None:
        t0_dir_ = ubound.select(pl.min('Time')).to_numpy()[0][0]
    else:
        t0_dir_ = t0_dir
    if tf_dir is None:
        tf_dir_ = ubound.select(pl.max('Time')).to_numpy()[0][0]
    else:
        tf_dir_ = tf_dir
    if t0_SP is None:
        t0_SP_ = ubound.select(pl.min('Time')).to_numpy()[0][0]
    else:
        t0_SP_ = t0_SP
    if tf_SP is None:
        tf_SP_ = ubound.select(pl.max('Time')).to_numpy()[0][0]
    else:
        tf_SP_ = tf_SP

    return t0_dir_,tf_dir_,t0_SP_,tf_SP_



def calculate_shadow_prices_from_sim(ds,ubound,lbound,mats,t0=None,tf=None):


    s = ubound.item(0,'Sim')
    
    mats_idx = mats['Sim'].index(s)
    t0_,tf_,_,_ = __parse_time_inputs__(t0,tf,None,None,ubound)

    tvals = ubound.select(pl.col('Time')).unique(maintain_order=True)
    tvals = tvals.filter(pl.col('Time') >= t0_, pl.col('Time') <= tf_)
    tvals = tvals.gather_every(ds).to_numpy().flatten()

    
    all_FBAs = [FBA_gene(ubound.filter(pl.col('Time').eq(t))['Bound'].to_numpy(),
                         lbound.filter(pl.col('Time').eq(t))['Bound'].to_numpy(),
                         mats['Au'][mats_idx],mats['Al'][mats_idx],mats['S'][mats_idx],
                         mats['Gu'][mats_idx],mats['Gl'][mats_idx],mats['beta'][mats_idx],
                         return_lagrange=True) for t in tvals]


    SPu = pl.DataFrame(np.array([i[3] for i in all_FBAs]))
    SPu = SPu.with_columns(pl.Series(tvals).alias('Time'))
    SPu.columns = ['%d___u' % i for i in range(mats['Gu'][mats_idx].shape[0])] + ['Time']
    SPu = SPu.unpivot(value_name='Lambda',index='Time',variable_name='Constraint')
    SPu = SPu.with_columns(pl.Series([s]*SPu.shape[0]).alias('Sim')) 

    SPl = pl.DataFrame(np.array([i[4] for i in all_FBAs]))
    SPl = SPl.with_columns(pl.Series(tvals).alias('Time'))
    SPl.columns = ['%d___l' % i for i in range(mats['Gl'][mats_idx].shape[0])] + ['Time']
    SPl=SPl.unpivot(value_name='Lambda',index='Time',variable_name='Constraint')
    SPl = SPl.with_columns(pl.Series([s]*SPl.shape[0]).alias('Sim'))

    return SPu,SPl



def SP_on_simoutcome(ubound,lbound,SPu,SPl,mats,t0_dir=None,tf_dir=None,t0_SP=None,tf_SP=None):

    s = ubound.item(0,'Sim')

    SP_on_outcome = pl.DataFrame()
    
    mats_idx = mats['Sim'].index(s)
    t0_dir_,tf_dir_,t0_SP_,tf_SP_ = __parse_time_inputs__(t0_dir,tf_dir,t0_SP,tf_SP,ubound)

    diff_ubound,diff_lbound,SPu_sim,SPl_sim = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
                                                             t0_SP_,tf_SP_,
                                                             SPu,SPl,ubound,lbound)
    
    diff_lbound = diff_lbound.with_columns(-pl.col('Bound')) #Make positive for selective pressure calculation

    Gu,Gl,rename=makeG(mats['Gu'][mats_idx],mats['Gl'][mats_idx],diff_ubound,diff_lbound)

    #Polars doesn't have built-in matrix multiplication (no alignment), so need to do 
    SP_sim_on_u = polars_matmul(polars_matmul(SPu_sim,Gu,'Constraint'),diff_ubound,'Gene')
    SP_sim_on_l = polars_matmul(polars_matmul(SPl_sim,Gl,'Constraint'),diff_lbound,'Gene') 

    SP_sim = SP_sim_on_u.join(SP_sim_on_l,on=('Time','Sim'),suffix='___l').select(pl.col('Time'),pl.col('Sim'),(pl.col('Bound')+pl.col('Bound___l')).alias('Selective Pressure'))
    SP_on_outcome = pl.concat((SP_on_outcome,SP_sim),how='vertical')

    return SP_on_outcome



def SP_on_genes(ubound,lbound,SPu,SPl,mats,t0_SP=None,tf_SP=None):

    s = ubound.item(0,'Sim')

    #We don't use the direction, but we need the direction to set the order of the genes
    mats_idx = mats['Sim'].index(s)
    t0_dir=None; tf_dir=None
    t0_dir_,tf_dir_,t0_SP_,tf_SP_ = __parse_time_inputs__(t0_dir,tf_dir,t0_SP,tf_SP,ubound)
        
    diff_ubound,diff_lbound,SPu_sim,SPl_sim = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
                                                             t0_SP_,tf_SP_,
                                                             SPu,SPl,ubound,lbound)


    #Make G matrices as pandas dataframes so that the dataframes are aligned before multiplication
    Gu,Gl,rename=makeG(mats['Gu'][mats_idx],mats['Gl'][mats_idx],diff_ubound,diff_lbound)

    SPu_genes = polars_matmul(SPu_sim,Gu,'Constraint')
    SPl_genes = polars_matmul(SPl_sim,Gl,'Constraint')

    SP_genes = SPu_genes.join(SPl_genes,on=('Time','Sim'),suffix='___l')
        
    return SP_genes.unpivot(index=('Time','Sim'),value_name='Selective Pressure',variable_name='Gene')
    

def SP_on_genesvssim_std(ubound,lbound,SPu,SPl,mats,t0_dir=None,tf_dir=None,t0_SP=None,tf_SP=None):

    s = ubound.item(0,'Sim')


    SP_sims_stats = {'Sim':[],'Selective Pressure std':[],'Selective Pressure mean':[]}
    SP_genes_stats = {'Sim':[],'Gene':[],'Selective Pressure std':[],'Selective Pressure mean':[]}
    
        
    mats_idx = mats['Sim'].index(s)
    t0_dir_,tf_dir_,t0_SP_,tf_SP_ = __parse_time_inputs__(t0_dir,tf_dir,t0_SP,tf_SP,ubound)
        
    diff_ubound,diff_lbound,SPu_sim,SPl_sim = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
                                                             t0_SP_,tf_SP_,
                                                             SPu,SPl,ubound,lbound)
    
    diff_lbound = diff_lbound.with_columns(-pl.col('Bound')) #Make positive for selective pressure calculation


    Gu,Gl,rename=makeG(mats['Gu'][mats_idx],mats['Gl'][mats_idx],diff_ubound,diff_lbound)
    
    #Polars doesn't have built-in matrix multiplication (no alignment), so need to do 
    SP_sim_on_u = polars_matmul(polars_matmul(SPu_sim,Gu,'Constraint'),diff_ubound,'Gene')
    SP_sim_on_l = polars_matmul(polars_matmul(SPl_sim,Gl,'Constraint'),diff_lbound,'Gene') 

    SP_sim = SP_sim_on_u.join(SP_sim_on_l,on=('Time','Sim'),suffix='___l').select(pl.col('Time'),pl.col('Sim'),(pl.col('Bound')+pl.col('Bound___l')).alias('Selective Pressure'))
    
    SP_sims_stats['Sim'] += [s]
    SP_sims_stats['Selective Pressure std'] += [SP_sim.select(pl.std('Selective Pressure')).item()]
    SP_sims_stats['Selective Pressure mean'] += [SP_sim.select(pl.mean('Selective Pressure')).item()]


    SPu_genes = polars_matmul(SPu_sim,Gu,'Constraint')
    SPl_genes = polars_matmul(SPl_sim,Gl,'Constraint')

    SP_genes = SPu_genes.join(SPl_genes,on=('Time','Sim'),suffix='___l')
    
    SP_genes_stats['Sim'] += [s]*(SP_genes.shape[1]-2)
    SP_genes_stats['Gene'] += SP_genes.select(pl.all().exclude('Time','Sim')).columns
    SP_genes_stats['Selective Pressure std'] += list(SP_genes.select(pl.all().exclude('Time','Sim').std()).to_numpy()[0])
    SP_genes_stats['Selective Pressure mean'] += list(SP_genes.select(pl.all().exclude('Time','Sim').mean()).to_numpy()[0])

    return pl.DataFrame(SP_sims_stats),pl.DataFrame(SP_genes_stats)
    
def SP_on_simoutcome_std_vsnull(n_null,ubound,lbound,SPu,SPl,mats,t0_dir=None,tf_dir=None,t0_SP=None,tf_SP=None):

    s = ubound.item(0,'Sim')


    SP_stds = {'Sim':[],'Selective Pressure std':[]}
    
    null_stds = np.zeros((1,n_null))

    mats_idx = mats['Sim'].index(s)
    t0_dir_,tf_dir_,t0_SP_,tf_SP_ = __parse_time_inputs__(t0_dir,tf_dir,t0_SP,tf_SP,ubound)


    diff_ubound,diff_lbound,SPu_sim,SPl_sim = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
                                                             t0_SP_,tf_SP_,
                                                             SPu,SPl,ubound,lbound)
    
    diff_lbound = diff_lbound.with_columns(-pl.col('Bound')) #Make positive for selective pressure calculation


    Gu,Gl,rename=makeG(mats['Gu'][mats_idx],mats['Gl'][mats_idx],diff_ubound,diff_lbound)
    
    #Polars doesn't have built-in matrix multiplication (no alignment), so need to do 
    SP_sim_on_u = polars_matmul(polars_matmul(SPu_sim,Gu,'Constraint'),diff_ubound,'Gene')
    SP_sim_on_l = polars_matmul(polars_matmul(SPl_sim,Gl,'Constraint'),diff_lbound,'Gene') 

    SP_sim = SP_sim_on_u.join(SP_sim_on_l,on=('Time','Sim'),suffix='___l').select(pl.col('Time'),pl.col('Sim'),(pl.col('Bound')+pl.col('Bound___l')).alias('Selective Pressure'))
    
    
    SP_stds['Sim'] += [s]
    SP_stds['Selective Pressure std'] += [SP_sim.select(pl.std('Selective Pressure')).item()]
    

    
    for i_null in range(n_null):
        null_u,null_l = norm_bounds(np.random.rand(Gu.shape[1]-1),np.random.rand(Gl.shape[1]-1)) #-1 to not count constraint label
        null_u = pl.DataFrame({'Bound':null_u,'Gene':diff_ubound.select(pl.col('Gene'))})
        null_l = pl.DataFrame({'Bound':null_l,'Gene':diff_lbound.select(pl.col('Gene'))})

        SP_null_on_u = polars_matmul(polars_matmul(SPu_sim,Gu,'Constraint'),null_u,'Gene')
        SP_null_on_l = polars_matmul(polars_matmul(SPl_sim,Gl,'Constraint'),null_l,'Gene') 

        SP_null = SP_null_on_u.join(SP_null_on_l,on=('Time','Sim'),suffix='___l').select(pl.col('Time'),pl.col('Sim'),(pl.col('Bound')+pl.col('Bound___l')).alias('Selective Pressure'))
        null_stds[0,i_null] = SP_null.select(pl.std('Selective Pressure')).item()

    null_stds = pl.from_numpy(null_stds).with_columns(pl.lit(s).alias('Sim'))
    null_stds = null_stds.unpivot(value_name='Null Selective Pressure std',variable_name='num',index='Sim').drop('num')

    return pl.DataFrame(SP_stds),null_stds




def predict_z_by_projection(mats,flux):

    s = flux.item(0,'Sim')
    predictions = {'Sim':[],'Reaction':[],'Prediction - Flux':[]}

    rxn_names = flux.unique('Reaction',maintain_order=True).select(pl.col('Reaction')).to_series().to_list()
    mats_idx = mats['Sim'].index(s)
    S = mats['S'][mats_idx]
    beta = mats['beta'][mats_idx]

    z_proj = (np.eye(S.shape[1]) - S.T @ np.linalg.inv(S @ S.T) @ S) @ beta
    z_proj = z_proj/z_proj.max()

    predictions['Sim'] += [s]*len(z_proj)
    predictions['Reaction'] += rxn_names
    predictions['Prediction - Flux'] += list(z_proj)

    return pl.DataFrame(predictions)


def detect_constant_flux_direction_svd(flux,t0=None,tf=None,downsample=1):
    
    s = flux.item(0,'Sim')
    svd_singular_values = {'Sim':[],'Singular Values':[]}
    t0_,tf_,_,_ = __parse_time_inputs__(t0,tf,None,None,flux)

    Q = flux.select(pl.all().gather_every(downsample).over('Sim','Reaction',mapping_strategy='explode')).\
            filter(pl.col('Time') >= t0_,pl.col('Time') <= tf_).\
            pivot(on='Reaction',values='Flux',index=None).drop('Sim','Time').to_numpy()

    sing = np.linalg.svd(Q,compute_uv=False)

    svd_singular_values['Sim'] += [s]
    svd_singular_values['Singular Values'] += [sing]
        
    return pl.DataFrame(svd_singular_values)


def detect_constant_gene_direction_svd(ubound,lbound,t0=None,tf=None,downsample=1):
    
    s = ubound.item(0,'Sim')
    t0_,tf_,_,_ = __parse_time_inputs__(t0,tf,None,None,ubound)


    svd_singular_values = {'Sim':[],'Singular Values':[]}
    
        
    ugene_names = ubound.unique('Gene',maintain_order=True).select(pl.col('Gene')+'___u').to_series().to_list()
    lgene_names = lbound.unique('Gene',maintain_order=True).select(pl.col('Gene')+'___l').to_series().to_list()
   
    Qu = ubound.select(pl.all().gather_every(downsample).over('Sim','Gene',mapping_strategy='explode')).\
            filter(pl.col('Time') >= t0_,pl.col('Time') <= tf_).\
            pivot(on='Gene',values='Bound',index=None).drop('Sim','Time').to_numpy()
    Ql = lbound.select(pl.all().gather_every(downsample).over('Sim','Gene',mapping_strategy='explode')).\
            filter(pl.col('Time') >= t0_,pl.col('Time') <= tf_).\
            pivot(on='Gene',values='Bound',index=None).drop('Sim','Time').to_numpy()

    Q = np.concat((Qu,Ql),axis=1)
    
    sing = np.linalg.svd(Q,compute_uv=False)

    svd_singular_values['Sim'] += [s]
    svd_singular_values['Singular Values'] += [sing]

    return pl.DataFrame(svd_singular_values)


def detect_constant_selective_advantage(biomass,t0=None,tf=None,downsample=1):

    s = biomass.item(0,'Sim')
    rsquareds = {'Sim':[],'R2':[]}
    t0_,tf_,_,_ = __parse_time_inputs__(t0,tf,None,None,biomass)

    Q = biomass.select(pl.all().gather_every(downsample)).\
            filter(pl.col('Time') >= t0_,pl.col('Time') <= tf_)

    X=biomass.select(pl.col('Time')).to_numpy()
    y=biomass.select(pl.col('Biomass')).to_numpy()
    lr = LinearRegression()
    lr.fit(X=X,y=y)
    

    rsquareds['Sim'] += [s]
    rsquareds['R2'] += [lr.score(X=X,y=y)]
        
    return pl.DataFrame(rsquareds)


def predict_z_from_chainsample(mats,ubound,lbound,round_d=8,n_samps=1000,solver='SCIPY'
                               ,equal_zero=False,scale=1,alpha=0.001):

    s = ubound.item(0,'Sim')

    predictions = {'Sim':[],'Simulation-Prediction Correlation - Gene':[],
                   'Simulation-Prediction Correlation - Constraint':[],'Prediction':[],
                   '#Lambda Found':[],'Lambda Distribution':[]}
    
    mats_idx = mats['Sim'].index(s)        
    t0_dir_,tf_dir_,_,_ = __parse_time_inputs__(None,None,None,None,ubound)

    
    #Get out direction and SP
    diff_ubound,diff_lbound = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
                                                             None,None,
                                                             [],[],ubound,lbound)

    #Make G
    Gu,Gl,rename=makeG(mats['Gu'][mats_idx],mats['Gl'][mats_idx],diff_ubound,diff_lbound)

    #Prep for prediction (combine upper and lower selective pressure, make block G)
    G_block = pl.concat((Gu,Gl),how='diagonal').fill_null(0)
    diff_abound = pl.concat((diff_ubound,diff_lbound.with_columns(-pl.col('Bound'))),how='vertical')

    #Make fix_idx
    fix_idx=[mats['uimmutable'][mats_idx][i][0] for i in mats['uimmutable'][mats_idx]] \
                    + [mats['limmutable'][mats_idx][i][0]+Gu.shape[1] for i in mats['limmutable'][mats_idx]]

    #Do chain sample to generate random samples
    ug0 = np.ones(Gu.shape[1]-1)
    lg0 = -np.ones(Gl.shape[1]-1)
    lams=chain_sample(n_samps,mats['Au'][mats_idx],mats['Al'][mats_idx],
                      mats['S'][mats_idx],
                      Gu.select(pl.all().exclude('Constraint')).to_numpy(),Gl.select(pl.all().exclude('Constraint')).to_numpy(),
                      mats['beta'][mats_idx],ug0,lg0,scale=scale,fix_idx=fix_idx)
    
    unique_chain,counts_chain = np.unique(lams.round(round_d),axis=0,return_counts=True)
    counts_argsort = np.argsort(counts_chain)[::-1]

    p=len(counts_chain[counts_chain/sum(counts_chain) > alpha])
    z_chain,fail_flag=cm_cutoff(unique_chain,counts_argsort,p,
                                G_block.select(pl.all().exclude('Constraint')).to_numpy(),solver=solver,equal_zero=equal_zero)

    simpregene_corr = np.corrcoef(z_chain,diff_abound.get_column('Bound').to_numpy().reshape(-1))[0,1]
    simprecon_corr = np.corrcoef(G_block.select(pl.all().exclude('Constraint')).to_numpy() @ z_chain,
                                 G_block.select(pl.all().exclude('Constraint')).to_numpy() @ diff_abound.get_column('Bound').to_numpy())[0,1]

    predictions['Sim'] += [s]
    predictions['Simulation-Prediction Correlation - Gene'] += [simpregene_corr]
    predictions['Simulation-Prediction Correlation - Constraint'] += [simprecon_corr]
    predictions['Prediction'] += [z_chain]
    predictions['#Lambda Found'] += [len(counts_chain)]
    predictions['Lambda Distribution'] += [counts_chain[counts_argsort]/sum(counts_chain)]

    return pl.DataFrame(predictions)



def predict_z_from_simSP(ubound,lbound,SPu,SPl,mats,t0_dir=None,tf_dir=None,t0_SP=None,tf_SP=None,
                             round_d=8,lam_idx=0,solver='SCS',equal_zero=False):
    
    s = ubound.item(0,'Sim')

    predictions = {'Sim':[],'Simulation-Prediction Correlation - Gene':[],'Simulation-Prediction Correlation - Constraint':[],'Prediction':[]}
    
    mats_idx = mats['Sim'].index(s)
    
    t0_dir_,tf_dir_,t0_SP_,tf_SP_ = __parse_time_inputs__(t0_dir,tf_dir,t0_SP,tf_SP,ubound)
    
    #Get out direction and SP
    diff_ubound,diff_lbound,SPu_sim,SPl_sim = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
                                                             t0_SP_,tf_SP_,
                                                             SPu,SPl,ubound,lbound)

    #Make G
    Gu,Gl,rename=makeG(mats['Gu'][mats_idx],mats['Gl'][mats_idx],diff_ubound,diff_lbound)


    #Prep for prediction (combine upper and lower selective pressure, make block G)
    SP = SPu_sim.join(SPl_sim,on=('Time','Sim'))
    G_block = pl.concat((Gu,Gl),how='diagonal').fill_null(0)
    diff_abound = pl.concat((diff_ubound,diff_lbound.with_columns(-pl.col('Bound'))),how='vertical')

    
    #Make fix_idx
    fix_idx=[mats['uimmutable'][mats_idx][i][0] for i in mats['uimmutable'][mats_idx]] \
                    + [mats['limmutable'][mats_idx][i][0]+Gu.shape[1]-1 for i in mats['limmutable'][mats_idx]]

    unique_lam = np.unique(SP.select(pl.col(G_block.get_column('Constraint'))).to_numpy().round(round_d),axis=0)
    D = Di(unique_lam,lam_idx)
    _,z = find_cm(unique_lam,G_block.select(pl.all().exclude('Constraint')).to_numpy(),
                      D,fix_idx=fix_idx,lam_idx=lam_idx,solver=solver)

    simpregene_corr = np.corrcoef(z,diff_abound.get_column('Bound').to_numpy().reshape(-1))[0,1]
    simprecon_corr = np.corrcoef(G_block.select(pl.all().exclude('Constraint')).to_numpy() @ z,
                                 G_block.select(pl.all().exclude('Constraint')).to_numpy() @ diff_abound.get_column('Bound').to_numpy())[0,1]

    predictions['Sim'] += [s]
    predictions['Simulation-Prediction Correlation - Gene'] += [simpregene_corr]
    predictions['Simulation-Prediction Correlation - Constraint'] += [simprecon_corr]
    predictions['Prediction'] += [z]

    return pl.DataFrame(predictions)



def genes_for_reactions(mats,flux,other=False):

    #Assumes Au/Al are identity (except for biomass reaction)

    s = flux.item(0,'Sim')

    g_on_G = {'Sim':[],'Reaction':[],'Side':[],'Genes':[]}

    rxn_names = flux.unique('Reaction',maintain_order=True).select(pl.col('Reaction')).to_series().to_list()
    mats_idx = mats['Sim'].index(s)
    Gu = mats['Gu'][mats_idx]
    Gl = mats['Gl'][mats_idx]

    g_on_G['Sim'] += [s]*len(rxn_names)
    g_on_G['Reaction'] += rxn_names
    g_on_G['Side'] += ['Upper']*len(rxn_names)
    g_on_G['Genes'] += list((Gu != 0).sum(axis=1)) + [0] #Biomass reaction has no genes

    g_on_G['Sim'] += [s]*len(rxn_names)
    g_on_G['Reaction'] += rxn_names
    g_on_G['Side'] += ['Lower']*len(rxn_names)
    g_on_G['Genes'] += list((Gl != 0).sum(axis=1)) + [0] #Biomass reaction has no genes

    return pl.DataFrame(g_on_G)


def other_genes_for_same_reaction(ubound,lbound,mats):

    s = ubound.item(0,'Sim')
    mats_idx = mats['Sim'].index(s)
    og_on_G = {'Sim':[],'Gene':[],'Other Genes':[]}


    Gu = mats['Gu'][mats_idx]
    Gl = mats['Gl'][mats_idx]


    Gu,Gl,rename = makeG(Gu,Gl,ubound.filter(pl.col('Time').eq(pl.col('Time').max())),
                          lbound.filter(pl.col('Time').eq(pl.col('Time').max())))
    
    og_on_G['Sim'] += [s]*len(Gu.select(pl.selectors.numeric()).columns)
    og_on_G['Gene'] += Gu.select(pl.selectors.numeric()).columns
    og = np.zeros((1,Gu.shape[1]-1))
    for i_g in range(Gu.shape[0]):
        row = Gu.select(pl.selectors.numeric())[i_g,:].to_numpy()
        og += row*(row.sum()-1)

    og_on_G['Other Genes'] += list(og[0])
    
    
    og_on_G['Sim'] += [s]*len(Gl.select(pl.selectors.numeric()).columns)
    og_on_G['Gene'] += Gl.select(pl.selectors.numeric()).columns
    og = np.zeros((1,Gu.shape[1]-1))
    for i_g in range(Gl.shape[0]):
        row = Gl.select(pl.selectors.numeric())[i_g,:].to_numpy()
        og += row*(row.sum()-1)
    og_on_G['Other Genes'] += list(og[0])


    return pl.DataFrame(og_on_G)

def constraints_for_reactions(mats,flux):

    #Assumes Gu/Gl are identity
    
    s = flux.item(0,'Sim')

    cons_on_A = {'Sim':[],'Reaction':[],'Side':[],'Constraints':[]}

    rxn_names = flux.unique('Reaction',maintain_order=True).select(pl.col('Reaction')).to_series().to_list()
    mats_idx = mats['Sim'].index(s)
    Au = mats['Au'][mats_idx]
    Al = mats['Al'][mats_idx]

    cons_on_A['Sim'] += [s]*Au.shape[1]
    cons_on_A['Reaction'] += rxn_names
    cons_on_A['Side'] += ['Upper']*Au.shape[1]
    cons_on_A['Constraints'] += list((Au != 0).sum(axis=0))

    cons_on_A['Sim'] += [s]*Al.shape[1]
    cons_on_A['Reaction'] += rxn_names
    cons_on_A['Side'] += ['Lower']*Al.shape[1]
    cons_on_A['Constraints'] += list((Al != 0).sum(axis=0))

    return pl.DataFrame(cons_on_A)

    
def second_sv_analysis(flux,ubound,lbound,mats,downsample=1):
   
    s = ubound.item(0,'Sim')

    #Assumes Au/Al are identity like

    singular_values = {'Sim':[],'SV Data':[],'SVs':[]}
    second_singular_vectors = {'Sim':[],'Gene':[],'2nd Right Singular Value Projection':[]}
    
    mats_idx = mats['Sim'].index(s)
    S = mats['S'][mats_idx]
    Au = mats['Au'][mats_idx]
    Al = mats['Al'][mats_idx]
    Gu = mats['Gu'][mats_idx]
    Gl = mats['Gl'][mats_idx]

    Au,Al = makeA(Au,Al,flux.filter(pl.col('Time').eq(pl.col('Time').max())))
    Gu,Gl,rename = makeG(Gu,Gl,ubound.filter(pl.col('Time').eq(pl.col('Time').max())),
                          lbound.filter(pl.col('Time').eq(pl.col('Time').max())))
    
    A_block = pl.concat((Au,Al),how='vertical')
    G_block = pl.concat((Gu,Gl),how='diagonal').fill_null(0)
    
    #Identify upper/lower genes based on direction of flux

    end_flux = flux.filter(pl.col('Time').eq(pl.col('Time').max())).select('Flux','Reaction')
    end_constraint = polars_matmul(A_block,end_flux,'Reaction')
    end_constraint_mask = end_constraint.with_columns(pl.col('Flux') > 0)

    A_active = end_constraint_mask.join(A_block,on='Constraint').filter(pl.col('Flux')).drop('Flux')
    G_block_masked = end_constraint_mask.join(G_block,on='Constraint').filter(pl.col('Flux'))

    active_genes = [col.name for col in G_block_masked.select(pl.selectors.numeric().sum() > 0) if col.item()]
    inactive_genes = [col.name for col in G_block_masked.select(pl.selectors.numeric().sum() > 0) if not col.item()]

    G_active = G_block_masked.drop(inactive_genes).drop('Flux')


    #Perform SVD on 'active genes'
        #First need to assemble time x gene matrices and time x flux matrices
        #But the gene matrix needs to be only with active genes
    Qu = ubound.select(pl.all().gather_every(downsample).over('Sim','Gene',mapping_strategy='explode'))
    Ql = lbound.select(pl.all().gather_every(downsample).over('Sim','Gene',mapping_strategy='explode'))
    
    if rename:
        Qu = Qu.with_columns(pl.col('Gene') + '___gu')
        Ql = Ql.with_columns(pl.col('Gene') + '___gl')

    Qu = Qu.pivot(on='Gene',values='Bound',index=None).drop('Sim','Time')
    Ql = Ql.pivot(on='Gene',values='Bound',index=None).drop('Sim','Time')

    upper_active = []
    lower_active = []
    for _gene_ in active_genes:
        if _gene_ in Qu.columns:
            upper_active += [_gene_]
        if _gene_ in Ql.columns:
            lower_active += [_gene_]

    if len(upper_active) + len(lower_active) != len(active_genes):
        print('SOMETHING WENT WRONG WHEN ALLOCATING GENES TO UPPER AND LOWER')
            
    
    Qgene = pl.concat((Qu.select(upper_active),Ql.select(lower_active)),how='horizontal')
    active_genes = Qgene.columns #Rewrite this list to make sure we have the order correct
    Qgene = Qgene.to_numpy()
    Qflux = flux.select(pl.all().gather_every(1).over('Sim','Reaction',mapping_strategy='explode')).\
            pivot(on='Reaction',values='Flux',index=None).drop('Sim','Time').to_numpy()

    uflux,sflux,vtflux = np.linalg.svd(Qflux,full_matrices=False)
    ugene,sgene,vtgene = np.linalg.svd(Qgene,full_matrices=False)

    #Partition second right singular vector into gene and null S variability

    #Need to convert to numpy arrays, but select by active genes to ensure that the singular vector and matrix are aligned
    G_ = G_active.select(active_genes).to_numpy()
    A_ = A_active.drop(pl.col('Constraint')).to_numpy()
    S_ = S
    
    u_S,s_S,vt_S = np.linalg.svd(S)
    N_S  = vt_S[len(s_S[s_S>0]):,:].T
    
    vp = vtgene[1,:].T
    x = cp.Variable(len(vp))
    obj = cp.Minimize(cp.sum_squares(x))
    con = [G_ @ x == A_ @ N_S @ np.linalg.pinv(A_ @ N_S) @ G_ @ vp]
    prob = cp.Problem(obj,con)
    prob.solve()

    singular_values['Sim'] += [s]
    singular_values['SV Data'] += ['Flux']
    singular_values['SVs'] += [sflux]
    singular_values['Sim'] += [s]
    singular_values['SV Data'] += ['Gene']
    singular_values['SVs'] += [sgene]

    second_singular_vectors['Sim'] += [s]*len(active_genes)
    second_singular_vectors['Gene'] += active_genes
    second_singular_vectors['2nd Right Singular Value Projection'] += list(x.value-vp)

    return pl.DataFrame(singular_values),pl.DataFrame(second_singular_vectors)




def calculate_bound_direction_variability(bound,ds,num_offset):
    
    return bound.select(pl.all().gather_every(ds,offset=num_offset*ds).over('Gene',mapping_strategy='explode')).\
        with_columns((pl.col('Bound').diff()).over('Gene')).drop_nulls().\
        with_columns((pl.col('Bound')/((pl.col('Bound')**2).sum().sqrt())).over('Time')).\
        group_by('Gene',maintain_order=True).agg(pl.col('Bound').std()/pl.col('Bound').mean())

def calculate_flux_direction_variability(flux,ds,num_offset):
    
    return flux.select(pl.all().gather_every(ds,offset=num_offset*ds).over('Reaction','Sim',mapping_strategy='explode')).\
        with_columns((pl.col('Flux').diff()).over('Reaction','Sim')).drop_nulls().\
        with_columns((pl.col('Flux')/((pl.col('Flux')**2).sum().sqrt())).over('Time','Sim')).\
        group_by(['Reaction','Sim'],maintain_order=True).agg(pl.col('Flux').std()/pl.col('Flux').mean())

                                         

def compare_direction_variability(flux,ubound,lbound,mats,ds,num_offset,t0_dir=None,tf_dir=None):

    s = ubound.item(0,'Sim')

    mats_idx = mats['Sim'].index(s)
    Au = mats['Au'][mats_idx]
    Al = mats['Al'][mats_idx]
    Gu = mats['Gu'][mats_idx]
    Gl = mats['Gl'][mats_idx]
    
    t0_dir_,tf_dir_,t0_SP_,tf_SP_ = __parse_time_inputs__(t0_dir,tf_dir,None,None,ubound)
        
    #diff_ubound,diff_lbound = __evo_dir_and_SP__(s,t0_dir_,tf_dir_,
    #                                                         None,None,
    #                                                         [],[],ubound,lbound)


    
    Au,Al = makeA(Au,Al,flux.filter(pl.col('Time').eq(pl.col('Time').max())))
    Gu,Gl,rename = makeG(Gu,Gl,ubound.filter(pl.col('Time').eq(pl.col('Time').max())),
                          lbound.filter(pl.col('Time').eq(pl.col('Time').max())))
    
    A_block = pl.concat((Au,Al),how='vertical')
    G_block = pl.concat((Gu,Gl),how='diagonal').fill_null(0)

    flux_cv = calculate_flux_direction_variability(flux,ds,num_offset)

    Aatfluxfinal = polars_matmul(A_block,flux.filter(pl.col('Time').eq(pl.col('Time').max())).drop('Sim','Time'),'Reaction')
    GatAatfluxfinal = polars_matmul(Aatfluxfinal.with_columns(pl.lit('*').alias('Temp')).pivot(on='Constraint',index='Temp'),G_block,'Constraint')
    GatAatfluxfinal = GatAatfluxfinal.drop('Temp').unpivot(variable_name='Gene',value_name='Final Flux')

    
    Aatfluxcv = polars_matmul(A_block,flux_cv.drop('Sim'),'Reaction')
    GatAatfluxcv = polars_matmul(Aatfluxcv.with_columns(pl.lit('*').alias('Temp')).pivot(on='Constraint',index='Temp'),G_block,'Constraint')
    GatAatfluxcv = GatAatfluxcv.drop('Temp').unpivot(variable_name='Gene',value_name='Flux CV')

    ubound_cv = calculate_bound_direction_variability(ubound,ds,num_offset) # We keep the bounds separate so that we don't normalize the bounds
    lbound_cv = calculate_bound_direction_variability(lbound.with_columns(-pl.col('Bound')),ds,num_offset) #Negate lower bounds so we have positive CV

    if rename:
        return GatAatfluxfinal,GatAatfluxcv,ubound_cv.with_columns(pl.col('Gene') + '___gu'),lbound_cv.with_columns(pl.col('Gene') + '___gl')
    else:
        return GatAatfluxfinal,GatAatfluxcv,ubound_cv,lbound_cv

    
    

    





