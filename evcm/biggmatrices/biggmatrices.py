import numpy as np
import sympy as sym
import pandas as pd
import cobra
import time
import warnings





def cmsim_biggmatrices_double(model,exch=True,save=False,regulated=False,R_bnamepath=''):
    
    nr = len(model.reactions) #number of reactions
    nm = len(model.metabolites) #number of metabolites
    ng = len(model.genes) #number of genes
    
    #List of the names of all the key objects
    genes = model.genes.list_attr("id")
    if 's0001' in genes:
        genes.remove('s0001')
        ng = len(genes)
    rxns = model.reactions.list_attr("id")
    metabolites = model.metabolites.list_attr("id")
    
    #Find the fitness reaction name
    fitness_rxn_id = cobra.util.solver.linear_reaction_coefficients(model)
    fitness_rxn = list(fitness_rxn_id.keys())[0]
    fitness_rxn_id = fitness_rxn.id
    # print(fitness_rxn_id)
    
    default_bound = fitness_rxn.upper_bound
    
    #Initialize matrices (as much as possible)
    Gu_df = pd.DataFrame(columns=genes,dtype=float)
    Gl_df = pd.DataFrame(columns=genes,dtype=float)
    Au_df = pd.DataFrame(columns=rxns,dtype=float)
    Al_df = pd.DataFrame(columns=rxns,dtype=float)
    S_df = pd.DataFrame(data=np.zeros((nm,nr)),index=metabolites,columns=rxns)
    
    uimm_dict = dict()
    limm_dict = dict()
    
    
    
    for r,r_i in zip(model.reactions,rxns):
        #cycle over cobra reactions (Reaction) and the names of reactions (r_i)
        
        isuimm = False
        if r.upper_bound != default_bound and r.upper_bound != 0:
            print(r_i,r.upper_bound)
            isuimm = True
        
        islimm = False
        if r.lower_bound != -default_bound and r.lower_bound != 0:
            print(r_i,r.lower_bound)
            islimm = True
    
        #Update S
        for m,s in r.metabolites.items():
            S_df.loc[m.id,r.id] = s
            
        
        #Update A and G
        s = r.gpr.as_symbolic()
        # print(r_i)
        
        if exch:
            Au_r,Al_r,Gu_r,Gl_r,uimm_r,limm_r = AG_reaction_exch(r,r_i,s,isuimm,islimm,genes,rxns)
        else:
            Au_r,Al_r,Gu_r,Gl_r,uimm_r,limm_r = AG_reaction_noexch(r,r_i,s,isuimm,islimm,genes,rxns,abs(S_df.loc[:,r_i]).sum())
        

        
        Gu_df = pd.concat((Gu_df,Gu_r),axis=0)
        Gl_df = pd.concat((Gl_df,Gl_r),axis=0)
        Au_df = pd.concat((Au_df,Au_r),axis=0)
        Al_df = pd.concat((Al_df,Al_r),axis=0)
        uimm_dict = uimm_dict | uimm_r
        limm_dict = limm_dict | limm_r
            

    #When we concatenate with a new gene, the rest of the column is set to NaN, so set to 0
    Gu_df[Gu_df.isnull()] = 0
    Gl_df[Gl_df.isnull()] = 0
    



    Gu_df = Gu_df.reset_index(drop=True)
    Au_df = Au_df.reset_index(drop=True)
    Gl_df = Gl_df.reset_index(drop=True)
    Al_df = Al_df.reset_index(drop=True)

    Gu_df = Gu_df.drop(index=Au_df.index[Au_df.loc[:,fitness_rxn_id] != 0]) #Remove corresponding row in G matric
    Au_df = Au_df.drop(index=Au_df.index[Au_df.loc[:,fitness_rxn_id] != 0]) #Remove constraint on fitness reaction
    Gl_df = Gl_df.drop(index=Al_df.index[Al_df.loc[:,fitness_rxn_id] != 0]) #Remove corresponding row in G matric
    Al_df = Al_df.drop(index=Al_df.index[Al_df.loc[:,fitness_rxn_id] != 0]) #Remove constraint on fitness reaction
    
    Gu_df = Gu_df.drop(columns=fitness_rxn_id+"_gene") #Remove biomass gene
    Gl_df = Gl_df.drop(columns=fitness_rxn_id+"_gene") #Remove biomass gene
    
    
    if not exch:
        Gu_df = Gu_df.drop(index=Gu_df.index[Au_df.sum(axis=1) == 0])
        Gl_df = Gl_df.drop(index=Gl_df.index[Al_df.sum(axis=1) == 0])
        Au_df = Au_df.drop(index=Au_df.index[Au_df.sum(axis=1) == 0])
        Al_df = Al_df.drop(index=Al_df.index[Al_df.sum(axis=1) == 0])
        
        #Remove genes from G that don't map to anything:
        Gu_df = Gu_df.drop(columns=Gu_df.columns[Gu_df.sum(axis=0) == 0])
        Gl_df = Gl_df.drop(columns=Gl_df.columns[Gl_df.sum(axis=0) == 0])
        
        

        
    
    if len(uimm_dict) > 0:
        cols = list(Gu_df.columns)
        for g in uimm_dict:
            uimm_dict[g] = (cols.index(g),uimm_dict[g][1])
            
    if len(limm_dict) > 0:
        cols = list(Gl_df.columns)
        for g in limm_dict:
            limm_dict[g] = (cols.index(g),limm_dict[g][1])
            
    beta = np.zeros(nr)
    beta[rxns.index(fitness_rxn_id)] = 1
    
    
    
    for g in uimm_dict.keys():
        if Gu_df.loc[:,g].sum() > 1:
            warnings.warn('Setting an immutable upper bound on a gene {} that corresponds to more than 1 constraint. Double check that this does not generate two different immutable values.'.format(g))
    
    for g in limm_dict.keys():
        if Gl_df.loc[:,g].sum() > 1:
            warnings.warn('Setting an immutable lower bound on a gene {} that corresponds to more than 1 constraint. Double check that this does not generate two different immutable values.'.format(g))
        
    
    
    uimm_df = pd.DataFrame(uimm_dict,index=['Index','Bound'])
    limm_df = pd.DataFrame(limm_dict,index=['Index','Bound'])
    
    
    if regulated and R_bnamepath != '':
        
        R_bnames = pd.read_csv(R_bnamepath,index_col=0)
        
        Ru = R_genes(Gu_df.columns,R_bnames)
        Gu_df = pd.concat((Gu_df,Gu_df @ Ru),axis=1)
        
        if len(np.unique(Gu_df.columns)) < len(Gu_df.columns):
            Gu_df = Gu_df.T.groupby(level=0).sum().T
        
        Rl = R_genes(Gl_df.columns,R_bnames)
        Gl_df = pd.concat((Gl_df,Gl_df @ Rl),axis=1)
        
        if len(np.unique(Gl_df.columns)) < len(Gl_df.columns):
            Gl_df = Gl_df.T.groupby(level=0).sum().T
    
    if save:
        timestr = time.strftime("%Y%m%d")
        S_df.to_csv("S_DataFrame_{}_{}.csv".format(model.id,timestr))
        Au_df.to_csv("Au_DataFrame_{}_{}.csv".format(model.id,timestr))
        Al_df.to_csv("Al_DataFrame_{}_{}.csv".format(model.id,timestr))
        uimm_df.to_csv("UImm_DataFrame_{}_{}.csv".format(model.id,timestr))
        limm_df.to_csv("LImm_DataFrame_{}_{}.csv".format(model.id,timestr))
        Gu_df.to_csv("Gu_DataFrame_{}_{}.csv".format(model.id,timestr))
        Gl_df.to_csv("Gl_DataFrame_{}_{}.csv".format(model.id,timestr))
        
    return Au_df,Al_df,S_df,Gu_df,Gl_df,beta,uimm_dict,limm_dict


#Every AG_reaction_* returns Au,Al,Gu,Gl for a specific reaction (rows of Au/Al
# and Gu/Gl that correspond to that reaction), as well as any immutable bounds

def AG_reaction_exch(r,r_i,s,isuimm,islimm,genes,rxns):
    
    Gu_df_r = pd.DataFrame(columns=genes,dtype=float)
    Gl_df_r = pd.DataFrame(columns=genes,dtype=float)
    A_df_r = pd.DataFrame(columns=rxns,dtype=float)
    #Every reaction has the upper and lower constraints so we can just make A

    uimm_dict = dict()
    limm_dict = dict()

        
    try:
        s = sym.to_cnf(s,simplify=True,force=False)
    except:
        # print(s)
        s = sym.to_cnf(s)
    s = str(s)
    
    if s != "":
        
        for n in s.split('&'): #Individual OR statements are separated by ANDs because of CNF
            
            n = n.strip(" ()") #Remove the junk

            j = len(Gu_df_r.index) #Need to append to the end of the dataframe
            
            Gu_df_r.loc[j,:] = 0
            Gl_df_r.loc[j,:] = 0
            A_df_r.loc[j,:] = 0
            A_df_r.loc[j,r_i] = 1 #Put a 1 in the current row, and in the correct reaction
            
           
            if '|' in n: #Cycle over individual genes in the OR
                for m in n.split('|'):
                    
                    if m.strip(" ") == "s0001": #'Gene is not mapped to a genome annotation' (Diffusion transport)
                        m = r_i+"_s"
                        Gu_df_r.loc[:,m] = 0 #Make our own gene
                        Gl_df_r.loc[:,m] = 0
                        
                    
                    if r.upper_bound != 0:
                        Gu_df_r.loc[j,m.strip(" ")] = 1 #Put a 1 in the correct gene row for the upper bound
                    
                    if r.lower_bound != 0:
                        Gl_df_r.loc[j,m.strip(" ")] = 1 #Put a 1 in the correct gene row for the lower bound
                                            
                    
            else:
                
                if n == "s0001": #'Gene is not mapped to a genome annotation' (Diffusion transport)
                    n = r_i+"_s"
                    Gu_df_r.loc[:,n] = 0 #Make our own gene
                    Gl_df_r.loc[:,n] = 0
                    
                if r.upper_bound != 0:
                    Gu_df_r.loc[j,n] = 1
                if r.lower_bound != 0:
                    Gl_df_r.loc[j,n] = 1
                    
            if isuimm:
                genes_rxn = Gu_df_r.loc[j,:] == 1
                ngene_rxn = Gu_df_r.loc[j,:].sum()
                
                if ngene_rxn > 1:
                    warnings.warn("Setting an immutable upper bound for reaction {} that has more than 1 corresponding gene. Double check.".format(r_i))
                
                cols = list(Gu_df_r.columns)
                for g in Gu_df_r.columns[genes_rxn]:
                    uimm_dict[g] = (cols.index(g),r.upper_bound/ngene_rxn)
                    
            if islimm:
                genes_rxn = Gl_df_r.loc[j,:] == 1
                ngene_rxn = Gl_df_r.loc[j,:].sum()
                
                if ngene_rxn > 1:
                    warnings.warn("Setting an immutable lower bound for reaction {} that has more than 1 corresponding gene. Double check.".format(r_i))
                
                cols = list(Gl_df_r.columns)
                for g in Gl_df_r.columns[genes_rxn]:
                    limm_dict[g] = (cols.index(g),-r.lower_bound/ngene_rxn)

                    
            
                

    else: #No gene information was included, so need to make our own gene
    
        j = len(Gu_df_r.index) #Need to append to end of the dataframe
        Gu_df_r.loc[:,r_i+"_gene"] = [0] #Make our own gene
        Gu_df_r[Gu_df_r.isnull()] = 0
        Gl_df_r.loc[:,r_i+"_gene"] = [0] #Make our own gene
        Gl_df_r[Gl_df_r.isnull()] = 0
        
        Gu_df_r.loc[j,:] = 0 #Fill in the new  upper bound row for the constraint with 0s
        Gl_df_r.loc[j,:] = 0
        A_df_r.loc[j,:] = 0
        A_df_r.loc[j,r_i] = 1 #Put the 1 in the correct spot
        
        if r.upper_bound != 0:
            Gu_df_r.loc[j,r_i+"_gene"] = 1
        if r.lower_bound != 0:    
            Gl_df_r.loc[j,r_i+"_gene"] = 1
        
            
        if isuimm:
            cols = list(Gu_df_r.columns)
            uimm_dict[r_i+"_gene"] = (cols.index(r_i+"_gene"),r.upper_bound)
            
        if islimm:
            cols = list(Gl_df_r.columns)
            limm_dict[r_i+"_gene"] = (cols.index(r_i+"_gene"),-r.lower_bound)
        

            

                             
    return A_df_r,A_df_r,Gu_df_r,Gl_df_r,uimm_dict,limm_dict

def AG_reaction_noexch(r,r_i,s,isuimm,islimm,genes,rxns,nummet):

    Gu_df_r = pd.DataFrame(columns=genes,dtype=float)
    Gl_df_r = pd.DataFrame(columns=genes,dtype=float)
    Au_df_r = pd.DataFrame(columns=rxns,dtype=float)
    Al_df_r = pd.DataFrame(columns=rxns,dtype=float)
    #Every reaction has the upper and lower constraints so we can just make A

    uimm_dict = dict()
    limm_dict = dict()
    
    
    try:
        s = sym.to_cnf(s,simplify=True,force=False)
    except:
        # print(s)
        s = sym.to_cnf(s)
    s = str(s)
    
    
    if 's0001' in s: #Diffusion transport - treat like an exchange reaction
            
        ju = len(Gu_df_r.index) #Need to append to the end of the dataframe
        jl = len(Gl_df_r.index)
        
        Gu_df_r.loc[ju,:] = 0 #Add new rows - we have some new constraint
        Gl_df_r.loc[jl,:] = 0
        Au_df_r.loc[ju,:] = 0
        Al_df_r.loc[jl,:] = 0
        
        m = r_i+"_s"
        Gu_df_r.loc[:,m] = 0 #Make our own gene - only so that number matches if exch=True
        Gl_df_r.loc[:,m] = 0
        
        if r.upper_bound == 0: #If upper bound is 0, then we need a constraint
            Au_df_r.loc[ju,r_i] = 1
        
        if isuimm: #If a fixed bound, need to incorporate gene
            Au_df_r.loc[ju,r_i] = 1
            Gu_df_r.loc[ju,m.strip(" ")] = 1
        
        if r.lower_bound == 0: #If lower bound is 0, then we need a constraint
            Al_df_r.loc[jl,r_i] = 1
            
        if islimm:
            Al_df_r.loc[jl,r_i] = 1
            Gl_df_r.loc[jl,m.strip(" ")] = 1
        
    
    elif s != "":    
            
        for n in s.split('&'): #Individual OR statements are separated by ANDs because of CNF
            
            n = n.strip(" ()") #Remove the junk

            ju = len(Gu_df_r.index) #Need to append to the end of the dataframe
            jl = len(Gl_df_r.index)
            
            Gu_df_r.loc[ju,:] = 0 #Add new rows - we have some new constraint
            Gl_df_r.loc[jl,:] = 0
            Au_df_r.loc[ju,:] = 0
            Al_df_r.loc[jl,:] = 0
            
           
            if '|' in n: #Cycle over individual genes in the OR
                for m in n.split('|'):
                    
                    if m.strip(" ") == "s0001": #'Gene is not mapped to a genome annotation' (Diffusion transport)
                        m = r_i+"_s"
                        Gu_df_r.loc[:,m] = 0 #Make our own gene
                        Gl_df_r.loc[:,m] = 0
                        #This if statement will not be reached if exch=False
                        
                    if nummet != 1: #Not an exchange reaction, so add genes
                        Au_df_r.loc[ju,r_i] = 1 #Internal reactions always have some type of constraint on both upper and lower bounds
                        Al_df_r.loc[jl,r_i] = 1
                        
                        if r.upper_bound != 0:
                            Gu_df_r.loc[ju,m.strip(" ")] = 1 #Put a 1 in the correct gene row for the upper bound

                        if r.lower_bound != 0:
                            Gl_df_r.loc[jl,m.strip(" ")] = 1 #Put a 1 in the correct gene row for the lower bound

                            
                    elif nummet == 1:
                        
                        if r.upper_bound == 0: #If upper bound is 0, then this is a fixed bound because it is an exchange reaction
                            Au_df_r.loc[ju,r_i] = 1
                            
                        if isuimm: #If a fixed bound, need to incorporate gene
                            Au_df_r.loc[ju,r_i] = 1
                            Gu_df_r.loc[ju,m.strip(" ")] = 1
                        
                        if r.lower_bound == 0: #If lower bound is 0, then this is a fixed bound because it is an exchange reaction
                            Al_df_r.loc[jl,r_i] = 1
                            
                        if islimm:
                            Al_df_r.loc[jl,r_i] = 1
                            Gl_df_r.loc[jl,m.strip(" ")] = 1
                                
                                            
                    
            else: #Only one gene
                
                
                if n == "s0001": #'Gene is not mapped to a genome annotation' (Diffusion transport)
                    n = r_i+"_s"
                    Gu_df_r.loc[:,n] = 0 #Make our own gene
                    Gl_df_r.loc[:,n] = 0
                    #This if statement will not be reached if exch=False
                
                if nummet != 1:
                    Au_df_r.loc[ju,r_i] = 1 #Internal reactions always have some type of constraint on both upper and lower bounds
                    Al_df_r.loc[jl,r_i] = 1
                    if r.upper_bound != 0:
                        Gu_df_r.loc[ju,n] = 1
                    if r.lower_bound != 0:
                        Gl_df_r.loc[jl,n] = 1
                elif nummet == 1:

                    if r.upper_bound == 0:
                        Au_df_r.loc[ju,r_i] = 1
                    if isuimm: #If a fixed bound, need to incorporate gene
                        Au_df_r.loc[ju,r_i] = 1
                        Gu_df_r.loc[ju,n] = 1
                        
                    if r.lower_bound == 0:
                        Al_df_r.loc[jl,r_i] = 1
                    
                    if islimm:
                        Al_df_r.loc[jl,r_i] = 1
                        Gl_df_r.loc[jl,n] = 1
                    
            if isuimm:
                genes_rxn = Gu_df_r.loc[ju,:] == 1
                ngene_rxn = Gu_df_r.loc[ju,:].sum()
                
                if ngene_rxn > 1:
                    warnings.warn("Setting an immutable upper bound for reaction {} that has more than 1 corresponding gene. Double check.".format(r_i))
                
                cols = list(Gu_df_r.columns)
                for g in Gu_df_r.columns[genes_rxn]:
                    uimm_dict[g] = (cols.index(g),r.upper_bound/ngene_rxn)
                    
            if islimm:
                genes_rxn = Gl_df_r.loc[jl,:] == 1
                ngene_rxn = Gl_df_r.loc[jl,:].sum()
                
                if ngene_rxn > 1:
                    warnings.warn("Setting an immutable lower bound for reaction {} that has more than 1 corresponding gene. Double check.".format(r_i))
                
                cols = list(Gl_df_r.columns)
                for g in Gl_df_r.columns[genes_rxn]:
                    limm_dict[g] = (cols.index(g),-r.lower_bound/ngene_rxn)

                    
            
                

    else: #No gene information was included, so need to make our own gene
    
        ju = len(Gu_df_r.index) #Need to append to end of the dataframe
        jl = len(Gl_df_r.index)
        Gu_df_r.loc[:,r_i+"_gene"] = [0] #Make our own gene
        Gu_df_r[Gu_df_r.isnull()] = 0
        Gl_df_r.loc[:,r_i+"_gene"] = [0] #Make our own gene
        Gl_df_r[Gl_df_r.isnull()] = 0
        
        Gu_df_r.loc[ju,:] = 0 #Fill in the new  upper bound row for the constraint with 0s
        Gl_df_r.loc[jl,:] = 0
        Au_df_r.loc[ju,:] = 0
        Al_df_r.loc[jl,:] = 0
        
        
        if nummet != 1:
            Au_df_r.loc[ju,r_i] = 1
            Al_df_r.loc[jl,r_i] = 1
            
            if r.upper_bound != 0:
                Gu_df_r.loc[ju,r_i+"_gene"] = 1
            if r.lower_bound != 0:    
                Gl_df_r.loc[jl,r_i+"_gene"] = 1
        
        elif nummet == 1:

            if r.upper_bound == 0:
                Au_df_r.loc[ju,r_i] = 1
                
            if isuimm: #If a fixed bound, need to incorporate gene
                Au_df_r.loc[ju,r_i] = 1
                Gu_df_r.loc[ju,r_i+"_gene"] = 1
                
            if r.lower_bound == 0:
                Al_df_r.loc[ju,r_i] = 1
                
            if islimm:
                Al_df_r.loc[jl,r_i] = 1
                Gl_df_r.loc[jl,r_i+"_gene"] = 1
        
            
        if isuimm:
            cols = list(Gu_df_r.columns)
            uimm_dict[r_i+"_gene"] = (cols.index(r_i+"_gene"),r.upper_bound)
            
        if islimm:
            cols = list(Gl_df_r.columns)
            limm_dict[r_i+"_gene"] = (cols.index(r_i+"_gene"),-r.lower_bound)


    return Au_df_r,Al_df_r,Gu_df_r,Gl_df_r,uimm_dict,limm_dict



def R_genes(genes,R_bnames,norm=True):
    
    
    genes_not_in_R = list(set(genes) - set(R_bnames.index))
    genes_in_R = list(set(genes) - set(genes_not_in_R))
    
    R_genes_df = R_bnames.loc[genes_in_R,:]
    R_genes_df = R_genes_df.drop(columns=R_genes_df.loc[:,R_genes_df.sum(axis=0) == 0].columns)
    
    
    for g in genes_not_in_R:
        R_genes_df.loc[g,:] = np.zeros(R_genes_df.shape[1])
        
        
    R_genes_df = R_genes_df.loc[genes,:] #Put the df in the expected order
    
    if norm:
        R_genes_df = R_genes_df.apply(lambda x:x/x.sum(),axis=0)
    
    return R_genes_df
        
