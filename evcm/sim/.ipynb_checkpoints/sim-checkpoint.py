import numpy as np
import pandas as pd
from evcm.utils import fixed_start, fixednoisy_start, random_start, sample_flux_p, random_fixation,\
 nearest_feasible_gene, mutate_bounds, selective_pressure_flux, selective_pressure_gene,\
 FBA_gene, selective_pressure_FBA_flux, selective_pressure_FBA_gene, biological_start
import warnings

__G__ = 'g'
__V__ = 'v'
__N__ = 'n'

__fix__ = 'f'
__random__ ='r'
__noisy__ = 'n'
__input__ = 'i'
__biological__ = 'b'

__full__ = 'f'
__neutral__ = 'n'
__count__ = 'c'

def run_sim(T,Au,Al,S,Gu,Gl,beta,
            pop_size=1000,
            fix_start = 'n',
            heritability_std=0.1,mutate_scale=0.1,simulation_scale=1,
            expected_mutations=1,expected_wiggles=1,
            uimmutable=[],limmutable=[],irreversible=False,
            do_print=True,
            do_SP_flux=False,do_SP_gene=False,
            converged_break=False,
            penalty="n",pen_power=1.05,alpha=0.1,
            fba_move=True,
            sample_rate=10,SP_rate=100,print_rate=10000,
            u_ccomp=None,l_ccomp=None,u_gcomp=None,l_gcomp=None,
            Sigmau=None,Sigmal=None,
            Reaction_labels=None,uGene_labels=None,lGene_labels=None,
            ucComparison_labels=None,lcComparison_labels=None,
            ugComparison_labels=None,lgComparison_labels=None,
            u_g0=None,l_g0=None,v0=None,
            mutation_data='c'):


    # T: int of number of epochs to simulate
    # Au, Al, S, Gu, Gl, and beta: corresponding matrices/vectors in the linear program (numpy arrays)
    
    # pop_size: the population size (N)
    
    # fix_start: boolean that sets if a fixed, fixed noisy, or random start should be used
    
    # heritability_std: spread of heritability "wiggles" (int)
    # mutate_scale: spread of mutation fitness effects (int)
    # simulation_scale: sets the overall scale of the simulation (how large are bounds) (int)
    
    # expected_mutations: int of number of mutations expected in a generation
    # expected_wiggles: int of number of constraints that will be changes between generations
    
    # uimmutable: list of (index, upper constraint) tuples
    # limmutable: list of (index, -lower constraint) tuples (i.e., positive numbers for constraints)
    # irreversible: boolean that indicates if reactions are irreversible
    
    # do_print: boolean that sets if intermediate results should be printed
    
    # do_SP_flux: boolean that indicates if selective pressure on flux constraints should be calculated
    # do_SP_gene: boolean that indicates if selective pressure on genes should be calculated
    
    # converged_break: boolean that sets if the simulation should stop when the relative fluxes have converged
    
    # penalty: "n", "v", or "g" - indicates if a penalty should be included and if it should be on fluxes or genes
    # pen_power: penalty is sum of entrywise power of the vector. This sets the power
    # alpha: float that scales the penalty
        # Ex: penalty = alpha*(v**pen_power).sum()
        
    # fba_move: True: use FBA to calculate new flux after a mutation. False: use nearest_feasible to calculate new flux
    
    # sample_rate: How often core data is recorded
    # SP_rate: How often selective pressure is calculated and recorded
    # print_rate: How often intermediate results are printed
    
    # u_ccomp: matrix of upper flux constraints to calculate selective pressure on
    # l_ccomp: matrix of lower flux constraints to calculate selective pressure on
    # u_gcomp: matrix of upper genes to calculate selective pressure on
    # l_gcomp: matrix of lower genes to calculate selective pressure on
    
    # Sigmau: variance of mutations in individual upper genes
    # Sigmal: variance of mutations in individual lower genes
    
    # Reaction_labels: list of strings that name reactions
    # uGene_labels: list of strings that name upper genes
    # lGene_labels: list of strings that name lower genes
    # ucComparison_label: list of strings that name vectors in u_ccomp
    # lcComparison_label: list of strings that name vectors in l_ccomp
    # ugComparison_label: list of strings that name vectors in u_gcomp
    # lgComparison_label: list of strings that name vectors in l_gcomp
    
    #u_g0: initial upper genes
    #l_g0: initial lower genes
    #v0: initial fluxes
    
    #mutation_data: how to track mutations
    #   'f': Full data, every mutation recorded with fitness difference
    #   'n': only Neutral mutations recorded with fitness difference
    #   'c': only final mutation Counts on each gene are recorded
    
    
    #%% Initialize different values if nothing was provided
    
    if do_SP_gene and (u_gcomp is None):
        u_gcomp = np.eye(Gu.shape[1])
    if do_SP_gene and (l_gcomp is None):
        l_gcomp = np.eye(Gl.shape[1])
        
    if do_SP_flux and (u_ccomp is None):
        u_ccomp = np.eye(Gu.shape[0])
    if do_SP_flux and (l_ccomp is None):
        l_ccomp = np.eye(Gl.shape[0])
    
    if Reaction_labels is None:
        Reaction_labels = ['%d' % i for i in range(S.shape[1])]
        
    if uGene_labels is None:
        uGene_labels = ['%d' % i for i in range(Gu.shape[1])]
    if lGene_labels is None:
        lGene_labels = ['%d' % i for i in range(Gl.shape[1])]
        
    if ucComparison_labels is None and u_ccomp is not None:
        ucComparison_labels = ['%d' % i for i in range(u_ccomp.shape[1])]
    if lcComparison_labels is None and l_ccomp is not None:
        lcComparison_labels = ['%d' % i for i in range(l_ccomp.shape[1])]
        
    if ugComparison_labels is None and u_gcomp is not None:
        ugComparison_labels = ['%d' % i for i in range(u_gcomp.shape[1])]
    if lgComparison_labels is None and l_gcomp is not None:
        lgComparison_labels = ['%d' % i for i in range(l_gcomp.shape[1])]
        
    if Sigmau is None:
        Sigmau = np.eye(Gu.shape[1])
    if Sigmal is None:
        Sigmal = np.eye(Gl.shape[1])
        
    if penalty == __G__:
        fitness = lambda v, gu, gl: beta.dot(v) - alpha*(gu**pen_power).sum() - alpha*((-gl)**pen_power).sum()
    elif penalty == __V__:
        fitness = lambda v, gu, gl: beta.dot(v) - alpha*(v**pen_power).sum()
        #set up a different lambda function
    elif penalty == __N__:
        fitness = lambda v, gu, gl: beta.dot(v)

    #%% Initialize flux vectors
    
    
    if fix_start==__fix__:
        u_g,l_g,flux,biomass = fixed_start(Au,Al,S,Gu,Gl,beta,uimmutable=uimmutable,limmutable=limmutable,scale=simulation_scale,irreversible=irreversible)
    elif fix_start==__random__:
        u_g,l_g,flux,biomass = random_start(Au,Al,S,Gu,Gl,beta,uimmutable=uimmutable,limmutable=limmutable,scale=simulation_scale,irreversible=irreversible)
    elif fix_start==__noisy__:
        u_g,l_g,flux,biomass = fixednoisy_start(Au,Al,S,Gu,Gl,beta,uimmutable=uimmutable,limmutable=limmutable,scale=simulation_scale,irreversible=irreversible)
    elif fix_start==__input__:
        if u_g0 is not None and l_g0 is not None and v0 is not None:
            u_g = u_g0
            l_g = l_g0
            flux = v0
        else: #Just default back to fixednoisy
            u_g,l_g,flux,biomass = fixednoisy_start(Au,Al,S,Gu,Gl,beta,uimmutable=uimmutable,limmutable=limmutable,scale=simulation_scale,irreversible=irreversible)
    elif fix_start==__biological__:
        u_g,l_g,flux,biomass = biological_start(Au,Al,S,Gu,Gl,beta,uimmutable=uimmutable,limmutable=limmutable,scale=simulation_scale,irreversible=irreversible)
        
    biomass = fitness(flux,u_g,l_g)
    if biomass < 0:
        warnings.warn("Fitness started at less than 0")
    
    #%% Initialize various matrices
    
    #"Classic" variables
    df_flux = {'Time':[], 'Reaction': [], 'Flux': []}
    df_biomass = {'Time': [], 'Biomass': []}
    df_optimal_biomass = {'Time':[], 'Biomass':[]}
    df_ubounds = {'Time': [], 'Bound': [], 'Gene': []}
    df_lbounds = {'Time': [], 'Bound': [], 'Gene': []}
    
    #Selective pressures
    df_ucSP_fba = {'Time':[],'Vector':[],'Selective Pressure':[]}
    df_lcSP_fba = {'Time':[],'Vector':[],'Selective Pressure':[]}
    df_ugSP_fba = {'Time':[],'Vector':[],'Selective Pressure':[]}
    df_lgSP_fba = {'Time':[],'Vector':[],'Selective Pressure':[]}

    df_ucSP_emp = {'Time':[],'Vector':[],'Selective Pressure':[]}
    df_lcSP_emp = {'Time':[],'Vector':[],'Selective Pressure':[]}
    df_ugSP_emp = {'Time':[],'Vector':[],'Selective Pressure':[]}
    df_lgSP_emp = {'Time':[],'Vector':[],'Selective Pressure':[]}

    #Mutations
    if mutation_data == __full__ or mutation_data == __neutral__:
        df_umutation = {'Time':[],'Gene':[],'Mutation':[]}
        df_lmutation = {'Time':[],'Gene':[],'Mutation':[]}
        df_neutral = {"Time":[],'Fitness Difference':[]}
    
    umutation_count = np.zeros(Gu.shape[1])
    neutral_umutation_count = np.zeros(Gu.shape[1])
    lmutation_count = np.zeros(Gl.shape[1])
    neutral_lmutation_count = np.zeros(Gl.shape[1])
    changing_bounds = []
    mutate_prob = expected_mutations/(Gu.shape[1] + Gl.shape[1])
    wiggle_prob = expected_wiggles/S.shape[1]
    
    
    #%% Run simlation
    for t in range(T):
        
        #%%% Mutation and flux wiggle
        new_u_g = mutate_bounds(u_g,mutate_prob,scale=mutate_scale*simulation_scale,immutable=uimmutable,Sigma=Sigmau)
        new_l_g = -mutate_bounds(-l_g,mutate_prob,scale=mutate_scale*simulation_scale,immutable=limmutable,Sigma=Sigmal)
            
        
        if fba_move:
            _,new_flux = FBA_gene(new_u_g,new_l_g,Au,Al,S,Gu,Gl,beta)
        else:
            new_flux = sample_flux_p(flux,heritability_std*simulation_scale,wiggle_prob)
            new_flux = nearest_feasible_gene(new_u_g,new_l_g,new_flux,Au,Al,S,Gu,Gl,irreversible=irreversible) 
            
        new_biomass = fitness(new_flux,new_u_g,new_l_g)
        selective_coeff = new_biomass - biomass
        prob_fix = random_fixation(selective_coeff, pop_size)
        
        #%%% Fixation
        if (new_biomass >= 0) and (np.random.random() < prob_fix):
            
            if mutation_data == __full__ or (mutation_data == __neutral__ and selective_coeff.round(8) == 0):
                #If we are recording every mutation or if it is neutral and 
                #we are recording neutral
                
                df_umutation['Time'] += [t]*Gu.shape[1]
                df_umutation['Gene'] += uGene_labels
                df_umutation['Mutation'] += (new_u_g - u_g).tolist()
                
                df_lmutation['Time'] += [t]*Gl.shape[1]
                df_lmutation['Gene'] += lGene_labels
                df_lmutation['Mutation'] += (new_l_g - l_g).tolist()
                
                df_neutral['Time'] += [t]
                df_neutral['Fitness Difference'] += [new_biomass - biomass] 
            
            
            flux_change = new_flux - flux
            umutation_count += new_u_g != u_g
            lmutation_count += new_l_g != l_g
            if selective_coeff.round(8) == 0:
                neutral_umutation_count += new_u_g != u_g
                neutral_lmutation_count += new_l_g != l_g
                
            flux = new_flux
            u_g = new_u_g
            l_g = new_l_g
            biomass = new_biomass
            changing_bounds.append(np.hstack([new_u_g,new_l_g])/new_u_g.max())
            
            

            
            
        else:
            flux_change = 0
        
        #%%% Record data
        if t%sample_rate == 0:

            # keep track of everything
            df_flux['Time'] += [t]*S.shape[1]
            df_flux['Reaction'] += Reaction_labels
            # df_flux['Relative Flux'] += (flux/flux.max()).tolist()
            df_flux['Flux'] += flux.tolist()

            df_ubounds['Time'] += [t]*Gu.shape[1]
            df_lbounds['Time'] += [t]*Gl.shape[1]
            df_ubounds['Gene'] += uGene_labels
            df_lbounds['Gene'] += lGene_labels
            df_ubounds['Bound'] += (u_g).tolist()
            df_lbounds['Bound'] += (l_g).tolist()
            df_biomass['Time'].append(t)
            df_biomass['Biomass'].append(biomass)
            
            
            
            cb = []
            for cbi in range(10,101,10):
                if len(changing_bounds) >= cbi:
                    cb.append(abs(changing_bounds[-1] - changing_bounds[-cbi]).max())
                else:
                    cb.append(1)
            if converged_break and max(cb) < 0.01:
                break
        
        #%%% Print results
        if (t%print_rate == 0) and do_print:
            print(t, np.round(biomass,3), np.round(flux,2), np.round(u_g,2), np.round(l_g,2), umutation_count, lmutation_count)
          
            
        #%%% Calculate selective pressure
        
        #%%%% Flux
        if t%SP_rate == 0 and do_SP_flux:

            #Use FBA
            SP_uc_fba,SP_lc_fba = selective_pressure_FBA_flux(Au,Al,S,Gu,Gl,beta,flux,u_g,l_g,u_ccomp,l_ccomp)

            df_ucSP_fba["Time"] += [t]*u_ccomp.shape[1]
            df_ucSP_fba["Vector"] += ucComparison_labels
            df_ucSP_fba["Selective Pressure"] += list(SP_uc_fba)
            df_lcSP_fba["Time"] += [t]*l_ccomp.shape[1]
            df_lcSP_fba["Vector"] += lcComparison_labels
            df_lcSP_fba["Selective Pressure"] += list(SP_lc_fba)
            
            #Do it "empirically"
            SP_uc_emp,SP_lc_emp = selective_pressure_flux(Au,Al,S,Gu,Gl,beta,flux,u_g,l_g,u_ccomp,l_ccomp)

            df_ucSP_emp["Time"] += [t]*u_ccomp.shape[1]
            df_ucSP_emp["Vector"] += ucComparison_labels
            df_ucSP_emp["Selective Pressure"] += list(SP_uc_emp)
            df_lcSP_emp["Time"] += [t]*l_ccomp.shape[1]
            df_lcSP_emp["Vector"] += lcComparison_labels
            df_lcSP_emp["Selective Pressure"] += list(SP_lc_emp)
      

        #%%%% Genes
        if t%SP_rate == 0 and do_SP_gene:
            
            #Use FBA
            SP_ug_fba,SP_lg_fba = selective_pressure_FBA_gene(Au,Al,S,Gu,Gl,beta,flux,u_g,l_g,u_gcomp,l_gcomp)
                        
            df_ugSP_fba["Time"] += [t]*u_gcomp.shape[1]
            df_ugSP_fba["Vector"] += ugComparison_labels
            df_ugSP_fba["Selective Pressure"] += list(SP_ug_fba)
            df_lgSP_fba["Time"] += [t]*l_gcomp.shape[1]
            df_lgSP_fba["Vector"] += lgComparison_labels
            df_lgSP_fba["Selective Pressure"] += list(SP_lg_fba)
            
            #Do it "empirically"
            SP_ug_emp,SP_lg_emp = selective_pressure_gene(Au,Al,S,Gu,Gl,beta,flux,u_g,l_g,u_gcomp,l_gcomp)

            df_ugSP_emp["Time"] += [t]*u_gcomp.shape[1]
            df_ugSP_emp["Vector"] += ugComparison_labels
            df_ugSP_emp["Selective Pressure"] += list(SP_ug_emp)
            df_lgSP_emp["Time"] += [t]*l_gcomp.shape[1]
            df_lgSP_emp["Vector"] += lgComparison_labels
            df_lgSP_emp["Selective Pressure"] += list(SP_lg_emp)
            
        #%%%% Optimal biomass
        if t%SP_rate == 0 and (do_SP_flux or do_SP_gene):
            
            optimal_biomass,_ = FBA_gene(u_g,l_g,Au,Al,S,Gu,Gl,beta,return_lagrange=False,irreversible=False)
            df_optimal_biomass["Time"] += [t]
            df_optimal_biomass["Biomass"] += [optimal_biomass]
            
    
        if mutation_data == __count__:
            df_umutation = {'Gene':uGene_labels,'Mutation Count':umutation_count}
            df_lmutation = {'Gene':lGene_labels,'Mutation Count':lmutation_count}
            df_neutral = {"Gene":uGene_labels+lGene_labels,'UpperLower':['Upper']*len(uGene_labels)+['Lower']*len(lGene_labels),'Neutral Count':list(neutral_umutation_count)+list(neutral_lmutation_count)}
            
            
            
    return pd.DataFrame(df_flux), pd.DataFrame(df_ubounds), pd.DataFrame(df_lbounds), pd.DataFrame(df_biomass), \
            pd.DataFrame(df_umutation), pd.DataFrame(df_lmutation), pd.DataFrame(df_neutral), \
            pd.DataFrame(df_ucSP_fba), pd.DataFrame(df_lcSP_fba), pd.DataFrame(df_ugSP_fba), pd.DataFrame(df_lgSP_fba), \
            pd.DataFrame(df_ucSP_emp), pd.DataFrame(df_lcSP_emp), pd.DataFrame(df_ugSP_emp), pd.DataFrame(df_lgSP_emp), \
            pd.DataFrame(df_optimal_biomass)

    
    