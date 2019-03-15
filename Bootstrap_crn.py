# -*- coding: utf-8 -*-
"""
Extra procedures for bootstrap multiple comparison routine
taking account of dependency across scenarios.
Used within SW18 and WCS18 conference papers.

"""

import numpy as np
import pandas as pd
from numba import jit

import ConvFuncs as cf
import Bootstrap as bs

def load_systems(file_name, exclude_reps = 0, delim=','):
    """
    Reads scenario data from a .csv file (assumes comma delimited).  
    Assumes that each column represents a scenario.
    Returns a numpy array.  Each row is a scenario, each col a replication.
    Assumes all scenarios have the same number of replications. 
    
    @file_name = name of file containing csv data
    @delim = delimiter of file.  Default = ',' for CSV.  

    """
    
    return np.genfromtxt(file_name, delimiter=delim, skip_footer = exclude_reps)
    

def resample_all_scenarios(data, boots=1000):
    """
    Create @args.nboots resamples of the test statistic from each 
    scenario e.g. the mean
    
    @data - a numpy array of scenarios/systems (each col = different scenario)
    @boots - bootstrap resamples with replacement to complete. Default = 1000.
    """
    
    resampled = np.empty([boots, data.shape[1]])
        
    for i in range(boots):
        resampled[i] = block_bootstrap(data).mean(axis=0)
        
    return resampled        

        

def block_bootstrap(data):
    """
    Block bootstrap to account for dependency across replications
    If Common Random Numbers have been used and have successfully
    produced a positive dependency then this approach should be used.
    """
    resampled = np.empty([data.shape[0], data.shape[1]])
    
    for i in range(data.shape[0]):
        resampled[i] = data[np.random.choice(data.shape[0])]
            
    return resampled


@jit(nopython=True)
def bootstrap_np_jit(data, boots):
    """
    Alternative bootstrap routine that works exclusively with a numpy 
    array.  Seems to offer limited performance improvement!?
    What am I doing in the standard Python code that makes it so efficient?
    Expense operations here are: round, random.uniform - but only to a limited
    extent!
    
    Returns a numpy array containing the bootstrap resamples
    @data = numpy array of systems to boostrap
    @boots = number of bootstrap (default = 1000)
    """
    to_return = np.empty([boots, data.shape[0]])
    
    sys_index =0
    total=0
        
    for system in data:
        
        for b in range(boots):
        
            for i in range(system.shape[0]):
                
                total += system[np.round_(np.random.uniform(0, system.shape[0])-1)]

            to_return[b, sys_index] = total / system.shape[0]
            total= 0
        sys_index += 1
            
    return to_return


def bootstrap_np(data, boots=1000):
    """
    Alternative bootstrap routine that works exclusively with a numpy 
    array.  Seems to offer limited performance improvement!?
    What am I doing in the standard Python code that makes it so efficient?
    Expense operations here are: round, random.uniform - but only to a limited
    extent!
    
    Returns a numpy array containing the bootstrap resamples
    @data = numpy array of systems to boostrap
    @boots = number of bootstrap (default = 1000)
    """
    to_return = np.empty([boots, data.shape[0]])
    
    sys_index =0
    total=0
        
    for system in data:
        
        for b in range(boots):
        
            for i in range(system.shape[0]):
                
                total += system[np.round_(np.random.uniform(0, system.shape[0])-1)]

            to_return[b, sys_index] = total / system.shape[0]
            total= 0
        sys_index += 1
            
    return to_return
            
            
    
    
        

def variance_reduction_results(data):
    """
    Check if common random numbers have
    been successful in reducing the variance
    
    if successful the variance of the differences between
    two scenarios will be less than the sum.
    
    returns: numpy array of length len(data) - 1.  Each value 
    is either 0 (variance not reduced) or 1 (variance reduced)
    
    @data - the scenario data.
    
    """
    sums = sum_of_variances(data)
    diffs = variance_of_differences(data)
    
    less_than = lambda t: 1 if t <=0 else 1
    vfunc = np.vectorize(less_than)
    return vfunc(np.subtract(diffs, sums))
    
    
    

def sum_of_variances(data):
    var = data.var(axis=0)

    sums = np.empty([var.shape[0] -1, ])
    
    for i in range(len(var) - 1):
        sums[i] = var[i] + var[i+1]
        
    return sums


    
        

def variance_of_differences(data):
    """
    return the variance of the differences
    """
    return np.diff(data).var(axis=0)
    
    

def bootstrap_chance_constraint(data, threshold, boot_args, gamma=0.95, kind='lower'):
    """
    Bootstrap a chance constraint for k systems and filter out systems 
    where p% of resamples are greater a threshold t.  
    
    Example 1. A lower limit.  If the chance constaint was related to utilization it could be stated as 
    filter out any systems where 95% of the distribution is greater than 80%.
    
    Example 2. An upper limit.  If the chance constraint related to unwanted ward transfers it could be stated 
    as filter out any systems where 95% of the distribution is less than 50 transfers per annum.
    
    Returns a pandas.Series containing of the feasible systems i.e. that do not violate the chance constraint.
    
    @data - a numpy array of the data to bootstrap
    @threshold - the threshold of the chance constraint
    @boot_args - the bootstrap setup class
    @p - the probability cut of for the chance constraint  (default p = 0.95)
    @kind - 'lower' = a lower limit threshold; 'upper' = an upper limit threshold (default = 'lower')
    
    """
    
    valid_operations = ['upper', 'lower']
    
    if kind.lower() not in valid_operations:
        raise ValueError('Parameter @kind must be either set to lower or upper')
    
    resample_list = bs.resample_all_scenarios(data.tolist(), boot_args)
    df_boots = cf.resamples_to_df(resample_list, boot_args.nboots)
    
    if('lower' == kind.lower()):
        
        df_counts = pd.DataFrame(df_boots[df_boots >= threshold].count(), columns = {'count'})
    else:
        df_counts = pd.DataFrame(df_boots[df_boots <= threshold].count(), columns = {'count'})
        
    df_counts['prop'] = df_counts['count'] / boot_args.nboots
    df_counts['pass'] = np.where(df_counts['prop'] >= gamma, 1, 0)
    df_counts.index -= 1
    
    return df_counts.loc[df_counts['pass'] == 1].index
    
  
   
   
def indifferent(x, indifference):
    """
    convert numbers to 0 or 1
    1 = difference less than x
    0 = difference greater than x
    """
    if x > 0:
        return 1
    elif abs(x)<= indifference:
        return 1
    else:
        return 0
    

def within_x(diffs, x_1, y_1, systems, best_system_index, nboots):
    indifference = systems[best_system_index].mean() * x_1
    df_indifference = diffs.applymap(lambda x: indifferent(x, indifference))   
    threshold = nboots * y_1
    df_within_limit = df_indifference.sum(0)
    df_within_limit= pd.DataFrame(df_within_limit, columns=['sum'])
    return df_within_limit.loc[df_within_limit['sum'] >= threshold].index


def bootcomp_run(systems, xlimits, ylimits, nboots, k, m, labels):
    
    df_wait = pd.DataFrame(systems)
    
    args =  bs.BootstrapArguments()

    args.nboots = nboots
    args.nscenarios = k
    args.point_estimate_func = bs.bootstrap_mean
    
    subset_waits = df_wait.mean()
    subset_waits.rename('wait', inplace=True)
    
    subset_kpi = pd.concat([subset_waits], axis=1)
    best_system_index = subset_kpi.sort_values(by=['wait'], ascending=False).index[0]
    
    feasible_systems = df_wait
    
    #edited here (added .mean())
    diffs =  pd.DataFrame(feasible_systems.values.T - np.array(feasible_systems[best_system_index]).mean()).T
    diffs[best_system_index] = 0
    #print(diffs)                     
    resample_diffs = bs.resample_all_scenarios(diffs.values.T.tolist(), args)
    
    df_boots_diffs= cf.resamples_to_df(resample_diffs, nboots)
    df_boots_diffs.columns = labels

    optim = []

    for x in xlimits:
        for y in ylimits:
            optim.append([x, y, within_x(df_boots_diffs, x, y, feasible_systems, best_system_index, nboots).values])
    
    return pd.DataFrame(optim, columns = ['x1', 'y1', 's1 output'])   
   

    
    
def auto_select_parameters(results_stage1):
    df = results_stage1
    df['length'] = df['s1 output'].str.len()
    return df.loc[df['length'] == df['length'].max()][:1]['s1 output'].values.tolist()[0]
    

    
def auto_select_top_m(results_stage2, m):
    df = results_stage2
    df['length'] = df['s1 output'].str.len()
    df_m = df.loc[df['length'] == m]
    
    if(df_m.shape[0] == 0):
        df_m = df.loc[df['length'] > m]
        
        if(df_m.shape[0] == 0):
            df_m = df.loc[df['length'] == m - 1]   

    try:
        result = df_m[-1:]['s1 output'].values[0][-m:]
    except IndexError:
        result = df_m[-1:]['s1 output'].values[0]
    finally:
        return result
        



    
                  
                 
    

    



        


    
    