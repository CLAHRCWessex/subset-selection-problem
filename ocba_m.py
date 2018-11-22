# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:31:06 2018

@author: tm3y13
"""

import numpy as np

k = 10
m = 3
T = 1000
n_0 = 5
delta = 50

#specific to this implementation
ifile_name = 'data/EG1.csv'
reps_available = 5000


def load_system(file_name, system, reps, reps_available, delim=','):
    """
    Reads system data from a txt file (assumes comma delimited by default).  
    Assumes that each column represents a system.
    Returns a numpy array.  Each row is a system (single row); each col is a replication
    
    
    @file_name = name of file containing csv data
    @system = index of system in txt file.
    @reps = replications wanted
    @reps_available = total number of replications that has been simulated.
    @delim = delimiter of file.  Default = ',' for CSV.  

    """
    
    return np.genfromtxt(file_name, delimiter=delim, usecols = system, skip_footer = reps_available - reps)

def simulate(allocations):
    """
    Simulates the systems.  
    Each system is allocated a different budget of replications
    
    Returns list of numpy arrays
    
    @allocations = numpy array.  budget of replications for the k systems 
    """
    x = [load_system(file_name = ifile_name, system = i,reps = allocations[i], reps_available=reps_available) 
         for i in range(k)]
    return x

def get_ranks(array):
    """
    Returns a numpy array containing ranks of numbers within a input numpy array
    e.g. [3, 2, 1] returns [2, 1, 0]
    e.g. [3, 2, 1, 4] return [2, 1, 0, 3]
        
    @array - numpy array (only tested with 1d)
        
    """
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def summary_statistics(systems, allocations):
    means = np.array([array.mean() for array in systems])
    stds = np.array([array.std() for array in systems])
    ses = np.divide(stds, np.sqrt(allocations))
    return means, ses

def top_m(means):
    return get_ranks(means)

def ocba_m(systems, allocations):
    while allocations.sum() < T:
       
        #calculate sample means and standard errors
        means, ses = summary_statistics(systems, allocations)

        #calculate paramater c and gammas
        c = ((ses[m+1] * means[m]) + (ses[m] * means[m+1]))/(ses[m]+ses[m+1])
        gammas = means - c

        #new allocation
        values = np.divide(allocations, np.square(np.divide(ses, gammas)))

        for i in range(delta):
            ranks = get_ranks(values)
            values[ranks.argmin()] += 1
            allocations[ranks.argmin()] += 1

        #simulate systems using new allocation of budget
        systems = simulate(allocations)
        
    means, ses = summary_statistics(systems, allocations)
    return means, ses, allocations


allocations = np.full(k, n_0, dtype=int)
systems = simulate(allocations)

means, ses, allocations = ocba_m(systems, allocations)

print(means)
print(allocations)

print(top_m(means))