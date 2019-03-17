# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:55:51 2018

Based on example in Law: Sim Modeling and Analysis, V4, P 53
CRN added from page 592

@author: ccurrie
"""

import numpy as np
#import scipy.stats as stats
#import operator
#from matplotlib import pyplot as plt


a=1

def initialize():
    global initial_seed
    global sim_time
    global inv_time
    global total_ordering_cost
    global time_next_event
    global inv_level
    global time_last_event
    global area_holding
    global area_shortage
    
    sim_time=0.0
    inv_level=initial_inv_level
    time_last_event=0.0
    
    total_ordering_cost=0.0
    area_holding=0.0
    area_shortage=0.0
    
    time_next_event[1]=1000000000000000000000
    time_next_event[2]=sim_time+ np.random.RandomState(initial_seed).exponential(mean_interdemand,1)
    time_next_event[3]=num_months
    time_next_event[4]=0.000

def order_arrival():
    global inv_level
    global time_next_event
    global amount
    global time_next_event
    
    #print(amount)
    inv_level += amount 
    time_next_event[1]=10000000000000000000000

def demand():
    #Hard code in the demand distribution from Law
    global inv_level
    global time_next_event
    global sim_time
    global rng_demand_size
    global rng_demand_interval
    
    #u = np.random.rand(1)[0]
    u = rng_demand_size.random_sample(1)
    #print(u)
    if u <1/6:
        inv_sample=1
    elif u<1/2:
        inv_sample=2
    elif u<5/6:
        inv_sample=3
    else:
        inv_sample=4
    inv_level -= inv_sample
    time_next_event[2]=sim_time + rng_demand_interval.exponential(mean_interdemand,1)#np.random.exponential(mean_interdemand)
    #print(time_next_event[2])

def evaluate():
    global total_ordering_cost
    global time_next_event
    global amount
    global smalls
    if inv_level < smalls:
        amount=bigs - inv_level
        #print(bigs,inv_level,amount,setup_cost,incremental_cost)
        total_ordering_cost += setup_cost+incremental_cost * amount
        time_next_event[1]=sim_time+np.random.uniform(minlag,maxlag)
    
    time_next_event[4]=sim_time+1.0


def report():
    global policy_num
    global avg_ordering_cost
    global avg_holding_cost
    global avg_shortage_cost
    global avg_total_cost
    
    avg_ordering_cost[policy_num] = total_ordering_cost/num_months
    avg_holding_cost[policy_num] = holding_cost*area_holding/num_months 
    avg_shortage_cost[policy_num] = shortage_cost*area_shortage/num_months
    avg_total_cost[policy_num] = avg_ordering_cost[policy_num]+avg_holding_cost[policy_num]+avg_shortage_cost[policy_num]


def update_time_avg_stats():
    global time_last_event
    global inv_level
    global sim_time
    global area_shortage
    global area_holding
    
    time_since_last_event = sim_time - time_last_event
    time_last_event=sim_time
    if inv_level<0:
        area_shortage -= inv_level*time_since_last_event
        #print(area_shortage," shortage ", sim_time)
    elif inv_level > 0:
        area_holding += inv_level* time_since_last_event
        #print(area_holding, " holding ", sim_time)
        
def timing():
    global sim_time
    global next_event_type
    global num_events
    
    min_time_next_event = 100000000000000000
    next_event_type = 0
    #print("I am in timing and num_events is",num_events)
    for i in range(1,num_events+1):
        #print("Inside loop and i is",i, "time_next_event[i] is",time_next_event[i])
        if time_next_event[i] <min_time_next_event:
            min_time_next_event=time_next_event[i]
            next_event_type = i
    #print("Time next event is ", min_time_next_event, " and next event is ", next_event_type)
            
    if next_event_type ==0:
        return #exit the function without updating the simulation clock
    
    sim_time=min_time_next_event

#Main file here ...
#Set up input data
num_reps=100
num_events = 4
initial_inv_level = 60.0
num_months = 120
num_policies = 9
num_values_demand = 4
mean_interdemand = 0.10
setup_cost = 32
incremental_cost = 3
holding_cost = 1
shortage_cost = 5
minlag = 0.5
maxlag = 1.0
time_next_event=np.zeros(5)
next_event_type = 0
sim_time=0
time_last_event=0
inv_level=initial_inv_level
amount = 0
avg_ordering_cost=np.zeros(num_policies+1)
avg_holding_cost=np.zeros(num_policies+1)
avg_shortage_cost=np.zeros(num_policies+1)
avg_total_cost = np.zeros(num_policies+1)
output_data = np.zeros(num_policies+1)

policy = np.array([[0,0],[20,40],[20,60],[20,80],[20,100],[40,60],[40,80],[40,100],[60,80],[60,100]])
for rep in range(num_reps+1):
    #Set seed for initialisation (note the same seed is used for all policies so the same value should be sampled)
    initial_seed,demand_size_seed,demand_interval_seed = np.random.RandomState().random_integers(1,2**31-1,3)
    #print(initial_seed, demand_size_seed,demand_interval_seed)
    rng_demand_size = np.random.RandomState(demand_size_seed)
    rng_demand_interval = np.random.RandomState(demand_interval_seed)
    for policy_num in range (1,num_policies+1):
        smalls = policy[policy_num][0]
    #print(smalls)
        bigs=policy[policy_num][1]
        initialize()
        #print(policy_num)
        while next_event_type != 3:
            timing()
            update_time_avg_stats()
            if next_event_type ==1:
                order_arrival()
            elif next_event_type ==2:
                demand()
            elif next_event_type ==3:
                report()
            elif next_event_type ==4:
                evaluate()
        next_event_type = 0
    #print(avg_total_cost)
    output_data=np.vstack((output_data,avg_total_cost))
    #print(output_data)

np.savetxt('EGLawCRN.csv',output_data,delimiter=',')
#np.savetxt('EGLaw.csv',[avg_ordering_cost,avg_holding_cost,avg_shortage_cost,avg_total_cost] ,delimiter=',')

