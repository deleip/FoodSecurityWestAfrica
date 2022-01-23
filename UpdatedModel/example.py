# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 17:17:02 2022

@author: leip
"""

# import necessary packages
import os
import pickle

# import all project related functions
import FoodSecurityModule as FS  

# set right directory
os.chdir("path/to/main/model/folder")


# %% ######################### 0. GROUPING CLUSTERS ###########################

# combinations of the aim and whether clusters in a group have to be adjacent
aim = "Similar"
adjacent = True
metric = "medoids" 
k = 9
s = 3
adj_text = "Adj"


with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
             "_PearsonDistSPEI.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    
print("Grouping for metric " + metric + ", group size s = " + str(s) + \
      " according to " + aim.lower() + "ity with " + \
      "adjacency " + str(adjacent), flush = True)
    
BestGrouping, BestCosts, valid = \
        FS.GroupingClusters(k = k, size = s, aim = aim, \
            adjacent = adjacent, metric = metric, title = None)
# output in InputData/Clusters/ClusterGroups
            
FS.PlotClusterGroups(grouping = BestGrouping,
                     k = k,
                     title = "Grouping for k = " + str(k) + ", s = " + str(s) +
                     ", aim = " + aim  + ", adjacent =" + str(adjacent) + ", metric = " + metric,
                     file = "Figures/ClusterGroups/k" + str(k) + "s" + str(s) + \
                             "Aim" + aim + adj_text + metric.capitalize())
# figure saved using filename relative to main folder

# %% ########################### 1. BASIC MODEL RUN ###########################

y = "fixed"
p = "fixed"

N = 1000
M = 5000

cl = 3

# should take about 1.5 min
settings, args, yield_information, population_information, \
status, durations, exp_incomes, crop_alloc, meta_sol, \
crop_allocF, meta_solF, crop_allocS, meta_solS, \
crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
   FS.FoodSecurityProblem(validation_size = M,
                      plotTitle = "Yields " + y + ", population trend " + p.lower() + ": cluster " + str(cl),
                      k_using = cl,
                      N = N,
                      yield_projection = y,
                      pop_scenario = p)
# full results saved in ModelOutput/SavedRuns
# specific indicators added to csv in ModelOutput/Pandas

# %% ########################### 2. READING RAW DATA ##########################

# should take less than a minute
import ReadingRawData

# %% ########################## 3. PREPATING INPUT DATA ####################### 

# takes wuite a long time, as clustering algorithm is run for k = 1, ..., 20
import InputDataCalculations
