# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:51:52 2022

@author: leip
"""

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import pickle

# import all project related functions
import FoodSecurityModule as FS  


# %% SAMPLE SIZES TO USE

# first runs
N = 25000

sample_sizes_cooperation = [(1, 25000),
                            (2, 50000),
                            (3, 100000),
                            (5, 100000),
                            ("all", 150000)]


# second runs
N = 50000

sample_sizes_cooperation = [(1, 50000),
                            (2, 100000),
                            (3, 150000),
                            (5, 150000),
                            ("all", 200000)]

# %% RUNS FOR FIGURE 3

# food production and total cultivation costs:
# for default government settings
# different food security target probabilities
# all clusters (no cooperation)
# for worst, best and stationary case

for (y, p) in [("trend", "fixed"),
               ("fixed", "fixed"),
               ("fixed", "High")
               ]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995]: 
        for cl in range(1, 10):
           print("Food security probability " + str(alpha * 100) + "%: cluster " + str(cl), flush = True)
           
           settings, args, yield_information, population_information, penalty_methods, \
           status, durations, exp_incomes, crop_alloc, meta_sol, \
           crop_allocF, meta_solF, crop_allocS, meta_solS, \
           crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(k_using = cl,
                                       plotTitle = "Food security probability " + str(alpha * 100) + "%: cluster " + str(cl),
                                       N = N,
                                       probF = alpha,
                                       yield_projection = y,
                                       pop_scenario = p,
                                       accuracyF_maxProb = 0.0005)
                
           print("\u033F "*65, flush = True)

# %% RUNS FOR FIGURE 4

# crop areas:
# for default government settings
# two reliability levels (90% and 99%)
# all clusters (no cooperation)
# for worst, best and stationary case

for (y, p) in [#("trend", "fixed"),
               #("fixed", "fixed"),
               ("fixed", "High")
               ]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]: 
       for cl in range(1, 10):
           print("Food security probability " + str(alpha * 100) + "%: cluster " + str(cl), flush = True)
            
           settings, args, yield_information, population_information, penalty_methods, \
           status, durations, exp_incomes, crop_alloc, meta_sol, \
           crop_allocF, meta_solF, crop_allocS, meta_solS, \
           crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(k_using = cl,
                                       # plotTitle = "Food security probability " + str(alpha * 100) + "%: cluster " + str(cl),
                                       N = N,
                                       probF = alpha,
                                       yield_projection = y,
                                       pop_scenario = p)
                
           print("\u033F "*65, flush = True)
                
# %% RUNS FOR FIGURE 5

# effect of government levers
# tax = 1, 5, 10%; covered risk = 1, 5% (covered risk default)
# different food security probabilties
# all clusters (no cooperation)
# stationary scenario


for tax in [0.01,
            #0.05,
            #0.1
            ]:
    for risk in [0.01, 0.05]:
        print("\u2017"*65, flush = True)
        print(str(tax*100) + "% tax, " + str(risk*100) + "% risk", flush = True)
        print("\u033F "*65, flush = True)
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            for cl in range(1, 10):
                print("probability " + str(alpha*100) + ", cluster " + str(cl), flush = True)
                
                settings, args, yield_information, population_information, penalty_methods, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(k_using = cl,
                                           # plotTitle = str(tax*100) + "% tax, " + str(risk*100) + "% risk: cluster " + str(cl),
                                           N = N,
                                           tax = tax,
                                           risk = risk,
                                           probF = alpha,
                                           yield_projection = "fixed",
                                           pop_scenario = "fixed")
                
                print("\u033F "*65, flush = True)
                
                    
# %% RUNS FOR FIGURE 6

# cooperation effects:
# for default government settings
# default food security probability
# all clusters and all cooperation levels
# equity and proximity grouping
# for worst, best and stationary case

for (metric, aim, adj, adj_text) in [#("medoids", "Similar", "Adj", "True"),
                                     ("equality", "Similar", "", "False")
                                     ]:
    print("\u2017"*65, flush = True)
    print("COOPERATION PLOTS FOR " + metric + ", " + aim  + ", " + adj + " GROUPING" , flush = True)
    print("\u033F "*65, flush = True)
    for (y, p) in [("trend", "fixed"),
                   ("fixed", "fixed"),
                   ("fixed", "High")
                   ]:
        for size, N_size in [sample_sizes_cooperation[1]]:
            if size == "all":
                print("-"*65, flush = True)
                print("-"*65, flush = True)
                print("\nMetric " + metric + ", aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all")
                
                settings, args, yield_information, population_information, penalty_methods, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(k_using = size,
                                           # plotTitle = "All clusters",
                                           N = N_size,
                                           yield_projection = y,
                                           pop_scenario = p)
                
            else:
                with open("InputData/Clusters/ClusterGroups/Grouping" + metric.capitalize() + \
                          "Size" + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)
                        
                for cluster_active in BestGrouping:
                    print("-"*65, flush = True)
                    print("-"*65, flush = True)
                    print("Metric " + metric + ", aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active))
                  
                    settings, args, yield_information, population_information, penalty_methods, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(k_using = list(cluster_active),
                                               # plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               N = N_size,
                                               yield_projection = y,
                                               pop_scenario = p)

