# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:51:17 2021

@author: leip
"""
# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import pickle

# import all project related functions
import FoodSecurityModule as FS  

# %% ######################### 0. GROUPING CLUSTERS ###########################

# combinations of the aim and whether clusters in a group have to be adjacent
comb = [("Similar", True),
        ("Dissimilar", True),
        ("Similar", False),
        ("Dissimilar", False)]

k = 9
for s in [1, 2, 3, 5]:
    for (aim, adjacent) in comb:
        for metric in ["medoids", "equality"]:
            print("metric " + metric + ", group size s = " + str(s) + \
                  " according to " + aim + "ity with " + \
                  "adjacency " + str(adjacent), flush = True)
            BestGrouping, BestCosts, valid = \
                    FS.GroupingClusters(k = k, size = s, aim = aim, \
                        adjacent = adjacent, metric = metric, title = None)


        
# %% ############## W/O COOPERATION, ALL YIELD AND POP SCENARIOS ############## 

print("\n SECTION 1 \n\n")

for y in ["fixed", "trend"]:
    for p in ["fixed", "Low", "Medium", "High"]:
        for cl in range(1, 10):
            print("\u2017"*65, flush = True)
            print("Yields " + y + ", population trend " + p.lower() + ": cluster " + str(cl), flush = True)
            print("\u033F "*65, flush = True)
            
            settings, args, yield_information, population_information, \
            status, durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = 10000,
                                       # plotTitle = "Yields " + y + ", population trend " + p.lower() + ": cluster " + str(cl),
                                       k_using = cl,
                                       N = 50000,
                                       yield_projection = y,
                                       pop_scenario = p)
                
                
                
##############################################################################
############################# WITHOUT COOPERATION ############################
##############################################################################
                
# %% ############### W/O COOPERATION, DIFF. INPUT PROBABILITIES ############### 

print("\n SECTION 2 \n\n")

for (y, p) in [("fixed", "fixed"), 
                ("trend", "Medium"),
                ("fixed", "High"),
                ("trend", "fixed")]:
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        for cl in range(1, 10):
            print("\u2017"*65, flush = True)
            print("Food security probability " + str(alpha * 100) + "%: cluster " + str(cl), flush = True)
            print("\u033F "*65, flush = True)
            
            settings, args, yield_information, population_information, \
            status, durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = 5000,
                                       # plotTitle = "Food security probability " + str(alpha * 100) + "%: cluster " + str(cl),
                                       k_using = cl,
                                       N = 10000,
                                       probF = alpha,
                                       yield_projection = y,
                                       pop_scenario = p)
            
            
# %% ################## W/O COOPERATION, DIFF. POLICY LEVERS ################## 

print("\n SECTION 3 \n\n")

for tax in [0.01, 0.05, 0.1]:
    for risk in [0.01, 0.05]:
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            for cl in range(3,4):
                print("\u2017"*65, flush = True)
                print(str(tax*100) + "% tax, " + str(risk*100) + "% risk, " + str(alpha*100) + ": cluster " + str(cl), flush = True)
                print("\u033F "*65, flush = True)
                
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = 1000,
                                           # plotTitle = str(tax*100) + "% tax, " + str(risk*100) + "% risk: cluster " + str(cl),
                                           k_using = cl,
                                           N = 10000,
                                           tax = tax,
                                           risk = risk,
                                           probF = alpha,
                                           yield_projection = "trend",
                                           pop_scenario = "Medium")
                    
##############################################################################
############################## WITH COOPERATION ##############################
##############################################################################

# %% #################### MEDOIDS, SIM, ADJ; DEFAULT RUNS #################### 

print("\n SECTION 4 \n\n")

metric = "medoids"
aim = "Similar"
adj = "Adj"
adj_text = "True"

sample_sizes = [(1, 10000, 50000),
                (2, 20000, 50000),
                (3, 50000, 100000),
                (5, 100000, 200000),
                ("all", 100000, 200000)]


for size, N, M in sample_sizes:
    if size == "all":
        print("\u2017"*65)
        print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all")
        print("\u033F "*65)
        settings, args, yield_information, population_information, \
        status, durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
            FS.FoodSecurityProblem(validation_size = M,
                                   # plotTitle = "All clusters",
                                   k_using = size,
                                   N = N)
    else:
        with open("InputData/Clusters/ClusterGroups/Grouping" + metric.capitalize() + \
                  "Size" + str(size) + aim + adj + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
                
        for cluster_active in BestGrouping:
            print("\u2017"*65)
            print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active))
            print("\u033F "*65)
            
            settings, args, yield_information, population_information, \
            status, durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = M,
                                       # plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                       k_using = list(cluster_active),
                                       N = N)
                

# %% ################## EQUALITY, SIM, NON-ADJ; DEFAULT RUNS ################## 

print("\n SECTION 5 \n\n")

metric = "equality"
aim = "Similar"
adj = ""
adj_text = "False"

sample_sizes = [(1, 10000, 50000),
                (2, 20000, 50000),
                (3, 50000, 100000),
                (5, 100000, 200000),
                ("all", 100000, 200000)]


for size, N, M in sample_sizes:
    if size == "all":
        print("\u2017"*65)
        print("Metric " + metric + ", aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all")
        print("\u033F "*65)
        settings, args, yield_information, population_information, \
        status, durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
            FS.FoodSecurityProblem(validation_size = M,
                                   # plotTitle = "All clusters",
                                   k_using = size,
                                   N = N)
    else:
        with open("InputData/Clusters/ClusterGroups/Grouping" + metric.capitalize() + \
                  "Size" + str(size) + aim + adj + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
                
        for cluster_active in BestGrouping:
            print("\u2017"*65)
            print("Metric " + metric + ", aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active))
            print("\u033F "*65)
            
            settings, args, yield_information, population_information, \
            status, durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = M,
                                       # plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                       k_using = list(cluster_active),
                                       N = N)