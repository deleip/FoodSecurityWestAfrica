#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:42:56 2020

@author: Debbora Leip
"""
# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import pickle

        
# %% ######################### 0. GROUPING CLUSTERS ###########################

# combinations of the aim and whether clusters in a group have to be adjacent
comb = [("Similar", True),
        ("Dissimilar", True),
        ("Similar", False), 
        ("Dissimilar", False)]

k = 9
for s in [1, 2, 3, 5]:
    for (aim, adjacent) in comb:
        print("group size s = " + str(s) + " according to " + aim + "ity with " + \
              "adjacency " + str(adjacent), flush = True)
        BestGrouping, BestCosts, valid = \
                FS.GroupingClusters(k = k, size = s, aim = aim, adjacent = adjacent, \
                    title ="Viaualization of " + "the grouping of clusters" + \
                        " for k = " + str(k) + " clusters " + "and group" + \
                        " size s = " + str(s) + " according to " + aim + "ity")
        

# %% ##### 2. DEFAULT RUN FOR ADJACENT CLUSTER GROUPS OF SIZE 1, 2, 3, 5  #####

# group size, sample size N, validation sample size M
comb = [#(1, 100000, 200000),
       # (2, 40000, 150000),
       # (3, 250000, 400000),
        #(5, 400000, 600000),
        ("all", 300000, 500000)
        ]

for size, N, M in comb:
    if size == "all":
        settings, args, AddInfo_CalcParameters, yield_information, \
        population_information, status, durations, crop_alloc, meta_sol, \
        crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
            FS.FoodSecurityProblem(validation_size = M,
                                   plotTitle = "All clusters",
                                   k_using = size,
                                   N = N)
        continue
    
    for aim in ["Dissimilar"]:    
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + aim + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
                
        for cluster_active in BestGrouping:
            print("\u2017"*49)
            print("Aim: " + aim + ", size: " + str(size) + ", clusters: " + str(cluster_active))
            print("\u033F "*49)
            
            settings, args, AddInfo_CalcParameters, yield_information, \
            population_information, status, durations, crop_alloc, meta_sol, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = M,
                                       plotTitle = "Aim: " + aim  + ", Adjacent: False",
                                       k_using = list(cluster_active),
                                      N = N)


# %% ##### 3. PLOTTING RESULTS  #####

# all default settings so no **kwargs
FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Dissimilar",
                                    adjacent = False,
                                    console_output = None)

FS.MainPandaPlotsFixedSettings(panda_file = "current_panda", 
                               grouping_aim = "Dissimilar",
                               adjacent = False)

# %% ##################### 4. RUNS USING ONE CLUSTER ##########################

# We use cluster 3 for some analysis using just a single cluster. 
# TODO: why do we use 3 and not one of the others?

# combinatins of tax, I_gov, and risk to test:
comb = [(0.01, 0.85, 0.05), \
        (0.03, 0.85, 0.05), \
        (0.05, 0.85, 0.05), \
        (0.03, 0.75, 0.05), \
        (0.03, 0.85, 0.05), \
        (0.03, 0.95, 0.05), \
        (0.03, 0.85, 0.03), \
        (0.03, 0.85, 0.05), \
        (0.03, 0.85, 0.10)]
    
for (tax, perc_guaranteed, risk) in comb:
    print("\u2017"*49)
    print("Tax: " + str(tax) + \
          ", perc_guaranteed: " + str(perc_guaranteed) + \
          ", risk: " + str(risk))
    print("\u033F "*49)
    
    crop_alloc, meta_sol, status, durations, settings, args, \
    rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
    validation_values, probSnew, fn = FS.FoodSecurityProblem(PenMet = "prob", 
                                                    probF = 0.95, 
                                                    probS = 0.90, 
                                                    validation = 200000,
                                                    k = 9,
                                                    k_using = [3],
                                                    tax = tax,
                                                    perc_guaranteed = perc_guaranteed,
                                                    risk = risk,
                                                    N = 50000)
    FS.PlotModelOutput(PlotType = "CropAlloc", \
                           title = "Tax: " + str(tax) + \
                     ", I_gov: " + str(perc_guaranteed) + \
                                ", risk: " + str(risk), \
                     file = fn, crop_alloc = crop_alloc, k = 9, \
                     k_using = [3], max_areas = args["max_areas"])
      
tax = 0.03
perc_guaranteed = 0.85
risk = 0.05
for probF in [0.97, 0.99]:
    print("\u2017"*49)
    print("Tax: " + str(tax) + \
          ", perc_guaranteed: " + str(perc_guaranteed) + \
          ", risk: " + str(risk))
    print("\u033F "*49)
    
    crop_alloc, meta_sol, status, durations, settings, args, \
    rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
    validation_values, probSnew, fn = FS.FoodSecurityProblem(PenMet = "prob", 
                                                    probF = probF, 
                                                    probS = 0.90, 
                                                    validation = 200000,
                           plotTitle = "Tax: " + str(tax) + \
                                       ", I_gov: " + str(perc_guaranteed) + \
                                       ", risk: " + str(risk),
                                                    k = 9,
                                                    k_using = [3],
                                                    tax = tax,
                                                    perc_guaranteed = perc_guaranteed,
                                                    risk = risk,
                                                    N = 50000)
# %%

settings, args, AddInfo_CalcParameters, yield_information, \
population_information, status, durations, crop_alloc, meta_sol, \
crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
    FS.FoodSecurityProblem(validation_size = 100000,
                           k_using = [2],
                           #plotTitle = "(Aim: Dissimilar, Adjacent: Flase)",
                           N = 10000)