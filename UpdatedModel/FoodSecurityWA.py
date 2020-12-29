#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:42:56 2020

@author: debbora
"""
from os import chdir 
import pickle
from termcolor import colored

chdir('/home/debbora/git_environment/FoodSecurityWestAfrica/UpdatedModel')
# chdir("H:\ForPublication/NewModel")

import FunctionsStoOpt as StoOpt
StoOpt.CheckFolderStructure()    
        
# %% ######################### 1. GROUPING CLUSTERS ###########################

# combinations of the aim and whether clusters in a group have to be adjacent
comb = [("Similar", "True"),
        ("Dissimilar", "True"),
        ("Similar", "False")]

k = 9
for s in [1, 2, 3, 5]:
    for (aim, adjacent) in comb:
        BestGrouping, BestCosts, valid = \
                StoOpt.GroupingClusters(k = k, size = s, aim = aim, adjacent = adjacent, \
                    title ="Viaualization of " + "the grouping of clusters" + \
                        " for k = " + str(k) + " clusters " + "and group" + \
                        " size s = " + str(s) + " according to " + aim + "ity")
        

# %% ##################### 2. RUNS USING ONE CLUSTER ##########################

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
    
errors = {}
for (tax, perc_guaranteed, risk) in comb:
    print("\u2017"*49)
    print("Tax: " + str(tax) + \
          ", perc_guaranteed: " + str(perc_guaranteed) + \
          ", risk: " + str(risk))
    print("\u033F "*49)
    try:
        crop_alloc, meta_sol, status, durations, settings, args, \
        rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
        validation_values, probSnew, fn = StoOpt.FoodSecurityProblem(PenMet = "prob", 
                                                        probF = 0.95, 
                                                        probS = 0.90, 
                                                        validation = 200000,
                                                        k = 9,
                                                        k_using = [3],
                                                        tax = tax,
                                                        perc_guaranteed = perc_guaranteed,
                                                        risk = risk,
                                                        N = 50000)
        StoOpt.PlotModelOutput(PlotType = "CropAlloc", \
                               title = "Tax: " + str(tax) + \
                         ", I_gov: " + str(perc_guaranteed) + \
                                    ", risk: " + str(risk), \
                         file = fn, crop_alloc = crop_alloc, k = 9, \
                         k_using = [3], max_areas = args["max_areas"])
    except StoOpt.PenaltyException as e:
        print(colored("Tax: " + str(tax) + \
          ", perc_guaranteed: " + str(perc_guaranteed) + \
          ", risk: " + str(risk) + " --- " + str(e), 'red'))
        errors["t" + str(tax) + \
               "p" + str(perc_guaranteed) + \
               "r" + str(risk)] = e
      
errors = {} 
tax = 0.03
perc_guaranteed = 0.85
risk = 0.05
for probF in [0.97, 0.99]:
    print("\u2017"*49)
    print("Tax: " + str(tax) + \
          ", perc_guaranteed: " + str(perc_guaranteed) + \
          ", risk: " + str(risk))
    print("\u033F "*49)
    try:
        crop_alloc, meta_sol, status, durations, settings, args, \
        rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
        validation_values, probSnew, fn = StoOpt.FoodSecurityProblem(PenMet = "prob", 
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
    except StoOpt.PenaltyException as e:
        print(colored("ProbF: " + str(probF) + " --- " + str(e), 'red'))
        errors["pF" + str(probF)] = e    
  
# %% ##################### 2. RUNS USING ONE CLUSTER ##########################

comb = [(1, 50000, 200000),
        (2, 75000, 200000),
        (3, 75000, 200000),
        (5, 100000, 200000)]

errors = {}
for size, M, N in comb:
    for aim in ["Similar", "Dissimilar"]:
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + aim + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
                
        for cluster_active in BestGrouping:
            print("\u2017"*49)
            print("Aim: " + aim + ", size: " + str(size) + ", clusters: " + str(cluster_active))
            print("\u033F "*49)
            try:
                crop_alloc, meta_sol, status, durations, settings, args, \
                rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
                validation_values, probSnew, fn = StoOpt.FoodSecurityProblem(PenMet = "prob", 
                                            probF = 0.99, 
                                            probS = 0.95, 
                                            validation = M,
                                            plotTitle = "Aim: " + aim + ", clusters: " + str(cluster_active),
                                            k = 9,
                                            k_using = list(cluster_active),
                                            tax = 0.03,
                                            perc_guaranteed = 0.85,
                                            risk = 0.05,
                                            N = N)
            except StoOpt.PenaltyException as e:
                print(colored("Aim: " + aim + ", size: " + str(size) + ", clusters: " + str(cluster_active) + " --- " + str(e), 'red'))
                errors["aim" + aim + "cl" + str(cluster_active)] = e
                
                
# %%


crop_alloc, meta_sol, status, durations, settings, args, \
                rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
                validation_values, probSnew, fn = StoOpt.FoodSecurityProblem(
                                            validation = 50000,
                                            k_using = [5],
                                            N = 5000)