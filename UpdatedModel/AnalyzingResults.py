# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:45:49 2020

@author: leip
"""

from os import chdir 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from termcolor import colored
import pandas as pd

chdir('/home/debbora/IIASA/ForPublication/NewModel')
# chdir("H:\ForPublication/NewModel")

import FunctionsStoOpt as StoOpt
StoOpt.CheckFolderStructure()

# %%

# combinatins of tax, I_gov, and risk to test:
comb = [("tax", [0.01, 0.03, 0.05], 0.85, 0.05),
        ("perc_guaranteed", 0.03, [0.75, 0.85, 0.95], 0.05),
        ("risk", 0.03, 0.85, [0.03, 0.05, 0.1])]

for (ResType, tax, perc_guaranteed, risk) in comb:
    CropAllocs, MaxAreas, labels, fn = \
        StoOpt.GetResultsToCompare(ResType = ResType,
                                   probF = 0.95,
                                   probS = 0.9,
                                   k = 9,
                                   k_using = 3,
                                   N = 50000,
                                   validation = 200000,
                                   tax = tax,
                                   perc_guaranteed = perc_guaranteed,
                                   risk = risk)
    try:
        StoOpt.CompareCropAllocs(CropAllocs = CropAllocs,
                                 MaxAreas = MaxAreas,
                                 labels = labels,
                                 title = "Representative Cluster",
                                 legend_title = ResType + ": ",
                                 comparing = "taxes", 
                                 filename = fn, 
                                 subplots = (1,3))
    except:
        continue

# %%

aim = "Similar"
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
              + str(2) + aim + ".txt", "rb") as fp:
        BestGrouping = pickle.load(fp)
    
for cluster_active in BestGrouping:
    print(cluster_active)
    CropAllocsPool, MaxAreasPool, labelsPool, fnPool = \
        StoOpt.GetResultsToCompare(ResType = "k_using",
                                   probF = 0.99,
                                   probS = 0.95,
                                   k = 9,
                                   k_using = cluster_active,
                                   N = 75000,
                                   validation = 200000,
                                   tax = 0.03,
                                   perc_guaranteed = 0.85,
                                   risk = 0.05)
    CropAllocsIndep, MaxAreasIndep, labelsIndep, fnIndep = \
        StoOpt.GetResultsToCompare(ResType = "k_using",
                                   probF = 0.95,
                                   probS = 0.85,
                                   k = 9,
                                   k_using = list(cluster_active),
                                   N = 50000,
                                   validation = 200000,
                                   tax = 0.03,
                                   perc_guaranteed = 0.75,
                                   risk = 0.05)
    try:
        StoOpt.CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, 
                                           MaxAreasPool, MaxAreasIndep, 
                                           labelsPool, labelsIndep, 
                                           filename = fnIndep,
                                           subplots = (2,1))
    except:
        continue
        
# %%

aim = "Similar"
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
              + str(2) + aim + ".txt", "rb") as fp:
        BestGrouping = pickle.load(fp)
    
CropAllocsPool, MaxAreasPool, labelsPool, fnPool = \
    StoOpt.GetResultsToCompare(ResType = "k_using",
                               probF = 0.99,
                               probS = 0.95,
                               k = 9,
                               k_using = BestGrouping,
                               N = 75000,
                               validation = 200000,
                               tax = 0.03,
                               perc_guaranteed = 0.85,
                               risk = 0.05)
CropAllocsIndep, MaxAreasIndep, labelsIndep, fnIndep = \
    StoOpt.GetResultsToCompare(ResType = "k_using",
                               probF = 0.99,
                               probS = 0.95,
                               k = 9,
                               k_using = StoOpt.MakeList(BestGrouping),
                               N = 200000,
                               validation = 50000,
                               tax = 0.03,
                               perc_guaranteed = 0.85,
                               risk = 0.05)
try:
    StoOpt.CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, 
                                       MaxAreasPool, MaxAreasIndep, 
                                       labelsPool, labelsIndep, 
                                       filename = fnPool,
                                       title = str(BestGrouping))
except:
    print("Nothing to plot")

        
# %%

comb = [
        (1, 200000, 50000),
        (2, 75000, 200000),
        # (3, 75000, 200000),
        (5, 100000, 250000)
        ]

for size, N, M in comb:
    for aim in ["Similar", "Dissimilar"]:
        if size == 1 and aim == "Dissimilar":
            continue
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + aim + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
        CropAllocs, MaxAreas, labels, fn = \
            StoOpt.GetResultsToCompare(ResType = "k_using",
                                       probF = 0.99,
                                       probS = 0.95,
                                       k = 9,
                                       k_using = BestGrouping,
                                       N = N,
                                       validation = M,
                                       tax = 0.03,
                                       perc_guaranteed = 0.85,
                                       risk = 0.05,
                                       prints = False, 
                                       groupSize = size,
                                       groupAim = aim)
        print(fn)
        try:
            StoOpt.CompareCropAllocs(CropAllocs = CropAllocs,
                                     MaxAreas = MaxAreas,
                                     labels = labels,
                                     title = "Groups of size " + str(size) + " (" + aim + "ity)",
                                     legend_title = "Cluster: ",
                                     comparing = "clusters", 
                                     filename = fn, 
                                     subplots = (3,3))
        except:
            continue
        
# %% 
        
CropAllocs, MaxAreas, labels, fn = \
    StoOpt.GetResultsToCompare(ResType = "k_using",
                               probF = 0.95,
                               probS = 0.85,
                               k = 9,
                               k_using = list(range(1,10)),
                               N = 50000,
                               validation = 200000,
                               tax = 0.03,
                               perc_guaranteed = 0.75,
                               risk = 0.05)       
try:
    StoOpt.CompareCropAllocs(CropAllocs = CropAllocs,
                             MaxAreas = MaxAreas,
                             labels = labels,
                             title = "Single cluster (lower probabilities)",
                             legend_title = "Cluster: ",
                             comparing = "clusters", 
                             filename = fn, 
                             subplots = (3,3))
except:
    print("Nothing to plot")
    

# %%

comb = [
        # (1, 200000, 50000),
        (2, 75000, 200000),
        # (3, 75000, 200000),
        # (5, 100000, 250000)
        ]

penalties = {"rhoF": [],
             "rhoS": []}
index =  []
for size, N, M in comb:
    for aim in ["Similar", "Dissimilar"]:
        if size == 1 and aim == "Dissimilar":
            continue
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + aim + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
        for gr in BestGrouping:
            try:
                crop_alloc, meta_sol, status, durations, settings, args, \
                rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
                    validation_values, fn = \
                        StoOpt.FoodSecurityProblem(probF = 0.99,
                                                   probS = 0.95,
                                                   k = 9,
                                                   k_using = gr,
                                                   N = N,
                                                   validation = M,
                                                   tax = 0.03,
                                                   perc_guaranteed = 0.85,
                                                   risk = 0.05)
                penalties["rhoF"].append(rhoF)
                penalties["rhoS"].append(rhoS)
                index.append(str(gr))
            except StoOpt.PenaltyException as e:
                penalties["rhoF"].append(np.nan)
                penalties["rhoS"].append(np.nan)
                index.append(str(gr))
                print(colored("Case " + str(gr) + " --- " + str(e), 'red'))
penalties = pd.DataFrame(penalties, columns = ["rhoF", "rhoS"])
penalties.index = index
penalties.to_csv("OtherResults/OverviewPenalties.csv")