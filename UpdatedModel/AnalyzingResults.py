# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:45:49 2020

@author: leip
"""

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import numpy as np
import pickle
from termcolor import colored
import pandas as pd

# set up folder structure (if not already done)
FS.CheckFolderStructure()

# %%

# combinatins of tax, I_gov, and risk to test:
comb = [("tax", [0.01, 0.03, 0.05], 0.85, 0.05),
        ("perc_guaranteed", 0.03, [0.75, 0.85, 0.95], 0.05),
        ("risk", 0.03, 0.85, [0.03, 0.05, 0.1])]

for (ResType, tax, perc_guaranteed, risk) in comb:
    CropAllocs, MaxAreas, labels, fn = \
        FS.GetResultsToCompare(ResType = ResType,
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
        FS.CompareCropAllocs(CropAllocs = CropAllocs,
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

aim = "Dissimilar"
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
              + str(2) + aim + ".txt", "rb") as fp:
        BestGrouping = pickle.load(fp)
    
for cluster_active in BestGrouping:
    print(cluster_active)
    CropAllocsPool, MaxAreasPool, labelsPool, fnPool = \
        FS.GetResultsToCompare(k_using = cluster_active,
                                N = 75000,
                                validation = 200000,
                                tax = 0.03,
                                perc_guaranteed = 0.85,
                                risk = 0.05)
    CropAllocsIndep, MaxAreasIndep, labelsIndep, fnIndep = \
        FS.GetResultsToCompare(ResType = "k_using",
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
        FS.CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, 
                                           MaxAreasPool, MaxAreasIndep, 
                                           labelsPool, labelsIndep, 
                                           filename = fnIndep,
                                           subplots = (2,1))
    except:
        continue
        
# %%

comb = [(1, 15000, 100000),
        (2, 30000, 200000),
        (3, 50000, 200000),
        (5, 800000, 300000)
        ]
aim = "Dissimilar"
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
              + str(comb[0][0]) + aim + ".txt", "rb") as fp:
        BestGrouping1 = pickle.load(fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
              + str(comb[1][0]) + aim + ".txt", "rb") as fp:
        BestGrouping2 = pickle.load(fp)
    
CropAllocsIndep, MaxAreasIndep, labelsIndep, fnIndep = \
    FS.GetResultsToCompare(k_using = BestGrouping1,
                            N = comb[0][1],
                            validation = comb[0][2])
CropAllocsPool, MaxAreasPool, labelsPool, fnPool = \
    FS.GetResultsToCompare(k_using = BestGrouping2,
                            N = comb[1][1],
                            validation = comb[1][2])

FS.CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, 
                                       MaxAreasPool, MaxAreasIndep, 
                                       labelsPool, labelsIndep, 
                                       filename = fnPool,
                                       title = str(BestGrouping))


        
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
            FS.GetResultsToCompare(ResType = "k_using",
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
            FS.CompareCropAllocs(CropAllocs = CropAllocs,
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
    FS.GetResultsToCompare(ResType = "k_using",
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
    FS.CompareCropAllocs(CropAllocs = CropAllocs,
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
                        FS.FoodSecurityProblem(probF = 0.99,
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
            except FS.PenaltyException as e:
                penalties["rhoF"].append(np.nan)
                penalties["rhoS"].append(np.nan)
                index.append(str(gr))
                print(colored("Case " + str(gr) + " --- " + str(e), 'red'))
penalties = pd.DataFrame(penalties, columns = ["rhoF", "rhoS"])
penalties.index = index
penalties.to_csv("OtherResults/OverviewPenalties.csv")