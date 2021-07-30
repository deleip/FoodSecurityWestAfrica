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
        

# %% ############### 2. DEFAULT RUN FOR DIFFERENT GROUP TYPES  ################

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]

grouping_types = [#("Dissimilar", "Adj"),
                  #("Dissimilar", ""),
                  ("Similar", "Adj")
                  ]

for aim, adj in grouping_types:
    # if aim == "Dissimilar":
    #     continue
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N)
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N)

# %% #### 2.RUN WITH YIELD TREND AND MEDIUM POPULATION GROWTH FOR DIFFERENT GROUP TYPES  ####

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]


grouping_types = [#("Dissimilar", "Adj"),
                  #("Dissimilar", ""),
                  ("Similar", "Adj")
                  ]

for aim, adj in grouping_types:
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N,
                                           yield_projection = "trend",
                                           pop_scenario = "Medium")
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)        
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N,
                                               yield_projection = "trend",
                                               pop_scenario = "Medium")
            
# %% ############# 2.RUN WITH ONLY YIELD TRENDS ###############

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]


grouping_types = [#("Dissimilar", "Adj"),
                  #("Dissimilar", ""),
                  ("Similar", "Adj")
                  ]

for aim, adj in grouping_types:
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N,
                                           yield_projection = "trend",
                                           pop_scenario = "fixed")
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)        
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N,
                                               yield_projection = "trend",
                                               pop_scenario = "fixed")
    
# %% ########### 2. RUN WITH YIELD TRENDS AND LOW POPULATION GROWTH ###########

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]


grouping_types = [#("Dissimilar", "Adj"),
                  #("Dissimilar", ""),
                  ("Similar", "Adj")
                  ]

for aim, adj in grouping_types:
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N,
                                           yield_projection = "trend",
                                           pop_scenario = "Low")
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)        
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N,
                                               yield_projection = "trend",
                                               pop_scenario = "Low")     
                        
# %% ########### 2.RUN WITH YIELD TRENDS AND HIGH POPULATION GROWTH ############

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]


grouping_types = [#("Dissimilar", "Adj"),
                  #("Dissimilar", ""),
                  ("Similar", "Adj")
                  ]

for aim, adj in grouping_types:
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N,
                                           yield_projection = "trend",
                                           pop_scenario = "High")
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)        
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N,
                                               yield_projection = "trend",
                                               pop_scenario = "High")                
# %% ############# 2.RUN WITH ONLY POPULATION TRENDS ###############

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]


grouping_types = [#("Dissimilar", "Adj"),
                  #("Dissimilar", ""),
                  ("Similar", "Adj")
                  ]

for aim, adj in grouping_types:
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N,
                                           yield_projection = "fixed",
                                           pop_scenario = "High")
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)        
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N,
                                               yield_projection = "fixed",
                                               pop_scenario = "High")
                
                
# %% ######## Custom grouping

gs1 = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
gs2 = [(2, 9), (1, 4), (5, 8), (3, 6), (7,)]
gs3 = [(2, 6, 9), (1, 3, 4), (5, 7, 8)]
gs5 = [(1, 2, 9, 4), (3, 5, 7, 8, 6)]
gs9 = [(1, 2, 9, 4, 3, 5, 7, 8, 6)]


with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(1) + "Custom" + ".txt", "wb") as fp:
    pickle.dump(gs1, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(2) + "Custom" + ".txt", "wb") as fp:
    pickle.dump(gs2, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                         + str(3) + "Custom" + ".txt", "wb") as fp:
    pickle.dump(gs3, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                          + str(5) + "Custom" + ".txt", "wb") as fp:
    pickle.dump(gs5, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                          + str(9) + "Custom" + ".txt", "wb") as fp:
    pickle.dump(gs9, fp)
                
   
# %% ############### 2. DEFAULT RUN FOR DIFFERENT GROUP TYPES  ################

# group size, sample size N, validation sample size M
comb1 = [(1, 10000, 50000),
        (2, 20000, 50000),
        (3, 50000, 100000),
        (5, 100000, 200000),
        ("all", 100000, 200000)
        ]

comb2 = [(1, 20000, 50000),
        (2, 40000, 100000),
        (3, 100000, 200000),
        (5, 200000, 300000),
        ("all", 250000, 400000)
        ]

comb3 = [(2, 100000, 200000),
        (3, 250000, 400000),
        (5, 400000, 500000),
        ("all", 500000, 600000)
        ]

combs = [comb1]

grouping_types = [("Custom", "")]

for aim, adj in grouping_types:
    # if aim == "Dissimilar":
    #     continue
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for idx, comb in enumerate(combs):
        for size, N, M in comb:
            if size == "all":
                print("\u2017"*65)
                print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: all, comb: " + str(idx + 1))
                print("\u033F "*65)
                settings, args, AddInfo_CalcParameters, yield_information, \
                population_information, status, durations, crop_alloc, meta_sol, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = M,
                                           plotTitle = "All clusters",
                                           k_using = size,
                                           N = N)
            else:
                with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(size) + aim + adj + ".txt", "rb") as fp:
                        BestGrouping = pickle.load(fp)
                        
                for cluster_active in BestGrouping:
                    print("\u2017"*65)
                    print("Aim: " + aim + ", adjacent: " + adj_text + ", size: " + str(size) + ", clusters: " + str(cluster_active) + ", comb: " + str(idx + 1))
                    print("\u033F "*65)
                    
                    settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, status, durations, crop_alloc, meta_sol, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N)                
   
# %% ##### 3. PLOTTING RESULTS  #####

plot_crop_areas = False

for (aim, adj) in [("Dissimilar", False),
                   ("Similar", True)]:
    if adj == "Adj":
        adj_text = "True"
    else:
        adj_text = "False"
    for (pop_scenario, yield_projection) in [("fixed", "fixed"),
                                             ("Medium", "trend")]:
        print("\u2017"*65)
        print("Aim: " + aim + ", adjacent: " + adj_text + ", population: " + pop_scenario + ", yield: " + yield_projection)
        print("\u033F "*65)
        
        if plot_crop_areas:
            print("\nPlotting crop areas", flush = True)
            FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                                groupAim = aim,
                                                adjacent = adj,
                                                console_output = None,
                                                yield_projection = yield_projection,
                                                pop_scenario = pop_scenario)
        
        print("\nPlotting coooperation plots", flush = True)
        FS.PandaPlotsCooperation(panda_file = "current_panda", 
                                        grouping_aim = aim,
                                        adjacent = adj,
                                        yield_projection = yield_projection,
                                        pop_scenario = pop_scenario)
        
        print("\n\nPlotting other plots", flush = True)
        FS.OtherPandaPlots(panda_file = "current_panda", 
                           grouping_aim = aim,
                           adjacent = adj,
                           yield_projection = yield_projection,
                           pop_scenario = pop_scenario)
    
# %% ##### 3. PLOTTING SCENARIO COMPARISONS  #####
    
print("\nPlotting scenario comparison plots", flush = True)
print("\n  - DissimNonAdj_FixedVsTrend", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["Fixed", "Trend"],
                         folder_comparisons = "DissimNonAdj_FixedVsTrend",
                         grouping_aim = "Dissimilar",
                         adjacent = False,
                         yield_projection = ["fixed", "trend"],
                         pop_scenario = ["fixed", "Medium"])   

print("\n  - SimAdj_FixedVsTrend", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["Fixed", "Trend"],
                         folder_comparisons = "SimAdj_FixedVsTrend",
                         grouping_aim = "Similar",
                         adjacent = True,
                         yield_projection = ["fixed", "trend"],
                         pop_scenario = ["fixed", "Medium"])   

print("\n  - Fixed_DissimNonAdjVsSimAdj", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["DissimNonAdj", "SimAdj"],
                         folder_comparisons = "Fixed_DissimNonAdjVsSimAdj",
                         grouping_aim = ["Dissimilar", "Similar"],
                         adjacent = [False, True],
                         yield_projection = "fixed",
                         pop_scenario = "fixed")   

print("\n  - Trend_DissimNonAdjVsSimAdj", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["DissimNonAdj", "SimAdj"],
                         folder_comparisons = "Trend_DissimNonAdjVsSimAdj",
                         grouping_aim = ["Dissimilar", "Similar"],
                         adjacent = [False, True],
                         yield_projection = "trend",
                         pop_scenario = "Medium")


print("\n  - YieldTrend_LowVsMediumVsHighPop", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["low population growth", "medium population growth", "high population growth"],
                         folder_comparisons = "YieldTrend_LowVsMediumVsHighPop",
                         grouping_aim = "Similar",
                         adjacent = True,
                         yield_projection = "trend",
                         pop_scenario = ["Low", "Medium", "High"])


print("\n  - OnlyYieldTrendVsOnlyHighPopTrendVsNoTrend", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["only yield trend", "only high population growth", "no trends"],
                         folder_comparisons = "OnlyYieldTrendVsOnlyHighPopTrendVsNoTrend",
                         grouping_aim = "Similar",
                         adjacent = True,
                         yield_projection = ["trend", "fixed", "fixed"],
                         pop_scenario = ["fixed", "High", "fixed"])
     
# %%
print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                        grouping_aim = "Similar",
                        adjacent = True,
                        yield_projection = "fixed",
                        pop_scenario = "fixed")

# %% Plotting results for runs without trend, custom grouping

print("\nPlotting crop areas", flush = True)
FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Custom",
                                    adjacent = False,
                                    console_output = None,
                                    yield_projection = "fixed",
                                    pop_scenario = "fixed")

print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                                grouping_aim = "Custom",
                                adjacent = False,
                                yield_projection = "fixed",
                                pop_scenario = "fixed")

print("\n\nPlotting other plots", flush = True)
FS.OtherPandaPlots(panda_file = "current_panda", 
                   grouping_aim = "Custom",
                   adjacent = False,
                   yield_projection = "fixed",
                   pop_scenario = "fixed") 

# %% Plotting results for runs with trend, dissimilar, non-adjacent

print("\nPlotting crop areas", flush = True)
FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Dissimilar",
                                    adjacent = False,
                                    console_output = None,
                                    yield_projection = "trend",
                                    pop_scenario = "Medium")

print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                                grouping_aim = "Dissimilar",
                                adjacent = False,
                                yield_projection = "trend",
                                pop_scenario = "Medium")

print("\n\nPlotting other plots", flush = True)
FS.OtherPandaPlots(panda_file = "current_panda", 
                   grouping_aim = "Dissimilar",
                   adjacent = False,
                   yield_projection = "trend",
                   pop_scenario = "Medium")


# %% Plotting results for runs without trend, dissimilar, non-adjacent

print("\nPlotting crop areas", flush = True)
FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Dissimilar",
                                    adjacent = False,
                                    console_output = None)

print("\nPlotting cooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         grouping_aim = "Dissimilar",
                         adjacent = False)

print("\n\nPlotting other plots", flush = True)
FS.OtherPandaPlots(panda_file = "current_panda", 
                   grouping_aim = "Dissimilar",
                   adjacent = False)

# %% Comparing results for runs with and without trends (dissimilar, non-adjacent)

print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                        grouping_aim = "Dissimilar",
                        adjacent = False,
                        yield_projection = ["fixed", "trend"],
                        pop_scenario = ["fixed", "Medium"],
                        scenarionames = ["fixed", "trends"])




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
    FS.FoodSecurityProblem(validation_size = 50000,
                           k_using = [6], 
                           N =  10000,
                           yield_projection = "trend",
                           pop_scenario = "Medium")
    
rho, rhos_tried_order, rhos_tried, crop_allocs, \
probabilities, necessary_help, file, \
objective = FS.LoadPenaltyStuff(objective = "F",
                           validation_size = 50000,
                           k_using = [4,7],
                           N = 20000)

# %%

settings, args, AddInfo_CalcParameters, yield_information, \
population_information, status, durations, crop_alloc, meta_sol, \
crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
    FS.FoodSecurityProblem(validation_size = 50000,
                           k_using = [5], 
                           N = 8000)