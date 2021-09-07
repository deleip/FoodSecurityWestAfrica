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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                    
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                    
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                    
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                    
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                        
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                    
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
                    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                        FS.FoodSecurityProblem(validation_size = M,
                                               plotTitle = "Aim: " + aim  + ", Adjacent: " + adj_text,
                                               k_using = list(cluster_active),
                                               N = N,
                                               yield_projection = "fixed",
                                               pop_scenario = "High")
                
                
# %% ######## Custom grouping (based on crop production)

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
                
# %% ######## Custom grouping (based on profit generation)

gs1 = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
gs2 = [(2, 8), (1, 9), (5, 7), (4, 6), (3,)]
gs3 = [(2, 7, 4), (1, 3, 9), (5, 6, 8)]
gs5 = [(1, 2, 8, 9), (3, 4, 5, 6, 7)]
gs9 = [(1, 2, 9, 4, 3, 5, 7, 8, 6)]


with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(1) + "Custom_Profit" + ".txt", "wb") as fp:
    pickle.dump(gs1, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                              + str(2) + "Custom_Profit" + ".txt", "wb") as fp:
    pickle.dump(gs2, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                         + str(3) + "Custom_Profit" + ".txt", "wb") as fp:
    pickle.dump(gs3, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                          + str(5) + "Custom_Profit" + ".txt", "wb") as fp:
    pickle.dump(gs5, fp)
with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                          + str(9) + "Custom_Profit" + ".txt", "wb") as fp:
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

grouping_types = [("Custom_Profit", "")]

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
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
                    
                    settings, args, yield_information, population_information, \
                    status, durations, exp_incomes, crop_alloc, meta_sol, \
                    crop_allocF, meta_solF, crop_allocS, meta_solS, \
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
 
print("\n  - CustomProfit_CustomCropProd_SimilarAdj", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                         scenarionames = ["Custom_Profit", "Custom_CropProd", "SimAdj"],
                         folder_comparisons = "CustomProfit_CustomCropProd_SimilarAdj",
                         grouping_aim = ["Custom_Profit", "Custom", "Similar"],
                         adjacent = [False, False, True],
                         yield_projection = "fixed",
                         pop_scenario = "fixed")  

    
# %%
print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                        grouping_aim = "Similar",
                        adjacent = True,
                        yield_projection = "fixed",
                        pop_scenario = "fixed")

# %% Plotting results for runs without trend, custom grouping (crop production)

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

# %% Plotting results for runs without trend, custom grouping (profit generation)

print("\nPlotting crop areas", flush = True)
FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Custom_Profit",
                                    adjacent = False,
                                    console_output = None,
                                    yield_projection = "fixed",
                                    pop_scenario = "fixed")

print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                                grouping_aim = "Custom_Profit",
                                adjacent = False,
                                yield_projection = "fixed",
                                pop_scenario = "fixed")

print("\n\nPlotting other plots", flush = True)
FS.OtherPandaPlots(panda_file = "current_panda", 
                   grouping_aim = "Custom_Profit",
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

# %% Plotting results for runs without trend, similar, adjacent

print("\nPlotting crop areas", flush = True)
FS.CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Similar",
                                    adjacent = True,
                                    console_output = None,
                                    yield_projection = "fixed",
                                    pop_scenario = "fixed")

print("\nPlotting coooperation plots", flush = True)
FS.PandaPlotsCooperation(panda_file = "current_panda", 
                                grouping_aim = "Similar",
                                adjacent = True,
                                yield_projection = "fixed",
                                pop_scenario = "fixed")

print("\n\nPlotting other plots", flush = True)
FS.OtherPandaPlots(panda_file = "current_panda", 
                   grouping_aim = "Similar",
                   adjacent = True,
                   yield_projection = "fixed",
                   pop_scenario = "fixed")


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




# %%

settings, args, yield_information, population_information, \
status, durations, exp_incomes, crop_alloc, meta_sol, \
crop_allocF, meta_solF, crop_allocS, meta_solS, \
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

settings, args, yield_information, population_information, \
status, durations, exp_incomes, crop_alloc, meta_sol, \
crop_allocF, meta_solF, crop_allocS, meta_solS, \
crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
    FS.FoodSecurityProblem(validation_size = 200000,
                           k_using = "all", 
                           N = 100000)
    
    
settings, args, yield_information, population_information, \
status, durations, exp_incomes, crop_alloc, meta_sol_direct, \
crop_allocF, meta_solF_direct, crop_allocS, meta_solS_direct, \
crop_alloc_vs, meta_sol_vss_direct, VSS_value, validation_values, fn = \
    FS.FoodSecurityProblem(validation_size = 5000,
                           k_using = [5], 
                           N = 1000)