# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:57:53 2021

@author: leip
"""
# set the right directory
import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import matplotlib.pyplot as plt
from ModelCode.GeneralSettings import figsize

if not os.path.isdir("Figures/PublicationPlots/Figure6"):
    os.mkdir("Figures/PublicationPlots/Figure6")

# %% ################# SCENARIO COMPARISONS WITH COOPERATION ##################

# for the 3 population/yield scenarios, this plots different output parameters
# over the different cooperation levels (using 2 different groupings)
# input probability and government parmeters use default values

 
print("\n  - Shaded_DiffScenarios_DiffGrouping", flush = True)
FS.CollectionPlotsCooperationAgg(panda_file = "current_panda", 
                         scenarionames = ["YieldFixedPopHigh_Equality",
                                          "YieldFixedPopFixed_Equality",
                                          "YieldTrendPopFixed_Equality",
                                          "YieldFixedPopHigh_Medoids",
                                          "YieldFixedPopFixed_Medoids",
                                          "YieldTrendPopFixed_Medoids"],
                         scenarios_shaded = True,
                         folder_comparisons = "Figure6",
                         fn_suffix = "_StandardScenarios_TwoGroupings",
                         publication_plot = True,
                         grouping_aim = "Similar",
                         grouping_metric = ["equality", "equality", "equality",
                                            "medoids", "medoids", "medoids"],
                         adjacent = [False, False, False, True, True, True],
                         yield_projection = ["fixed", "fixed", "trend",
                                            "fixed", "fixed", "trend"],
                         pop_scenario = ["High", "fixed", "fixed",
                                         "High", "fixed", "fixed"],
                         figsize = (16, 13))  

print("\n  - Shaded_DiffScenarios_DiffGrouping", flush = True)
FS.CollectionPlotsCooperationSingle(panda_file = "current_panda", 
                         scenarionames = ["YieldFixedPopHigh_Equality",
                                          "YieldFixedPopFixed_Equality",
                                          "YieldTrendPopFixed_Equality",
                                          "YieldFixedPopHigh_Medoids",
                                          "YieldFixedPopFixed_Medoids",
                                          "YieldTrendPopFixed_Medoids"],
                         folder_comparisons = "Figure6",
                         fn_suffix = "_StandardScenarios_TwoGroupings",
                         publication_plot = True,
                         grouping_aim = "Similar",
                         grouping_metric = ["equality", "equality", "equality",
                                            "medoids", "medoids", "medoids"],
                         adjacent = [False, False, False, True, True, True],
                         yield_projection = ["fixed", "fixed", "trend",
                                            "fixed", "fixed", "trend"],
                         pop_scenario = ["High", "fixed", "fixed",
                                         "High", "fixed", "fixed"],
                         figsize = (16, 23))  