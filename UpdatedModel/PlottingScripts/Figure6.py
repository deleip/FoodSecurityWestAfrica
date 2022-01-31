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
from ModelCode.PandaPlotFunctions import PlotPandaAggregate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

if not os.path.isdir("Figures/PublicationPlots/Figure6"):
    os.mkdir("Figures/PublicationPlots/Figure6")

from PlottingScripts.PlottingSettings import publication_colors


# %% ################# SCENARIO COMPARISONS WITH COOPERATION ##################

# for the 3 population/yield scenarios, this plots different output parameters
# over the different cooperation levels (using 2 different groupings)
# input probability and government parmeters use default values

panda_file = "current_panda"

output_vars = ["Resulting probability for food security",
               "Average aggregate food shortage per capita (including only samples that have shortage)",
               "Resulting probability for solvency",
               "Average aggregate debt after payout per capita (including only samples with negative final fund)",
               # or: "Average aggregate debt after payout per capita (including only samples with catastrophe)",
               "Average yearly total cultivated area",
               "Total cultivation costs (sto. solution)"]

ylabels = ["Target probability for food security, %",
           r"Average food shortage per capita, $10^{3}\,$kcal",
           "Probability for solvency, %",
           r"Average debt after payout per capita, $10^9\,\$$",
           # or: "Average aggregate debt after payout per capita (including only samples with catastrophe)",
           r"Average yearly total cultivated area, $10^9\,$ha",
           r"Total cultivation costs, $10^9\,\$$"]

agg_types = ["agg_avgweight",
             "agg_avgweight",
             "agg_avgweight",
             "agg_avgweight",
             "agg_sum",
             "agg_sum"]

var_weights = ["Share of West Africa's population that is living in total considered region (2015)",
               "Share of West Africa's population that is living in total considered region (2015)",
               "Share of West Africa's population that is living in total considered region (2015)",
               "Share of West Africa's population that is living in total considered region (2015)",
               None, 
               None]

weight_titles = ["population",
                 "population",
                 "population",
                 "population",
                 None,
                 None]

scenarionames = ["YieldFixedPopHigh_Equality",
                 "YieldFixedPopFixed_Equality",
                 "YieldTrendPopFixed_Equality",
                 "YieldFixedPopHigh_Medoids",
                 "YieldFixedPopFixed_Medoids",
                 "YieldTrendPopFixed_Medoids"]

subplot_titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

scenarios_shaded = True

grouping_aim = "Similar"
grouping_metric = ["equality", "equality", "equality",
                   "medoids", "medoids", "medoids"]
adjacent = [False, False, False, True, True, True]
yield_projection = ["fixed", "fixed", "trend",
                   "fixed", "fixed", "trend"]
pop_scenario = ["High", "fixed", "fixed",
                "High", "fixed", "fixed"]

# din a4 is 8.5 x 14 inches                         
figsize = (22, 29)


fig = PlotPandaAggregate(panda_file = panda_file,
                   agg_type = agg_types,
                   var_weight = var_weights,
                   weight_title = weight_titles,
                   output_var = output_vars,
                   scenarionames = scenarionames,
                   scenarios_shaded = scenarios_shaded,
                   grouping_aim = grouping_aim,
                   grouping_metric = grouping_metric,
                   adjacent = adjacent,
                   yield_projection = yield_projection,
                   pop_scenario = pop_scenario,
                   plt_legend = True,
                   ylabels = ylabels,
                   subplot_titles = subplot_titles,
                   plt_title = True,
                   close_plots = False,
                   foldername = "PublicationPlots/",
                   cols = [publication_colors["blue"], publication_colors["red"]],
                   figsize = figsize)

ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
legend_elements = [Line2D([0], [0], color ='black', lw = 2, 
                      label='worst case'),
                Line2D([0], [0], color ='black', lw = 2, ls = "dashdot",
                      label='stationary'),
                Line2D([0], [0], color ='black', lw = 2, ls = "--",
                      label='best case'),
                Patch(color ='royalblue', alpha = 0.6, label = 'equality grouping'),
                Patch(color ='darkred', alpha = 0.6, label = 'proximity grouping')]
ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (0.5, -0.06),
          loc = "upper center", ncol = 2)

fig.savefig("Figures/PublicationPlots/Figure6_Cooperation.jpg", bbox_inches = "tight", pad_inches = 1)

# %% ################# SCENARIO COMPARISONS WITH COOPERATION ##################

# for the 3 population/yield scenarios, this plots different output parameters
# over the different cooperation levels (using 2 different groupings)
# input probability and government parmeters use default values

# WITHOUT LEGENDS

print("\n  - Shaded_DiffScenarios_DiffGrouping aggregated plots w/o legend", flush = True)
FS.CollectionPlotsCooperationAgg(panda_file = "current_panda", 
                         scenarionames = ["YieldFixedPopHigh_Equality",
                                          "YieldFixedPopFixed_Equality",
                                          "YieldTrendPopFixed_Equality",
                                          "YieldFixedPopHigh_Medoids",
                                          "YieldFixedPopFixed_Medoids",
                                          "YieldTrendPopFixed_Medoids"],
                         scenarios_shaded = True,
                         folder_comparisons = "Figure6",
                         fn_suffix = "_StandardScenarios_TwoGroupings_woLegend",
                         publication_plot = True,
                         grouping_aim = "Similar",
                         grouping_metric = ["equality", "equality", "equality",
                                            "medoids", "medoids", "medoids"],
                         adjacent = [False, False, False, True, True, True],
                         yield_projection = ["fixed", "fixed", "trend",
                                            "fixed", "fixed", "trend"],
                         pop_scenario = ["High", "fixed", "fixed",
                                         "High", "fixed", "fixed"],
                         figsize = (16, 13),
                         plt_legend = False)  

print("\n  - Shaded_DiffScenarios_DiffGrouping non-aggregated plots w/o legend", flush = True)
FS.CollectionPlotsCooperationSingle(panda_file = "current_panda", 
                         scenarionames = ["YieldFixedPopHigh_Equality",
                                          "YieldFixedPopFixed_Equality",
                                          "YieldTrendPopFixed_Equality",
                                          "YieldFixedPopHigh_Medoids",
                                          "YieldFixedPopFixed_Medoids",
                                          "YieldTrendPopFixed_Medoids"],
                         folder_comparisons = "Figure6",
                         fn_suffix = "_StandardScenarios_TwoGroupings_woLegend",
                         publication_plot = True,
                         grouping_aim = "Similar",
                         grouping_metric = ["equality", "equality", "equality",
                                            "medoids", "medoids", "medoids"],
                         adjacent = [False, False, False, True, True, True],
                         yield_projection = ["fixed", "fixed", "trend",
                                            "fixed", "fixed", "trend"],
                         pop_scenario = ["High", "fixed", "fixed",
                                         "High", "fixed", "fixed"],
                         figsize = (16, 23),
                         plt_legend = False) 


# WITH LEGENDS

print("\n  - Shaded_DiffScenarios_DiffGrouping aggregated plots w/ legend", flush = True)
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
                         figsize = (16, 13),
                         plt_legend = True)  

print("\n  - Shaded_DiffScenarios_DiffGrouping non-aggregated plots w/ legend", flush = True)
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
                         figsize = (16, 23),
                         plt_legend = True) 


# %% Checking behavior of resulting food security probabiltiy 

# it seems, that the resulting food security probability for the stationary 
# scenario is missing -> probably exactly beneth the best case scenario?

# For metric equality this makes sense: for group size = 1 they are all the 
# same anyway, and for all bigger group sizes the stationary scenario is 
# already at the highest probability (input probability 99%), and for the 
# best case scenario it is even easier to reach this, so it is also at that 
# probability
# For metric medoids, the same applies for size = 1 and size >= 3, the only
# case that we still need to check is size = 2.

# plotting both scenarios separately to check that they are really the same
FS.PlotPandaAggregate(panda_file = "current_panda",
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var = ['Resulting probability for food security'],
                       scenarionames = ["equality", "medoids"],
                       grouping_aim = "Similar",
                       grouping_metric = ["equality", "medoids"],
                       adjacent = [False, True],
                       plt_legend = True,
                       close_plots = False,
                       figsize =  (16, 23),
                       pop_scenario = "fixed",
                       yield_projection = "fixed")


FS.PlotPandaAggregate(panda_file = "current_panda",
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var = ['Resulting probability for food security'],
                       scenarionames = ["equality", "medoids"],
                       grouping_aim = "Similar",
                       grouping_metric = ["equality", "medoids"],
                       adjacent = [False, True],
                       plt_legend = True,
                       close_plots = False,
                       figsize =  (16, 23),
                       pop_scenario = "fixed",
                       yield_projection = "trend")
# -> yes, look exactly the same

# checking underlying data that was plotted (already aggregated over cluster groups per size)
res_stationary = FS.Panda_GetResultsSingScen(file = "current_panda", 
                             out_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       output_var = ['Resulting probability for food security'],
                           grouping_aim = "Similar",
                           grouping_metric = "medoids",
                           adjacent = True,
                           sizes = [1, 2, 3, 5, 9],
                       pop_scenario = "fixed",
                       yield_projection = "fixed")


res_trend = FS.Panda_GetResultsSingScen(file = "current_panda", 
                             out_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       output_var = ['Resulting probability for food security'],
                           grouping_aim = "Similar",
                           grouping_metric = "medoids",
                           adjacent = True,
                           sizes = [1, 2, 3, 5, 9],
                       pop_scenario = "fixed",
                       yield_projection = "trend")


# checking underlying data per cluster group of size 2 for medoids metric
import pickle
grouping_metric = "medoids"
size = 2
grouping_aim = "Similar"
add = "Adj"

with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric.capitalize() + 
          "Size"  + str(size) + grouping_aim + add + ".txt", "rb") as fp:
        BestGrouping = pickle.load(fp)

panda_tmp_stationary = FS.ReadFromPanda(file = "current_panda", \
                              output_var = 'Resulting probability for food security', \
                              k_using = BestGrouping, \
                       pop_scenario = "fixed",
                       yield_projection = "fixed")

panda_tmp_trend = FS.ReadFromPanda(file = "current_panda", \
                              output_var = 'Resulting probability for food security', \
                              k_using = BestGrouping, \
                       pop_scenario = "fixed",
                       yield_projection = "trend")
    
# in both cases, the cluster that is one its own (cluster 4) is not food secure
# at all (probability 0%), while all other groups are at the highest probabilty
# (99%) -> resulting overall probabiltiy is the same.