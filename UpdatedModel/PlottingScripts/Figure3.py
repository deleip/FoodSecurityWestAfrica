# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:19:17 2021

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
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

if not os.path.isdir("Figures/PublicationPlots/Figure3"):
    os.mkdir("Figures/PublicationPlots/Figure3")
    
if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")

# %% ################ PANEL A, B - FOOD PRODUCTION DISTRIBUTION ###############

# Stationary scenario (fixed population, fixed yield distributions)
# Food production distribution (over all years, which is ok as all years behave
# the same as we use the stationary scenario) for changing probability for food 
# security
# Each cluster as separate plot (one good and one bad will be used in main text)

p = "fixed"
y = "fixed"

for cl in range(1, 10):
    panelAB = plt.figure(figsize = (14, 8))
    
    for (alpha, col) in [(0.5, "#62275D"),
                         (0.7, "#C32C57"),
                         (0.9, "#F38F1D"), 
                         (0.95, "#67D120"),
                         (0.99, "#2E6FCC")]:
        # get results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p,
                                       probF = alpha)
                    
        plt.hist(meta_sol["food_supply"].flatten()/args["demand"][0] * 100, bins = 200, alpha = 0.6,
                 density = True, color = col, label = r"$\alpha_f$ = " + str(alpha * 100) + "%")
    
    plt.axvline(100, color = "#003479", linestyle = "dashed", alpha = 0.6, label = "Food demand")
    plt.xlabel(r"Food production as share of demand (" + \
               str(np.round(args["demand"][0], 2)) + " $10^{12}\,kcal$" + "), %", fontsize = 24)
    plt.xticks(fontsize = 16)
    ax = plt.gca()
    ax.yaxis.set_ticks([])
    plt.title(r"Region " + str(cl), pad = 20, fontsize = 28)
    plt.legend(fontsize = 20)
        
    panelAB.savefig("Figures/PublicationPlots/Figure3/PanelAB_cl" + str(cl) + ".jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
            
    plt.close(panelAB)
    

# %% ################################ FOR SI ##################################

# Same as above for panel A and B, but all clusters in same figure as separate
# suplots, for SI.

p = "fixed"
y = "fixed"

SI1 = plt.figure(figsize = (14, 8))
    
SI1.subplots_adjust(hspace = 0.39)
for cl in range(1, 10):
    ax = SI1.add_subplot(3, 3, cl)
    for (alpha, col) in [(0.5, "#62275D"),
                         (0.7, "#C32C57"),
                         (0.9, "#F38F1D"), 
                         (0.95, "#67D120"),
                         (0.99, "#2E6FCC")]:
        # get results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p,
                                       probF = alpha)
                    
        plt.hist(meta_sol["food_supply"].flatten()/args["demand"][0] * 100, bins = 200, alpha = 0.6,
                 density = True, color = col)
    
    plt.axvline(100, color = "#003479", linestyle = "dashed", alpha = 0.6, label = "Food demand")
    plt.xticks(fontsize = 12)
    ax = plt.gca()
    ax.yaxis.set_ticks([])
    plt.title(r"Region " + str(cl), fontsize = 18)
        
# add a big axis, hide frame
SI1.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Food production as share of demand, %", fontsize = 24, labelpad = 20)
    
    
SI1.savefig("Figures/PublicationPlots/SI/FoodProduction.jpg", 
            bbox_inches = "tight", pad_inches = 1, format = "jpg")
        
plt.close(SI1)
    
# legend
SI1legend = plt.figure(figsize  = (5, 3))
legend_elements1 = [Line2D([0], [0], color ='#003479', lw = 2, ls = "dashed",
                          label="Food demand", alpha = 0.6),
                    Patch(color = "#62275D", label='50%', alpha = 0.6),
                    Patch(color = "#C32C57", label='70%', alpha = 0.6),
                    Patch(color = "#F38F1D", label='90%', alpha = 0.6),
                    Patch(color = "#67D120", label='95%', alpha = 0.6),
                    Patch(color = "#2E6FCC", label='99%', alpha = 0.6)
                    ]

ax = SI1legend.add_subplot(1, 1, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(handles = legend_elements1, fontsize = 14, loc = 6)

SI1legend.savefig("Figures/PublicationPlots/SI/FoodProductionsLegend.jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
plt.close(SI1legend)

# %% ######################## PANEL C - PARETO FRONT ##########################

# Input probability for food security vs. resulting total cultivation costs
# for different population and yield scenarios

panelC = plt.figure(figsize = (9, 16))
alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst case", "#C53B21"), 
                ("fixed", "fixed", "stationary", "#C9AF8C"),
                ("trend", "fixed", "best case", "#4B9A8D")]:

    costs = []
    
    for alpha in alphas:
        
        print("alpha: " + str(alpha) + "%, yield " + y + ", population " + p)
        
        tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                          out_type = "agg_sum",
                                          sizes = 1,
                                          yield_projection = y,
                                          pop_scenario = p,
                                          probF = alpha/100)
        
        costs.append(tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0])

    
    plt.scatter(alphas, costs, label = scen, color = col, s = 80)
    plt.plot(alphas, costs, color = col, lw = 2.5)

plt.xlabel("Input probability for food security, %", fontsize = 24)
plt.ylabel(r"Total cultivation costs, $10^9\$$", fontsize = 24)
plt.legend(fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
# plt.title("Trade-off between food security probability and cultivation costs")

panelC.savefig("Figures/PublicationPlots/Figure3/PanelC.jpg", 
            bbox_inches = "tight", pad_inches = 1, format = "jpg")

plt.close(panelC)

