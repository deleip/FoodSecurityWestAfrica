# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:41:02 2022

@author: leip
"""

# %% #########################     SETTINGS      ##############################


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
from string import ascii_uppercase as letter

from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters

from ModelCode.PlotMaps import PlotClusterGroups

if not os.path.isdir("Figures/PublicationPlots"):
    os.mkdir("Figures/PublicationPlots")
    


# %% ######################### FIG 1 - MAP OF CLUSTERS ########################

PlotClusterGroups(k = 9, title = "", plot_cmap = False, 
                  basecolors = list(publication_colors.values()),
                  close_plt = False, figsize = (20, 15),
                  file = "Figures/PublicationPlots/Figure1_MapClusters")

PlotClusterGroups(k = 9, title = "", figsize = (20, 15), 
                  basecolors = list(publication_colors.values()),
                  file = "Figures/PublicationPlots/Figure1_MapClusters_wCmap")

# %% ############# FIG 3 - FOOD PRODUCTION AND CULTIVATION COSTS ##############

fig, axd = plt.subplot_mosaic([["upper left", "right"],
                          ["lower left", "right"]], figsize = (25, 16),
                         gridspec_kw = {"width_ratios" : [1.3, 1]})
fig.subplots_adjust(hspace = 0.3)

## PANEL A, B - FOOD PRODUCTION DISTRIBUTION 

# Stationary scenario (fixed population, fixed yield distributions)
# Food production distribution (over all years, which is ok as all years behave
# the same as we use the stationary scenario) for changing probability for food 
# security
# Each cluster as separate plot (one good and one bad will be used in main text)

p = "fixed"
y = "fixed"

for (cl, panel) in [(4, "(b)"), (5, "(a)")]:
    if cl == 4:
        ax = axd["lower left"]
    else:
        ax = axd["upper left"]
    
    for (alpha, col) in [(0.5, publication_colors["purple"]),
                          (0.7, publication_colors["red"]),
                          (0.9, publication_colors["orange"]), 
                          (0.95, publication_colors["yellow"]),
                         (0.99, publication_colors["green"])]:
        # get results
        settings, args, yield_information, population_information, penalty_methods, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p,
                                       probF = alpha)
                    
        ax.hist(meta_sol["food_supply"].flatten()/args["demand"][0] * 100, bins = 200, alpha = 0.6,
                 density = True, color = col, label = r"$\alpha$ = " + str(alpha * 100) + "%")
    
    ax.axvline(100, color = "#003479", linestyle = "dashed", alpha = 0.6, label = "Food demand", linewidth = 2.5)
    ax.set_xlabel(r"Food production as percentage of demand (" + \
               str(np.round(args["demand"][0])) + " $10^{12}\,kcal$" + ")", fontsize = 24)
    ax.set_ylabel(r"Probability density", fontsize = 24)
    ax.tick_params(axis = "both", labelsize = 20)
    # ax = plt.gca()
    if cl == 5:
        ax.set_xlim(left = 30)
    else:
        ax.set_xlim(left = 15)
        
        
    # ax.yaxis.set_ticks([])
    ax.set_title("    " + panel + r" Region " + cluster_letters[cl-1], pad = 20, fontsize = 28, loc = "left")
    ax.legend(fontsize = 22, loc = "upper left")
        

## PANEL C - PARETO FRONT 

# Input probability for food security vs. resulting total cultivation costs
# for different population and yield scenarios

ax = axd["right"]
alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst-case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best-case", publication_colors["green"])
                ]:

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

    
    ax.scatter(alphas, costs, label = scen, color = col, s = 80)
    ax.plot(alphas, costs, color = col, lw = 2.5)

ax.set_title(r"    (c)", pad = 20, fontsize = 28, loc = "left")
ax.set_xlabel("Reliability target for food security, %", fontsize = 24)
ax.set_ylabel(r"Total cultivation costs, $10^9\$$", fontsize = 24)
ax.legend(fontsize = 22, loc = "upper left")
ax.tick_params(axis = "both", labelsize = 20)

fig.savefig("Figures/PublicationPlots/Figure3_FoodProdAndCosts.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

plt.close(fig)

# %% ########################### FIG 4 - CROP AREAS ###########################

# two yield/population scenarios as inner and outer pie chart
# all 9 clusters as columns (separate pie charts)
# two time steps as rows
# default probability and government parameters
# crop areas for maize and rice, and unused area

fig4 = plt.figure(figsize = (15, 7))

fig4.subplots_adjust(wspace=0.005)

colors = [publication_colors["green"], publication_colors["yellow"], publication_colors["grey"]]
size = 0.5

ax_tmp = fig4.add_subplot(4, 10, 1, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.62, "t = 2020 \n" + r"$\alpha$ = 99%", fontsize = 18)
    
ax_tmp = fig4.add_subplot(4, 10, 11, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.62, "t = 2030 \n" + r"$\alpha$ = 99%", fontsize = 18)

ax_tmp = fig4.add_subplot(4, 10, 21, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.62, "t = 2030 \n" + r"$\alpha$ = 90%", fontsize = 18)

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))

    # get results
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_worst99, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High")
                
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_best99, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed")            
                    
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_worst90, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High",
                                   probF = 0.9)
                
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_best90, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed",
                                   probF = 0.9) 
    # settings, args, yield_information, population_information, penalty_methods,  \
    # status, all_durations, exp_incomes, crop_alloc_fixed, meta_sol, \
    # crop_allocF, meta_solF, crop_allocS, meta_solS, \
    # crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
    #             FS.LoadFullResults(k_using = cl,
    #                                 yield_projection = "fixed",
    #                                 pop_scenario = "fixed")    
                
    def _getAreas(year, crops):
        year_rel = year - settings["sim_start"]
        areas = [crops[year_rel,0,0], 
                crops[year_rel,1,0], 
                round(args["max_areas"][0] - np.sum(crops[year_rel,:,0]), 5)]
        return(areas)
    
    pos = letter.index(cluster_letters[cl-1]) + 1
    
    ax_tmp = fig4.add_subplot(4, 10, pos + 1)
    areas_outer = _getAreas(2020, crop_alloc_worst99)
    areas_inner = _getAreas(2020, crop_alloc_best99)
    print("                 1. outer: " + str(areas_outer) + ", inner: " + str(areas_inner))
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"),
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8),
               startangle = 180, counterclock = False)
    ax_tmp.set_title(cluster_letters[cl-1], fontsize = 18)
        
    
    ax_tmp = fig4.add_subplot(4, 10, pos + 11)
    areas_outer = _getAreas(2030, crop_alloc_worst99)
    areas_inner = _getAreas(2030, crop_alloc_best99)
    print("                 2. outer: " + str(areas_outer) + ", inner: " + str(areas_inner))
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"), 
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8), 
               startangle = 180, counterclock = False)
    
    ax_tmp = fig4.add_subplot(4, 10, pos + 21)
    areas_outer = _getAreas(2030, crop_alloc_worst90)
    areas_inner = _getAreas(2030, crop_alloc_best90)
    print("                 3. outer: " + str(areas_outer) + ", inner: " + str(areas_inner))
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"), 
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8), 
               startangle = 180, counterclock = False)
    
ax = fig4.add_subplot(4, 1, 4, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

legend_elements = [Patch(color = colors[0], alpha = 0.9, label='Rice'),
                   Patch(color = colors[1], alpha = 0.9, label='Maize'),
                   Patch(color = colors[2], alpha = 0.9, label='Not used'),
                   Patch(color = "w", label='Inner circle: best-case scenario'),
                   Patch(color = "w", label='Outer circle: worst-case scenario')]
ax.legend(handles = legend_elements, fontsize = 18,
          loc = "center", ncol = 2)


fig4.savefig("Figures/PublicationPlots/Figure4_CropAreas.jpg", 
                bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

# %% ####################### FIG 4 - CROP AREASVS 2 ###########################

# two yield/population scenarios as inner and outer pie chart
# all 9 clusters as columns (separate pie charts)
# two time steps as rows
# default probability and government parameters
# crop areas for maize and rice, and unused area

fig4 = plt.figure(figsize = (15, 7))

fig4.subplots_adjust(wspace=0.005)

colors = [publication_colors["green"], publication_colors["yellow"], publication_colors["grey"]]
size = 0.5

ax_tmp = fig4.add_subplot(4, 10, 1, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.62, "t = 2020 \n" + r"$\alpha$ = 90%", fontsize = 18)
    
ax_tmp = fig4.add_subplot(4, 10, 11, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.62, "t = 2030 \n" + r"$\alpha$ = 90%", fontsize = 18)

ax_tmp = fig4.add_subplot(4, 10, 21, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.62, "t = 2030 \n" + r"$\alpha$ = 99%", fontsize = 18)

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))

    # get results
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_worst99, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High")
                
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_best99, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed")            
                    
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_worst90, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High",
                                   probF = 0.9)
                
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc_best90, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed",
                                   probF = 0.9) 
    # settings, args, yield_information, population_information, penalty_methods,  \
    # status, all_durations, exp_incomes, crop_alloc_fixed, meta_sol, \
    # crop_allocF, meta_solF, crop_allocS, meta_solS, \
    # crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
    #             FS.LoadFullResults(k_using = cl,
    #                                 yield_projection = "fixed",
    #                                 pop_scenario = "fixed")    
                
    def _getAreas(year, crops):
        year_rel = year - settings["sim_start"]
        areas = [crops[year_rel,0,0], 
                crops[year_rel,1,0], 
                round(args["max_areas"][0] - np.sum(crops[year_rel,:,0]), 5)]
        return(areas)
    
    pos = letter.index(cluster_letters[cl-1]) + 1
    
    ax_tmp = fig4.add_subplot(4, 10, pos + 1)
    areas_outer = _getAreas(2020, crop_alloc_worst90)
    areas_inner = _getAreas(2020, crop_alloc_best90)
    print("                 1. outer: " + str(areas_outer) + ", inner: " + str(areas_inner))
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"),
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8),
               startangle = 180, counterclock = False)
    ax_tmp.set_title(cluster_letters[cl-1], fontsize = 18)
        
    
    ax_tmp = fig4.add_subplot(4, 10, pos + 11)
    areas_outer = _getAreas(2030, crop_alloc_worst90)
    areas_inner = _getAreas(2030, crop_alloc_best90)
    print("                 2. outer: " + str(areas_outer) + ", inner: " + str(areas_inner))
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"), 
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8), 
               startangle = 180, counterclock = False)
    
    ax_tmp = fig4.add_subplot(4, 10, pos + 21)
    areas_outer = _getAreas(2030, crop_alloc_worst99)
    areas_inner = _getAreas(2030, crop_alloc_best99)
    print("                 3. outer: " + str(areas_outer) + ", inner: " + str(areas_inner))
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"), 
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8), 
               startangle = 180, counterclock = False)
    
ax = fig4.add_subplot(4, 1, 4, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

legend_elements = [Patch(color = colors[0], alpha = 0.9, label='Rice'),
                   Patch(color = colors[1], alpha = 0.9, label='Maize'),
                   Patch(color = colors[2], alpha = 0.9, label='Not used'),
                   Patch(color = "w", label='Inner circle: best-case scenario'),
                   Patch(color = "w", label='Outer circle: worst-case scenario')]
ax.legend(handles = legend_elements, fontsize = 18,
          loc = "center", ncol = 2)


fig4.savefig("Figures/PublicationPlots/Figure4_CropAreas_new.jpg", 
                bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

# %% ####################### FIG 5 - GOVERNMENT LEVERS ########################

# Input probabiliy for food security vs. resulting probability for solvency
# Middle of the road scenario (yield trend and medium population growth)
# for different combinations of policy levers
# Example cluster in main text, other clusters in SI

alphas = [50, 60, 70, 80, 90, 95, 99]

for cl in range(5, 6):
    fig = plt.figure(figsize = (14, 9))
    
    miny = 100
    for (risk, ls) in [(0.01, "solid"), (0.05, "dashed")]:
        for (tax, mk, col) in [(0.01, "o", publication_colors["yellow"]),
                               (0.05,  "X", publication_colors["orange"]),
                               (0.1, "s", publication_colors["red"])]:
            solvency = []
            
            for alpha in alphas:
                
                tmp = FS.ReadFromPanda(output_var = ['Resulting probability for solvency'],
                              k_using = [cl],
                              risk = risk,
                              tax = tax,
                              probF = alpha/100,
                              yield_projection = "fixed",
                              pop_scenario = "fixed")
                
                solvency.append(tmp.loc[:,"Resulting probability for solvency"].values[0]*100)
                
            plt.scatter(alphas, solvency, color = col, marker = mk, s = 80,
                        label = "risk " + str(risk * 100) + "%, tax " + str(tax * 100) + "%", alpha = 0.7)
            plt.plot(alphas, solvency, color = col, linestyle = ls, lw = 3, alpha = 0.7)
            
            miny = min(miny, np.min(solvency))
            
    plt.legend()
    plt.xlabel("Reliability target for food security, %", fontsize = 20)
    plt.ylabel("Probability for solvency, %", fontsize = 20)
    plt.title("Region " + cluster_letters[cl-1], fontsize = 24, pad = 16)
    plt.ylim(0.96 * miny, 101)
    plt.xlim(48, 101)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    
    
    legend_elements1 = [Line2D([0], [0], color ='black', lw = 3, 
                          label='Covered risk: 1%'),
                    Line2D([0], [0], color ='black', lw = 3, ls = "--",
                          label='Covered risk: 5%'),
                    Line2D([0], [0], color = publication_colors["yellow"], marker = "o", linestyle = "none", 
                          label='Tax: 1%', ms = 10),
                    Line2D([0], [0], color = publication_colors["orange"], marker = "X", linestyle = "none", 
                          label='Tax: 5%', ms = 10),
                    Line2D([0], [0], color = publication_colors["red"], marker = "s", linestyle = "none", 
                          label='Tax: 10%', ms = 10)]

    ax.legend(handles = legend_elements1, fontsize = 18, bbox_to_anchor = (1, 0.5), loc = "center left")
    
    fig.savefig("Figures/PublicationPlots/Figure5_GovLevers.jpg", 
                bbox_inches = "tight", pad_inches = 0.2, format = "jpg")
    
    plt.close(fig)
    
# %% ############# FIG 6 - SCENARIO COMPARISONS WITH COOPERATION ##############

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

ylabels = ["Food supply reliability, %",
           r"Average food shortage per capita, $10^{3}\,$kcal",
           "Solvency probabilityy, %",
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


fig = FS.PlotPandaAggregate(panda_file = panda_file,
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
                      label='worst-case'),
                Line2D([0], [0], color ='black', lw = 2, ls = "dashdot",
                      label='stationary'),
                Line2D([0], [0], color ='black', lw = 2, ls = "--",
                      label='best-case'),
                Patch(color ='royalblue', alpha = 0.6, label = 'equality grouping'),
                Patch(color ='darkred', alpha = 0.6, label = 'proximity grouping')]
ax.legend(handles = legend_elements, fontsize = 24, bbox_to_anchor = (0.5, -0.06),
          loc = "upper center", ncol = 2)

fig.savefig("Figures/PublicationPlots/Figure6_Cooperation.jpg", bbox_inches = "tight", pad_inches = 0.2)