# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:51:28 2022

@author: leip
"""
# %% #########################     SETTINGS      ##############################


# set the right directory
import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  
import ModelCode.DataPreparation as DP

# import other modules
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from string import ascii_uppercase as letter

from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters

from ModelCode.PlotMaps import PlotClusterGroups

if not os.path.isdir("Figures/PublicationPlots"):
    os.mkdir("Figures/PublicationPlots")
    
    
# %% INTER- AND INTRA CLUSTER SIMILARITY CLUSTERING

with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
    pearsonDist = pickle.load(fp)  

between_all = []
between_closest = []
within_cluster = []
kmax = 19

# calculate the distances within clusters and between clusters
for k in range(2, kmax + 1):
    with open("InputData/Clusters/Clustering/kMediods" + \
                          str(k) + "_PearsonDistSPEI.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    all_dists, closest_dist = DP.MedoidMedoidDistd(medoids, pearsonDist)
    between_closest.append(np.nanmean(closest_dist))
    within_cluster.append(costs/(np.sum(~np.isnan(clusters))))
    
title = ["a. Comparison using all clusters for SPEI", \
         "Comparison using closest clusters for SPEI"]

# plot distances for different numbers of clusters
fig = plt.figure(figsize = (18, 9))
ax = fig.add_subplot(1,2,1) 
ax.scatter(within_cluster, between_closest,  c=range(2, kmax + 1))
plt.title("a. Inter- and intra-cluster similarity", fontsize = 22)
plt.xlabel("Similarity within clusters", fontsize = 16)
plt.ylabel("Similarity between clusters", fontsize = 16)
plt.xticks(fontsize =14)
plt.yticks(fontsize =14)
for t, txt in enumerate(np.arange(2, kmax + 1, 4)):
    ax.annotate(txt, (within_cluster[3*t] - 0.0024, \
                      between_closest[3*t] + 0.0029)) 
fig.add_subplot(1,2,2) 
metric, cl_order = DP.MetricClustering(within_cluster, between_closest, \
                                    refX = 0, refY = max(between_closest)) 
plt.scatter(cl_order, metric)
plt.xticks(range(2, kmax + 1), fontsize =14)
plt.yticks(fontsize =14)
plt.title("b. Quantification of tradeoff", fontsize = 22)
plt.xlabel("Number of clusters", fontsize = 16)
plt.ylabel("Euclidean distance to (0, "+ \
                              str(np.round(max(between_closest), 2)) + \
                              ") in figure a.", fontsize = 16)
# plt.suptitle("Tradeoff of distances within and between cluster")
    
fig.savefig("Figures/PublicationPlots/SI/SI_IntraAndInterClusterSimilarity.jpg", 
            bbox_inches = "tight", pad_inches = 0.2)     
    
# %% YIELD TRENDS


with open("InputData/Other/CultivationCosts.txt", "rb") as fp:
    costs = pickle.load(fp)
with open("InputData//Prices/RegionFarmGatePrices.txt", "rb") as fp:    
    prices = pickle.load(fp)

# yield thresholds for making profits
threshold = costs / prices
# 1.83497285, 1.03376958     

# plot yield trends
k = 9
with open("ProcessedData/YieldAverages_k" + str(k) + ".txt", "rb") as fp:    
    yields_avg = pickle.load(fp)
    crops = pickle.load(fp)        
    
cols = [publication_colors["green"], publication_colors["yellow"]]
start_year = 1981
len_ts = yields_avg.shape[0]

fig = plt.figure(figsize = (16, 11))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                wspace=0.15, hspace=0.35)
years = range(start_year, start_year + len_ts)
ticks = np.arange(start_year + 2, start_year + len_ts + 0.1, 8)
for cl in range(0, k):
    pos = letter.index(cluster_letters[cl-1]) + 1
    if k > 6:
        ax = fig.add_subplot(3, int(np.ceil(k/3)), pos)
    elif k > 2:
        ax = fig.add_subplot(2, int(np.ceil(k/2)), pos)
    else:
        ax = fig.add_subplot(1, k, pos)
    dict_labels = {}
    for cr in [0, 1]:
        sns.regplot(x = np.array(range(start_year, \
              start_year + len_ts)), y = yields_avg[:, cr, cl], \
              color = cols[cr], ax = ax, marker = ".", truncate = True)
    plt.plot(range(start_year, start_year + len_ts), \
             np.repeat(threshold[0], len_ts), ls = "--", 
             color = cols[0], alpha = 0.85)
    plt.plot(range(start_year, start_year + len_ts), \
             np.repeat(threshold[1], len_ts), ls = "--", 
             color = cols[1], alpha = 0.85)
    val_max = np.max(yields_avg[:,0,cl])
    ax.set_ylim(bottom = -0.05 * val_max)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(ticks)   
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_fontsize(14)
    plt.title("Region "  + cluster_letters[cl-1], fontsize = 18)
         
  
# add a big axis, hide frame, ticks and tick labels of overall axis
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Year", fontsize = 22, labelpad = 18)
plt.ylabel("Crop yield, t/ha", fontsize = 22, labelpad = 18)


legend_elements = [Line2D([0], [0], lw = 2, ls = "dashed", color= "black",
                          label="Threshold for profitability"),
                    Patch(color = publication_colors["green"], label='Rice', alpha = 0.6),
                    Patch(color = publication_colors["yellow"], label='Maize', alpha = 0.6)]
ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (0.5, -0.1),
          loc = "upper center")

fig.savefig("Figures/PublicationPlots/SI/SI_YieldTrends.jpg",
            bbox_inches = "tight", pad_inches = 0.2)   
plt.close()
    

# %% AVERAGE FARM GATE PRICES

with open("InputData/Prices/CountryAvgFarmGatePrices.txt", "rb") as fp:    
    prices = pickle.load(fp)

countries = prices["Countries"]
prices  = np.asarray(prices.iloc[:,1:3])        

col = [publication_colors["green"], publication_colors["yellow"]]
    
# visualize resulting average prices
y = np.arange(len(countries))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (16, 11))
rects1 = ax.barh(y - width/2, prices[:,0], width, label='Maize', \
                                        color = col[0], alpha = 0.5)
rects2 = ax.barh(y + width/2, prices[:,1], width, label='Rice', \
                                        color = col[1],  alpha = 0.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Average farm-gate prices, USD/t', fontsize = 20)
# ax.set_title('Average farm-gate prices per country and crop')
ax.set_yticks(y)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.set_yticklabels(countries, fontsize = 20)
ax.legend(fontsize = 24)
plt.show()
fig.savefig("Figures/PublicationPlots/SI/SI_AverageFarmGatePrices.jpg",
            bbox_inches = "tight", pad_inches = 0.2)   
plt.close()

# %% FOOD PRODUCTION DISTRIBUTINOS FOR ALL CLUSTERS

p = "fixed"
y = "fixed"

SI1 = plt.figure(figsize = (14, 8))
    
SI1.subplots_adjust(hspace = 0.45, wspace = 0.1)
for cl in range(1, 10):
    pos = letter.index(cluster_letters[cl-1]) + 1
    ax = SI1.add_subplot(3, 3, pos)
    for (alpha, col) in [(0.5, publication_colors["purple"]),
                          (0.7, publication_colors["red"]),
                          (0.9, publication_colors["orange"]), 
                          (0.95, publication_colors["yellow"]),
                         (0.99, publication_colors["green"])]:
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
    
    plt.axvline(100, color = "#003479", linestyle = "dashed", alpha = 0.6, label = "Food demand", linewidth = 2.5)
    plt.xticks(fontsize = 14)
    ax = plt.gca()
    ax.yaxis.set_ticks([])
    plt.title(r"Region " + cluster_letters[cl-1], fontsize = 18)
        
# add a big axis, hide frame
ax = SI1.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Food production as percentage of demand", fontsize = 24, labelpad = 20)
    
legend_elements = [Line2D([0], [0], color ='#003479', lw = 2, ls = "dashed",
                          label="Food demand", alpha = 0.6),
                    Patch(color = publication_colors["purple"], label=r'$\alpha$ = 50%', alpha = 0.6),
                    Patch(color = publication_colors["red"], label=r'$\alpha$ = 70%', alpha = 0.6),
                    Patch(color = publication_colors["orange"], label=r'$\alpha$ = 90%', alpha = 0.6),
                    Patch(color = publication_colors["yellow"], label=r'\alpha$ = 95%', alpha = 0.6),
                    Patch(color = publication_colors["green"], label=r'$\alpha$ = 99%', alpha = 0.6)
                    ]

ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (1, 0.5), loc = "center left")
    
SI1.savefig("Figures/PublicationPlots/SI/SI_FoodProduction.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")
        
plt.close(SI1)

# %% PIE CHARTS CROP AREAS - TWO RELIABILITY LEVELS


SI1 = plt.figure(figsize = (15, 5))

SI1.subplots_adjust(wspace=0.005)

colors = [publication_colors["green"], publication_colors["yellow"], publication_colors["grey"]]
size = 0.5

ax_tmp = SI1.add_subplot(3, 10, 1, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.85, r"$\alpha$ = 99%  ", fontsize = 18)
    
ax_tmp = SI1.add_subplot(3, 10, 11, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.85, r"$\alpha$ = 90%  ", fontsize = 18)

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))

    # get results
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_worst99, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High")
                
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_best99, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed")            
                    
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_worst90, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High",
                                   probF = 0.9)
                
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_best90, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed",
                                   probF = 0.9)   
                
                
    def _getAreas(year, crops):
        year_rel = year - settings["sim_start"]
        areas = [crops[year_rel,0,0], 
                crops[year_rel,1,0], 
                args["max_areas"][0] - np.sum(crops[year_rel,:,0])]
        return(areas)
    
    pos = letter.index(cluster_letters[cl-1]) + 1
    
    ax_tmp = SI1.add_subplot(3, 10, pos + 1)
    areas_outer = _getAreas(2030, crop_alloc_worst90)
    areas_inner = _getAreas(2030, crop_alloc_best90)
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"),
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8),
               startangle = 180, counterclock = False)
    ax_tmp.set_title(cluster_letters[cl-1], fontsize = 18)
        
    
    ax_tmp = SI1.add_subplot(3, 10, pos + 11)
    areas_outer = _getAreas(2030, crop_alloc_worst99)
    areas_inner = _getAreas(2030, crop_alloc_best99)
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"), 
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8), 
               startangle = 180, counterclock = False)
  
ax = SI1.add_subplot(3, 1, 3, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

legend_elements = [Patch(color = colors[0], alpha = 0.9, label='Rice'),
                   Patch(color = colors[1], alpha = 0.9, label='Maize'),
                   Patch(color = colors[2], alpha = 0.9, label='Not used'),
                   Patch(color = "w", label='Inner circle: best case scenario'),
                   Patch(color = "w", label='Outer circle: worst case scenario')]
ax.legend(handles = legend_elements, fontsize = 18,
          loc = "center", ncol = 2)


SI1.savefig("Figures/PublicationPlots/SI/SI_CropAreas_PieCharts_2030.jpg", 
                bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

# %% CROP AREA AS TIMESERIES FOR ALPHA = 99%


# three yield/population scenarios
# all 9 clusters as subplots to same figure
# default probability and government parameters
# crop areas for maize and rice over time


for alpha in [0.99, 0.9]:
    SI2 = plt.figure(figsize = (16, 11))
    
    SI2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.95,
                    wspace=0.15, hspace=0.35)
    
    col1 = publication_colors["green"]
    col2 = publication_colors["yellow"]
    
    print("Plotting in progress ...", flush = True)
    for cl in range(1, 10):
        print("             ... cluster " + str(cl))
        
        pos = letter.index(cluster_letters[cl-1]) + 1
        
        ax_tmp = SI2.add_subplot(3, 3, pos)
    
        # get results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc_worst, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = "fixed",
                                       pop_scenario = "High",
                                       probF = alpha)
                    
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc_best, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = "trend",
                                       pop_scenario = "fixed",
                                       probF = alpha)            
                        
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc_fixed, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = "fixed",
                                       pop_scenario = "fixed",
                                       probF = alpha)    
                    
        sim_start = settings["sim_start"]
        T = settings["T"]
        years = range(sim_start, sim_start + T)
        ticks = np.arange(sim_start, sim_start + T + 0.1, 6)
        
        # plot crop lines
        ax_tmp.plot(years, (crop_alloc_worst[:,0,0]/args["max_areas"]) * 100, color = col1, lw = 3)
        ax_tmp.plot(years, (crop_alloc_fixed[:,0,0]/args["max_areas"]) * 100, color = col1, lw = 3, ls = "dashdot")
        ax_tmp.plot(years, (crop_alloc_best[:,0,0]/args["max_areas"]) * 100, color = col1, lw = 3, ls = "--")
        
        ax_tmp.plot(years, (crop_alloc_worst[:,1,0]/args["max_areas"]) * 100, color = col2, lw = 3)
        ax_tmp.plot(years, (crop_alloc_fixed[:,1,0]/args["max_areas"]) * 100, color = col2, lw = 3, ls = "dashdot")
        ax_tmp.plot(years, (crop_alloc_best[:,1,0]/args["max_areas"]) * 100, color = col2, lw = 3, ls = "--")
        
        # shade area between worst and best case
        ax_tmp.fill_between(years, (crop_alloc_worst[:,0,0]/args["max_areas"]) * 100,
                            (crop_alloc_best[:,0,0]/args["max_areas"]) * 100,
                          color = col1, alpha = 0.5, label = "Range of rice")
        ax_tmp.fill_between(years, (crop_alloc_worst[:,1,0]/args["max_areas"]) * 100,
                            (crop_alloc_best[:,1,0]/args["max_areas"]) * 100,
                          color = col2, alpha = 0.5, label = "Range of maize")
         
        # subplot titles
        ax_tmp.set_title("Region " + cluster_letters[cl-1], fontsize = 16)
        
        # set limits and fontsizes
        ax_tmp.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax_tmp.set_ylim(-5, 105)
        ax_tmp.set_xticks(ticks)   
        ax_tmp.xaxis.set_tick_params(labelsize=16)
        ax_tmp.yaxis.set_tick_params(labelsize=16)
        ax_tmp.yaxis.offsetText.set_fontsize(16)
        ax_tmp.xaxis.offsetText.set_fontsize(16)
            
        
    # add a big axis, hide frame, ticks and tick labels of overall axis
    ax = SI2.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Year", fontsize = 24, labelpad = 20)
    plt.ylabel("Crop area as percentage of available arable area", fontsize = 24, labelpad = 20)
    
    legend_elements = [Line2D([0], [0], color ='black', lw = 2, 
                              label='worst case'),
                        Line2D([0], [0], color ='black', lw = 2, ls = "dashdot",
                              label='stationary'),
                        Line2D([0], [0], color ='black', lw = 2, ls = "--",
                              label='best case'),
                        Patch(color = col1, label='Rice'),
                        Patch(color = col2,  label='Maize')]
    ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (0.5, -0.15),
              loc = "upper center", ncol = 2)
    
    SI2.savefig("Figures/PublicationPlots/SI/SI_CropAreas_Timeseries_alpha" + str(int(alpha*100)) + "perc.jpg", 
                    bbox_inches = "tight", pad_inches = 1, format = "jpg")
    plt.close(SI2)
    
# %% FINAL FUND DISTRIBUTIONS

# distribution of final fund after payout for samples with catastrophe
# for the three population/yield scenarios
# default input probability and government parameters
# a single figure with each cluster as a subplot, and legend in separate plot

if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")

fig = plt.figure(figsize = (16, 11))

fig.subplots_adjust(wspace=0.15, hspace=0.35)

for cl in range(1, 10):
    pos = letter.index(cluster_letters[cl-1]) + 1
    ax = fig.add_subplot(3, 3, pos)
    for (y, p, scen, col) in [("fixed", "High", "worst case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best case", publication_colors["green"])]:
        
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p)
                    
        year_rel = 2030 - settings["sim_start"]
        with_catastrophe = (args["terminal_years"] != -1)
        plt.hist(meta_sol["final_fund"][with_catastrophe], bins = 200, alpha = 0.7,
                 density = True, color = col)
    ax.yaxis.set_ticks([])
    ax.set_title("Region " + cluster_letters[cl-1], fontsize = 18)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.xaxis.offsetText.set_fontsize(16)
     
# add a big axis, hide frame, ticks and tick labels from overall axis
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Final fund after payouts, $10^{9}\,\$$", fontsize = 24, labelpad = 20)
   
legend_elements = [Patch(color = publication_colors["red"], label='worst case', alpha = 0.7),
                    Patch(color = publication_colors["yellow"], label= "stationary", alpha = 0.7),
                    Patch(color = publication_colors["green"], label='best case', alpha = 0.7)]

ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (0.5, -0.12),
          loc = "upper center")

fig.savefig("Figures/PublicationPlots/SI/SI_FinalFund.jpg", 
            bbox_inches = "tight", format = "jpg")
        
plt.close(fig)

# %% GOVERNMENT LEVERS FOR ALL CLUSTERS

alphas = [50, 60, 70, 80, 90, 95, 99]

fig = plt.figure(figsize = (14, 9))

fig.subplots_adjust(hspace = 0.39)
for cl in range(1, 10):
    pos = letter.index(cluster_letters[cl-1]) + 1
    ax = fig.add_subplot(3, 3, pos)
    
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
                              yield_projection = "trend",
                              pop_scenario = "Medium")
                
                solvency.append(tmp.loc[:,"Resulting probability for solvency"].values[0]*100)
                
            plt.scatter(alphas, solvency, color = col, marker = mk, s= 40)
            plt.plot(alphas, solvency, color = col, linestyle = ls, lw = 2)
            
    plt.title("Region " + cluster_letters[cl-1], fontsize = 18)
    plt.ylim(-4, 101)
    plt.xlim(48, 101)
    ax = plt.gca()
    ax.set_xticks([50,60,70,80,90,100])   
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_fontsize(14)
    

# add a big axis, hide frame, ticks and tick labels of overall axis
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Target probability for food security, %", fontsize = 24, labelpad = 20)
plt.ylabel("Probability for solvency, %", fontsize = 24, labelpad = 20)
   
legend_elements = [Line2D([0], [0], color = publication_colors["yellow"], marker = "o", linestyle = "none", 
                      label='Tax: 1%', ms = 10),
                Line2D([0], [0], color = publication_colors["orange"], marker = "X", linestyle = "none", 
                      label='Tax: 5%', ms = 10),
                Line2D([0], [0], color = publication_colors["red"], marker = "s", linestyle = "none", 
                      label='Tax: 10%', ms = 10),
                Line2D([0], [0], color ='black', lw = 2, 
                      label='Covered risk: 1%'),
                Line2D([0], [0], color ='black', lw = 2, ls = "--",
                      label='Covered risk: 5%')]
ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (0.5, -0.14),
          loc = "upper center", ncol = 2)

fig.savefig("Figures/PublicationPlots/SI/SI_GovernmentLevers.jpg", 
            bbox_inches = "tight", format = "jpg")

plt.close(fig)    
    