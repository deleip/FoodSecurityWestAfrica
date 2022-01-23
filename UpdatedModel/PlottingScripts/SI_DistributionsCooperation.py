# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:03:06 2021

@author: leip
"""
# set the right directory
import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(dir_path)

# import other modules
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import pickle
import matplotlib.gridspec as gs

# import all project related functions
import FoodSecurityModule as FS  
from ModelCode.GeneralSettings import figsize

if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")

# %% ################# DISTRIBUTION OF FOOD SUPPLY AND INCOME #################
# without trends!
# including cooperation!

aim = "Similar"
adj = "Adj"
metric = "medoids"


# for similar adjacent (medoids)
mapping_sim_adj = \
          [pd.DataFrame({"group" : [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
                         "colfirst" : [5, 0, 7, 4, 2, 8, 3, 1, 6],
                         "collast" : [6, 1, 8, 5, 3, 9, 4, 2, 7],
                         "row": [0, 0, 0, 0, 0, 0, 0, 0, 0]}),
           pd.DataFrame({"group": [(4,), (1, 9), (2, 8), (3, 6), (5, 7)],
                         "colfirst" : [4, 5, 0, 7, 2],
                         "collast" : [5, 7, 2, 9, 4],
                         "row": [1, 1, 1, 1, 1, ]}),
           pd.DataFrame({"group" : [(1, 4, 7), (2, 5, 8), (3, 6, 9)],
                         "colfirst" : [3, 0, 6],
                         "collast" : [6, 3, 9],
                         "row": [2, 2, 2]}),
           pd.DataFrame({"group" : [(2, 5, 7, 8), (1, 3, 4, 6, 9)],
                         "colfirst" : [0, 4],
                         "collast" : [4, 9],
                         "row": [3, 3]}),
           pd.DataFrame({"group": [(1, 2, 3, 4, 5, 6, 7, 8, 9)],
                         "colfirst" : [0],
                         "collast" : [9],
                         "row": [4]})]


# set up figure structure
fig = []
ax = []
box = []
inner = []

for i in range(0, 4):
    fig.append(plt.figure(figsize = figsize))
    ax.append(fig[i].add_subplot(1,1,1))
    box.append(ax[i].get_position())
    ax[i].set_position([box[i].x0, box[i].y0, box[i].width * 0.8, box[i].height])
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    inner.append(gs.GridSpecFromSubplotSpec(5, 9, ax[i], wspace = 0.2, hspace = 0.3))

# plot for all cluster groups of all sizes
for idx, size in enumerate([1, 2, 3, 5, 9]):
    # get grouping for that size
    with open("InputData/Clusters/ClusterGroups/Grouping" + metric.capitalize() + \
              "Size" + str(size) + aim + adj + ".txt", "rb") as fp:
            BestGrouping = pickle.load(fp)
    tmp = mapping_sim_adj[idx]
    for cl in BestGrouping:
        # position of subplot
        rows = tmp["row"][tmp.loc[:, "group"] == cl].values[0]
        colfirst = tmp["colfirst"][tmp.loc[:, "group"] == cl].values[0]
        collast = tmp["collast"][tmp.loc[:, "group"] == cl].values[0]
        
        # get results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = [cl])
                    
        # increase in cultivation costs for sto. solution
        cultivation_costs_det = np.sum(crop_alloc_vss * args["costs"])
        cultivation_costs_sto = np.sum(crop_alloc * args["costs"])      
        rel_increase = (cultivation_costs_sto - cultivation_costs_det)/cultivation_costs_det
        
        # plot distribution of food supply
        ax0_tmp = plt.Subplot(fig[0], inner[0][rows, colfirst:collast])
        ax0_tmp.axvline(args["demand"][0], color = "blue", linestyle = "dashed", alpha = 0.6)
        ax0_tmp.hist(meta_sol_vss["food_supply"].flatten(), bins = 100, alpha = 0.6)   
        ax0_tmp.hist(meta_sol["food_supply"].flatten(), bins = 100, alpha = 0.6) 
        if size == 1:
            ax0_tmp.set_title("Region " + str(cl[0]))
        ax0_tmp.set_yticks([])
        ax0_tmp.text(x = 0.01, y = 0.98, 
                     s = str(np.round(rel_increase * 100, 2)) + "%",
                     ha = "left", va = "top",
                     transform = ax0_tmp.transAxes)
        fig[0].add_subplot(ax0_tmp)
    
        # plot distribution of profits/losses
        ax1_tmp = plt.Subplot(fig[1], inner[1][rows, colfirst:collast])
        ax1_tmp.axvline(np.sum(meta_sol["guaranteed_income"], axis = 1)[0], color = "blue", linestyle = "dashed", alpha = 0.6)
        ax1_tmp.axvline(np.sum(meta_sol_vss["guaranteed_income"], axis = 1)[0], color = "green", linestyle = "dashed", alpha = 0.6)
        ax1_tmp.hist(np.sum(meta_sol_vss["profits_afterTax"], axis = 2).flatten(), bins = 100, alpha = 0.6)
        ax1_tmp.hist(np.sum(meta_sol["profits_afterTax"], axis = 2).flatten(), bins = 100, alpha = 0.6) 
        if size == 1:
            ax1_tmp.set_title("Cluster " + str(cl[0]))
        ax1_tmp.set_yticks([])
        ax1_tmp.text(x = 0.01, y = 0.98, 
                     s = str(np.round(rel_increase * 100, 2)) + "%",
                     ha = "left", va = "top",
                     transform = ax1_tmp.transAxes)
        fig[1].add_subplot(ax1_tmp)
        
        # plot distribution of profits including government payouts
        guaranteed = (1 - args["tax"]) * args["cat_clusters"] * meta_sol["guaranteed_income"] # final profit including payouts will be guaranteed income minus taxes
        guaranteed[guaranteed == 0] = -np.inf
        guaranteed_vss = (1 - args["tax"]) * args["cat_clusters"] * meta_sol_vss["guaranteed_income"]
        guaranteed_vss[guaranteed_vss == 0] = -np.inf
        income = np.maximum(meta_sol["profits_afterTax"], guaranteed)
        income_vss = np.maximum(meta_sol_vss["profits_afterTax"], guaranteed_vss)
        income_F = np.maximum(meta_solF["profits_afterTax"], guaranteed)
        ax2_tmp = plt.Subplot(fig[2], inner[2][rows, colfirst:collast])
        ax2_tmp.axvline(np.sum(meta_sol["guaranteed_income"], axis = 1)[0], color = "red", linestyle = "dashed", alpha = 0.6)
        ax2_tmp.axvline(np.sum(meta_sol_vss["guaranteed_income"], axis = 1)[0], color = "blue", linestyle = "dashed", alpha = 0.6)
        ax2_tmp.hist(np.sum(income_vss, axis = 2).flatten(), bins = 100, alpha = 0.6)   
        ax2_tmp.hist(np.sum(income, axis = 2).flatten(), bins = 100, alpha = 0.6) 
        if size == 1:
            ax2_tmp.set_title("Cluster " + str(cl[0]))
        ax2_tmp.set_yticks([])
        ax2_tmp.text(x = 0.01, y = 0.98, 
                     s = str(np.round(rel_increase * 100, 2)) + "%",
                     ha = "left", va = "top",
                     transform = ax2_tmp.transAxes)
        fig[2].add_subplot(ax2_tmp)
        
        # final fund (after payouts) - only for cases with catastrophe
        with_catastrophe = (args["terminal_years"] != -1)
        ax3_tmp = plt.Subplot(fig[3], inner[3][rows, colfirst:collast])
        ax3_tmp.axvline(0, color = "blue", linestyle = "dashed", alpha = 0.6)
        ax3_tmp.hist(meta_sol_vss["final_fund"][with_catastrophe].flatten(), bins = 100, alpha = 0.6)  
        ax3_tmp.hist(meta_sol["final_fund"][with_catastrophe].flatten(), bins = 100, alpha = 0.6)  
        if size == 1:
            ax3_tmp.set_title("Cluster " + str(cl[0]))
        ax3_tmp.set_yticks([])
        ax3_tmp.text(x = 0.01, y = 0.98, 
                     s = str(np.round(rel_increase * 100, 2)) + "%",
                     ha = "left", va = "top",
                     transform = ax3_tmp.transAxes)
        fig[3].add_subplot(ax3_tmp)

    

# add labels  
titles = ["Distribution of food production",
          "Distribution of profits (after tax payments) w/o payouts",
          "Distribution of profits (after tax payments) including payouts",
          "Distribution of final fund (after payouts) - only cases w/ catastrophe"]
xlabels = [r"Food supply in cluster [$10^{12}\,kcal$]",
           r"Aggregated profits of farmers in cluster [$10^9\,\$$]",
           r"Aggregated income of farmers (including payouts) in cluster [$10^9\,\$$]",
           r"Final fund size after payouts [$10^9\,\$$]"]
legend_labels = ["Food demand", 
                 "Guaranteed income (sto.)",
                 "Guaranteed income (det.)",
                 "Zero"]
filenames = ["FoodSupplyDistribution_NoTrends_WithCoop",
             "ProfitDistribution_NoTrends_WithCoop",
             "IncomeDistribution_NoTrends_WithCoop",
             "FinalFundDistribution_NoTrends_WithCoop"]

for i in range(0, 4):
    ax[i].set_title(titles[i], fontsize = 30, pad = 35)
    ax[i].set_xlabel(xlabels[i], fontsize = 14, labelpad = 25)
    
    legend_elements =  [Patch(facecolor='#1f77b4', 
                             label='Deterministic solution', alpha = 0.6),
                       Patch(facecolor='#ff7f0e',
                             label='Robust solution', alpha = 0.6),
                       Line2D([0], [0], color = 'blue', lw = 1.5, ls = "dashed",
                              label = legend_labels[i], alpha = 0.6)]
    if i in [1,2]:
        legend_elements.append( Line2D([0], [0], color = 'red', lw = 1.5, 
              ls = "dashed", label = "Guaranteed income (sto.)", alpha = 0.6))
        
    ax[i].legend(handles = legend_elements, bbox_to_anchor = (1.02, 0.5),
                 loc = 'center left', fontsize = 14)
    fig[i].savefig("Figures/PublicationPlots/SI/" + filenames[i] + ".jpg",
                   bbox_inches = "tight", pad_inches = 1, format = "jpg")
    plt.close(fig[i])
    
    
    