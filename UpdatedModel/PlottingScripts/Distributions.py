# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 18:58:24 2021

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
from matplotlib.patches import Patch

# %% PROFIT DISTRIBUTION - separate plots

# distribution of farmers profits (after paying taxes)
# for the three population/yield scenarios
# default input probability and government parameters
# each cluster as separate plot

if not os.path.isdir("Figures/PublicationPlots/ProfitDistribution"):
    os.mkdir("Figures/PublicationPlots/ProfitDistribution")

for cl in range(1, 10):
    fig = plt.figure(figsize = (8, 5))
    
    for (y, p) in [("fixed", "High"),
                   ("fixed", "fixed"),
                   ("trend", "fixed")]:
        
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p)
                    
        year_rel = 2030 - settings["sim_start"]
        plt.hist(meta_sol["profits_afterTax"].flatten(), bins = 200, alpha = 0.5,
                 density = True, label = "Yield " + y + ", population " + p)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel(r"Profits (after paying taxes) [$10^{9}\,\$$]", fontsize = 14)
        
    fig.savefig("Figures/PublicationPlots/ProfitDistribution/Cl" + str(cl) + ".jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
            
    plt.close(fig)


# %% FINAL FUND DISTRIBUTION - separate plots

# distribution of final fund after payout for samples with catastrophe
# for the three population/yield scenarios
# default input probability and government parameters
# each cluster as separate plot

if not os.path.isdir("Figures/PublicationPlots/FinalFundDistribution"):
    os.mkdir("Figures/PublicationPlots/FinalFundDistribution")

for cl in range(1, 10):
    fig = plt.figure(figsize = (8, 5))
    
    for (y, p, scen, col) in [("fixed", "High", "worst case", "#C53B21"), 
                    ("fixed", "fixed", "stationary", "#C9AF8C"),
                    ("trend", "fixed", "best case", "#4B9A8D")]:
        
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
                 density = True, label = scen, color = col)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel(r"Final fun after payouts [$10^{9}\,\$$]", fontsize = 14)
        
    fig.savefig("Figures/PublicationPlots/FinalFundDistribution/Cl" + str(cl) + ".jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
            
    plt.close(fig)

# %% FINAL FUND DISTRIBUTION - same plot

# distribution of final fund after payout for samples with catastrophe
# for the three population/yield scenarios
# default input probability and government parameters
# a single figure with each cluster as a subplot, and legend in separate plot

if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")

fig = plt.figure(figsize = (14, 8))

fig.subplots_adjust(hspace = 0.39)
for cl in range(1, 10):
    ax = fig.add_subplot(3, 3, cl)
    for (y, p, scen, col) in [("fixed", "High", "worst case", "#C53B21"), 
                    ("fixed", "fixed", "stationary", "#C9AF8C"),
                    ("trend", "fixed", "best case", "#4B9A8D")]:
        
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
    plt.xticks(fontsize = 12)
    ax.yaxis.set_ticks([])
    plt.title(r"Region " + str(cl), fontsize = 18)
     
# add a big axis, hide frame, ticks and tick labels from overall axis
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Final fund after payouts, $10^{9}\,\$$", fontsize = 24, labelpad = 20)
   
fig.savefig("Figures/PublicationPlots/SI/FinalFund.jpg", 
            bbox_inches = "tight", pad_inches = 1, format = "jpg")
        
plt.close(fig)

# LEGEND AS SEPARATE PLOT
legend = plt.figure(figsize  = (5, 3))
legend_elements1 = [Patch(color = "#C53B21", label='worst case', alpha = 0.7),
                    Patch(color = "#C9AF8C", label= "stationary", alpha = 0.7),
                    Patch(color = "#4B9A8D", label='best case', alpha = 0.7)]

ax = legend.add_subplot(1, 1, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(handles = legend_elements1, fontsize = 14, loc = 6)

legend.savefig("Figures/PublicationPlots/SI/FinalFundLegend.jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
plt.close(legend)

# %% CALORIE PRODUCTION DISTRIBUTION - separate plots

# distribution of calorie production
# for the three population/yield scenarios
# default input probability and government parameters
# each cluster as separate plot


if not os.path.isdir("Figures/PublicationPlots/CalorieProductionDistribution"):
    os.mkdir("Figures/PublicationPlots/CalorieProductionDistribution")

for cl in range(1, 10):
    fig = plt.figure(figsize = (8, 5))
    
    for (y, p) in [("fixed", "High"),
                   ("fixed", "fixed"),
                   ("trend", "fixed")]:
        
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p)
                    
        year_rel = 2030 - settings["sim_start"]
        plt.hist(meta_sol["food_supply"].flatten(), bins = 200, alpha = 0.5,
                 density = True, label = "Yield " + y + ", population " + p)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel(r"Food supply (showing all years) [$10^{9}\,kcal$]", fontsize = 14)
        
    fig.savefig("Figures/PublicationPlots/CalorieProductionDistribution/Cl" + str(cl) + ".jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
            
    plt.close(fig)
    
# %% CALORIE PRODUCTION DISTRIBUTION - separate plots

# distribution of revenues (profits pre tax + costs + payouts)
# for the three population/yield scenarios
# default input probability and government parameters
# each cluster as separate plot

if not os.path.isdir("Figures/PublicationPlots/RevenueDistribution"):
    os.mkdir("Figures/PublicationPlots/RevenueDistribution")

for cl in range(1, 10):
    fig = plt.figure(figsize = (8, 5))
    
    for (y, p) in [("fixed", "High"),
                   ("fixed", "fixed"),
                   ("trend", "fixed")]:
        
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p)
                    
        year_rel = 2030 - settings["sim_start"]
        
        revenue = meta_sol["profits_preTax"] + meta_sol["yearly_fixed_costs"] + meta_sol["payouts"]
        
        plt.hist(revenue.flatten(), bins = 200, alpha = 0.5,
                 density = True, label = "Yield " + y + ", population " + p)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel(r"Revenue (profits pre tax + costs + payouts) [$10^{9}\,\$$]", fontsize = 14)
        
    fig.savefig("Figures/PublicationPlots/RevenueDistribution/Cl" + str(cl) + ".jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
            
    plt.close(fig)