# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:04:51 2022

@author: leip
"""



# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + "/..")

import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# import all project related functions
import FoodSecurityModule as FS  

from string import ascii_uppercase as letter
from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters
    
if not os.path.isdir("Figures/PublicationPlots/AdditionalInfo"):
    os.mkdir("Figures/PublicationPlots/AdditionalInfo")

# %% WHAT REGIONS CAN REACH THE FOOD SECURITY TARGET?

for (y, p, scen_name) in [("fixed", "High", "WorstCase"), 
                ("fixed", "fixed", "Stationary"),
                ("trend", "fixed", "Best Case")]:
    maxProbabilities = FS.Panda_GetResultsSingScen(file = "current_panda",
                               output_var = "Max. possible probability for food security (excluding solvency constraint)",
                               out_type = "all", 
                               sizes = 1).iloc[0,1]
    
    maxProbabilities = dict(zip(cluster_letters, [round(p, 2) for p in maxProbabilities]))
    maxProbabilities = {key : value for key, value in sorted(maxProbabilities.items())}
    
    print(scen_name + ":")
    print(maxProbabilities, flush = True)
    
    json.dump(maxProbabilities, open("Figures/PublicationPlots/AdditionalInfo/MaxProbabilities" + scen_name + ".json", "w"))


# %% HOW DOES EXPECTED FOOD SHORTAGE DECREASE WITH FOOD SECURITY TARGET?

# for each scenario
# over all clusters (whout coopreatoin) averaged using population as weight
# for different food security probabilites
# default government levers


fig = plt.figure(figsize = (14, 9))

alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best case", publication_colors["green"])
                ]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    
    shortages = []
    
    for alpha in alphas: 
        print("Food security probability " + str(alpha) + "%", flush = True)
        
        shortages.append(FS.Panda_GetResultsSingScen(file = "current_panda", 
                                   output_var = "Average aggregate food shortage per capita (including only samples that have shortage)",
                                   out_type = "agg_avgweight", 
                                   var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                   sizes = 1,
                                   probF = alpha/100,
                                   yield_projection = y,
                                   pop_scenario = p).iloc[0,1])
        
    plt.scatter(alphas, shortages, label = scen, color = col, s = 80)
    plt.plot(alphas, shortages, color = col, lw = 2.5)

plt.xlabel("Reliability target for food security, %", fontsize = 24)
plt.ylabel(r"Average expected food shortage per capita, $10^{12}\,$kcal", fontsize = 24)
plt.legend(fontsize = 22, loc = "upper right")

ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
    
fig.savefig("Figures/PublicationPlots/AdditionalInfo/AdditionalPlot1_AvgFoodShortagePC.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

plt.close(fig)

# %% HOW DOES EXPECTED FOOD SHORTAGE DECREASE WITH FOOD SECURITY TARGET?

# for each scenario
# over all clusters (whout coopreatoin) averaged using population as weight
# for different food security probabilites
# default government levers


fig = plt.figure(figsize = (14, 9))

alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best case", publication_colors["green"])]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    
    shortages = []
    
    for alpha in alphas: 
        print("Food security probability " + str(alpha) + "%", flush = True)
        
        shortages.append(FS.Panda_GetResultsSingScen(file = "current_panda", 
                                   output_var = "Average aggregate food shortage per capita (including only samples that have shortage)",
                                   out_type = "agg_avgweight", 
                                   var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                   sizes = 1,
                                   probF = alpha/100,
                                   yield_projection = y,
                                   pop_scenario = p).iloc[0,1])
        
    shortages = (shortages / shortages[0]) * 100
    plt.scatter(alphas, shortages, label = scen, color = col, s = 80)
    plt.plot(alphas, shortages, color = col, lw = 2.5)

plt.xlabel("Reliability target for food security, %", fontsize = 24)
plt.ylabel("Average expected food shortage per capita as share of \n expected shortage in risk-neutral strategy, %", fontsize = 24)
plt.legend(fontsize = 22, loc = "upper right")

ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
    
fig.savefig("Figures/PublicationPlots/AdditionalInfo/AdditionalPlot1_AvgFoodShortagePC_RelativeToRiskNeutralStrategy.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

plt.close(fig)
    
# %% HOW DOES EXPECTED FOOD SHORTAGE DECREASE WITH FOOD SECURITY TARGET?

# RESULTING FOOD SECURITY PROBABILITES FOR PERSPECTIVE

# for each scenario
# over all clusters (whout coopreatoin) averaged using population as weight
# for different food security probabilites
# default government levers


fig = plt.figure(figsize = (14, 9))

alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best case", publication_colors["green"])]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    
    probabilities = []
    
    for alpha in alphas: 
        print("Food security probability " + str(alpha) + "%", flush = True)
        
        probabilities.append(FS.Panda_GetResultsSingScen(file = "current_panda", 
                                   output_var = "Resulting probability for food security",
                                   out_type = "agg_avgweight", 
                                   var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                   sizes = 1,
                                   probF = alpha/100,
                                   yield_projection = y,
                                   pop_scenario = p).iloc[0,1])
        
    plt.scatter(alphas, [p * 100 for p in probabilities], label = scen, color = col, s = 80)
    plt.plot(alphas, [p * 100 for p in probabilities], color = col, lw = 2.5)

plt.xlabel("Reliability target for food security, %", fontsize = 24)
plt.ylabel(r"Average probability for food security, %", fontsize = 24)
plt.legend(fontsize = 22, loc = "upper left")

ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
    
fig.savefig("Figures/PublicationPlots/AdditionalInfo/AdditionalPlot1_AvgFoodSecuriyProb.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

plt.close(fig)


# %% HOW DOES EXPECTED FOOD SHORTAGE DECREASE WITH FOOD SECURITY TARGET?

# for each scenario
# over all clusters (whout coopreatoin) averaged using population as weight
# for different food security probabilites
# default government levers


fig = plt.figure(figsize = (14, 9))

alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best case", publication_colors["green"])]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    
    shortages = []
    
    for alpha in alphas: 
        print("Food security probability " + str(alpha) + "%", flush = True)
        
        shortages.append(FS.Panda_GetResultsSingScen(file = "current_panda", 
                                   output_var = "Average aggregate food shortage",
                                   out_type = "agg_sum", 
                                   sizes = 1,
                                   probF = alpha/100,
                                   yield_projection = y,
                                   pop_scenario = p).iloc[0,1])
        
    plt.scatter(alphas, shortages, label = scen, color = col, s = 80)
    plt.plot(alphas, shortages, color = col, lw = 2.5)

plt.xlabel("Reliability target for food security, %", fontsize = 24)
plt.ylabel(r"Aggregated expected food shortage, $10^{12}\,$kcal", fontsize = 24)
plt.legend(fontsize = 22, loc = "upper right")

ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
    
fig.savefig("Figures/PublicationPlots/AdditionalInfo/AdditionalPlot1_AggFoodShortage.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

plt.close(fig)

 # %% HOW DOES EXPECTED FOOD SHORTAGE DECREASE WITH FOOD SECURITY TARGET?

# for each scenario
# over all clusters (whout coopreatoin) averaged using population as weight
# for different food security probabilites
# default government levers


fig = plt.figure(figsize = (14, 9))

alphas = [50, 60, 70, 80, 90, 95, 99, 99.5]

for (y, p, scen, col) in [("fixed", "High", "worst case", publication_colors["red"]), 
                ("fixed", "fixed", "stationary", publication_colors["yellow"]),
                ("trend", "fixed", "best case", publication_colors["green"])]:
    print("\u2017"*65, flush = True)
    print("Scenario: yield " + y + ", population " + p, flush = True)
    print("\u033F "*65, flush = True)
    
    shortages = []
    
    for alpha in alphas: 
        print("Food security probability " + str(alpha) + "%", flush = True)
        
        shortages.append(FS.Panda_GetResultsSingScen(file = "current_panda", 
                                   output_var = "Average aggregate food shortage",
                                   out_type = "agg_sum", 
                                   sizes = 1,
                                   probF = alpha/100,
                                   yield_projection = y,
                                   pop_scenario = p).iloc[0,1])
        
    shortages = (shortages / np.float64(shortages[0])) * 100
    plt.scatter(alphas, shortages, label = scen, color = col, s = 80)
    plt.plot(alphas, shortages, color = col, lw = 2.5)

plt.xlabel("Reliability target for food security, %", fontsize = 24)
plt.ylabel("Aggregated expected food shortage as share of\nexpected shortage in risk-neutral strategy, %", fontsize = 24)
plt.legend(fontsize = 22, loc = "upper right")

ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
    
fig.savefig("Figures/PublicationPlots/AdditionalInfo/AdditionalPlot1_AggFoodShortage_RelativeToRiskNeutralStrategy.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")

plt.close(fig)
    