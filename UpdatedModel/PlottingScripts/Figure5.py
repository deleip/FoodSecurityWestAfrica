# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:38:39 2021

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
from matplotlib.lines import Line2D
import numpy as np
from string import ascii_uppercase as letter

from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters
    
if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")

    
# %% ########################### GOVERNMENT LEVERS ###########################

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
                              yield_projection = "trend",
                              pop_scenario = "Medium")
                
                solvency.append(tmp.loc[:,"Resulting probability for solvency"].values[0]*100)
                
            plt.scatter(alphas, solvency, color = col, marker = mk, s = 80,
                        label = "risk " + str(risk * 100) + "%, tax " + str(tax * 100) + "%", alpha = 0.7)
            plt.plot(alphas, solvency, color = col, linestyle = ls, lw = 3, alpha = 0.7)
            
            miny = min(miny, np.min(solvency))
            
    plt.legend()
    plt.xlabel("Target probability for food security, %", fontsize = 20)
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
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
    
    plt.close(fig)


# %% for SI

# same as above but all 9 cluster as subplots of the main figure, with legend
# in separate plot

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
    
  