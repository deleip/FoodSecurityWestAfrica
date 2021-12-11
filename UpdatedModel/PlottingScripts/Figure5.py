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
from ModelCode.GeneralSettings import figsize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


if not os.path.isdir("Figures/PublicationPlots/Figure5"):
    os.mkdir("Figures/PublicationPlots/Figure5")
    
if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")

    
# %% ########################### GOVERNMENT LEVERS ###########################

# Input probabiliy for food security vs. resulting probability for solvency
# Middle of the road scenario (yield trend and medium population growth)
# for different combinations of policy levers
# Example cluster in main text, other clusters in SI

alphas = [50, 60, 70, 80, 90, 95, 99]

for cl in range(1, 10):
    fig = plt.figure(figsize = (14, 9))
    
    for (risk, ls) in [(0.01, "solid"), (0.05, "dashed")]:
        for (tax, mk, col) in [(0.01, "o", "#6F6058"),
                               (0.05,  "X", "#BB7369"),
                               (0.1, "s", "#E4C496")]:
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
                        label = "risk " + str(risk * 100) + "%, tax " + str(tax * 100) + "%")
            plt.plot(alphas, solvency, color = col, linestyle = ls, lw = 3)
            
    plt.legend()
    plt.xlabel("Input probability for food security, %", fontsize = 20)
    plt.ylabel("Output probability for solvency, %", fontsize = 20)
    plt.title("Region " + str(cl), fontsize = 24)
    plt.ylim(-4, 101)
    plt.xlim(48, 101)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    
    
    legend_elements1 = [Line2D([0], [0], color ='black', lw = 3, 
                          label='Covered risk: 1%'),
                    Line2D([0], [0], color ='black', lw = 3, ls = "--",
                          label='Covered risk: 5%'),
                    Line2D([0], [0], color = "#6F6058", marker = "o", linestyle = "none", 
                          label='Tax: 1%', ms = 10),
                    Line2D([0], [0], color = "#BB7369", marker = "X", linestyle = "none", 
                          label='Tax: 5%', ms = 10),
                    Line2D([0], [0], color = "#E4C496", marker = "s", linestyle = "none", 
                          label='Tax: 10%', ms = 10)]

    ax.legend(handles = legend_elements1, fontsize = 18)
    
    fig.savefig("Figures/PublicationPlots/Figure5/cl" + str(cl) + ".jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
    
    plt.close(fig)


# %% for SI

# same as above but all 9 cluster as subplots of the main figure, with legend
# in separate plot

alphas = [50, 60, 70, 80, 90, 95, 99]

fig = plt.figure(figsize = (14, 9))

fig.subplots_adjust(hspace = 0.39)
for cl in range(1, 10):
    ax = fig.add_subplot(3, 3, cl)
    
    for (risk, ls) in [(0.01, "solid"), (0.05, "dashed")]:
        for (tax, mk, col) in [(0.01, "o", "#6F6058"),
                               (0.05,  "X", "#BB7369"),
                               (0.1, "s", "#E4C496")]:
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
            
    plt.title("Region " + str(cl), fontsize = 18)
    plt.ylim(-4, 101)
    plt.xlim(48, 101)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    

# add a big axis, hide frame, ticks and tick labels of overall axis
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Input probability for food security, %", fontsize = 24, labelpad = 20)
plt.ylabel("Output probability for solvency, %", fontsize = 24, labelpad = 20)
   
fig.savefig("Figures/PublicationPlots/SI/GovernmentLevers.jpg", 
            bbox_inches = "tight", pad_inches = 1, format = "jpg")

plt.close(fig)    
    
  
# LEGEND IN SEPARATE PLOT
legend = plt.figure(figsize  = (5, 3))
legend_elements1 = [Line2D([0], [0], color ='black', lw = 2, 
                      label='Covered risk: 1%'),
                Line2D([0], [0], color ='black', lw = 2, ls = "--",
                      label='Covered risk: 5%'),
                Line2D([0], [0], color = "#6F6058", marker = "o", linestyle = "none", 
                      label='Tax: 1%', ms = 10),
                Line2D([0], [0], color = "#BB7369", marker = "X", linestyle = "none", 
                      label='Tax: 5%', ms = 10),
                Line2D([0], [0], color = "#E4C496", marker = "s", linestyle = "none", 
                      label='Tax: 10%', ms = 10)]

ax = legend.add_subplot(1, 1, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(handles = legend_elements1, fontsize = 14, loc = 6)

legend.savefig("Figures/PublicationPlots/SI/GovernmentLeversLegend.jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
plt.close(legend)
    
    