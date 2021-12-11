# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:59:06 2021

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

if not os.path.isdir("Figures/PublicationPlots/Figure4"):
    os.mkdir("Figures/PublicationPlots/Figure4")


# %%  CROP AREAS 

# three yield/population scenarios
# all 9 clusters as subplots to same figure
# default probability and government parameters
# crop areas for maize and rice over time

panelA = plt.figure(figsize = (16, 11))

panelA.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.95,
                wspace=0.15, hspace=0.35)

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))
    ax_tmp = panelA.add_subplot(3, 3, cl)

    # get results
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_worst, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High")
                
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_best, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed")            
                    
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc_fixed, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "fixed")    
                
    sim_start = settings["sim_start"]
    T = settings["T"]
    years = range(sim_start, sim_start + T)
    ticks = np.arange(sim_start, sim_start + T + 0.1, 6)
        
    # ax_tmp.plot([years[0] - 0.5, years[-1] + 0.5], np.repeat(args["max_areas"], 2),
    #         color = "dimgrey", lw = 5, alpha = 0.4)
    
    # plot crop lines
    ax_tmp.plot(years, (crop_alloc_worst[:,0,0]/args["max_areas"]) * 100, color = "#656C12", lw = 3)
    ax_tmp.plot(years, (crop_alloc_fixed[:,0,0]/args["max_areas"]) * 100, color = "#656C12", lw = 3, ls = "dashdot")
    ax_tmp.plot(years, (crop_alloc_best[:,0,0]/args["max_areas"]) * 100, color = "#656C12", lw = 3, ls = "--")
    
    ax_tmp.plot(years, (crop_alloc_worst[:,1,0]/args["max_areas"]) * 100, color = "#ECC216", lw = 3)
    ax_tmp.plot(years, (crop_alloc_fixed[:,1,0]/args["max_areas"]) * 100, color = "#ECC216", lw = 3, ls = "dashdot")
    ax_tmp.plot(years, (crop_alloc_best[:,1,0]/args["max_areas"]) * 100, color = "#ECC216", lw = 3, ls = "--")
    
    # shade area between worst and best case
    ax_tmp.fill_between(years, (crop_alloc_worst[:,0,0]/args["max_areas"]) * 100,
                        (crop_alloc_best[:,0,0]/args["max_areas"]) * 100,
                      color = "#656C12", alpha = 0.5, label = "Range of rice")
    ax_tmp.fill_between(years, (crop_alloc_worst[:,1,0]/args["max_areas"]) * 100,
                        (crop_alloc_best[:,1,0]/args["max_areas"]) * 100,
                      color = "#ECC216", alpha = 0.5, label = "Range of maize")
     
    # subplot titles
    ax_tmp.set_title("Region " + str(cl), fontsize = 16)
    
    # set limits and fontsizes
    ax_tmp.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax_tmp.set_ylim(-5, 105)
    ax_tmp.set_xticks(ticks)   
    ax_tmp.xaxis.set_tick_params(labelsize=16)
    ax_tmp.yaxis.set_tick_params(labelsize=16)
    ax_tmp.yaxis.offsetText.set_fontsize(16)
    ax_tmp.xaxis.offsetText.set_fontsize(16)
        
    
# add a big axis, hide frame, ticks and tick labels of overall axis
panelA.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Year", fontsize = 24, labelpad = 20)
plt.ylabel("Crop area as share of available arable area, %", fontsize = 24, labelpad = 20)

panelA.savefig("Figures/PublicationPlots/Figure4/CropAreas.jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
plt.close(panelA)


# LEGEND AS SEPARATE FIGURE
panelAlegend = plt.figure(figsize  = (5, 3))
legend_elements1 = [Line2D([0], [0], color ='black', lw = 2, 
                          label='worst case'),
                    Line2D([0], [0], color ='black', lw = 2, ls = "dashdot",
                          label='stationary'),
                    Line2D([0], [0], color ='black', lw = 2, ls = "--",
                          label='best case'),
                    # Line2D([0], [0], color ='dimgrey',  lw = 5,
                    #       label="Available arable area")
                    ]

legend_elements2 = [Patch(color ='#656C12', label='Rice'),
                    Patch(color ='#ECC216',  label='Maize')]

ax = panelAlegend.add_subplot(2, 1, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(handles = legend_elements1, fontsize = 14, loc = 6)

ax = panelAlegend.add_subplot(2, 1, 2)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(handles = legend_elements2, fontsize = 14, loc = 6)

panelAlegend.savefig("Figures/PublicationPlots/Figure4/CropAreasLegend.jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
plt.close(panelAlegend)