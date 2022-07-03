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
from string import ascii_uppercase as letter

from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters


# %%  CROP AREAS 

# three yield/population scenarios
# all 9 clusters as subplots to same figure
# default probability and government parameters
# crop areas for maize and rice over time

panelA = plt.figure(figsize = (16, 11))

panelA.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.95,
                wspace=0.15, hspace=0.35)

col1 = publication_colors["green"]
col2 = publication_colors["yellow"]

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))
    
    pos = letter.index(cluster_letters[cl-1]) + 1
    
    ax_tmp = panelA.add_subplot(3, 3, pos)

    # get results
    settings, args, yield_information, population_information, penalty_methods, \
    status, all_durations, exp_incomes, crop_alloc_worst, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High")
                
    settings, args, yield_information, population_information, penalty_methods, \
    status, all_durations, exp_incomes, crop_alloc_best, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed")            
                    
    settings, args, yield_information, population_information, penalty_methods, \
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
ax = panelA.add_subplot(111, frameon=False)
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

panelA.savefig("Figures/PublicationPlots/SI/Figure4Alternative_CropAreas.jpg", 
                bbox_inches = "tight", pad_inches = 1, format = "jpg")
plt.close(panelA)



# %%  CROP AREAS  -- ALTERNATIVE

# two yield/population scenarios as inner and outer pie chart
# all 9 clusters as columns (separate pie charts)
# two time steps as rows
# default probability and government parameters
# crop areas for maize and rice, and unused area

fig4 = plt.figure(figsize = (15, 5))

fig4.subplots_adjust(wspace=0.005)

colors = [publication_colors["green"], publication_colors["yellow"], publication_colors["grey"]]
size = 0.5

ax_tmp = fig4.add_subplot(3, 10, 1, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.85, "Year 2020  ", fontsize = 18)
    
ax_tmp = fig4.add_subplot(3, 10, 11, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlim(0,2)
plt.ylim(0,2)
plt.text(-0.2, 0.85, "Year 2030  ", fontsize = 18)

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))

    # get results
    settings, args, yield_information, population_information, penalty_methods, \
    status, all_durations, exp_incomes, crop_alloc_worst, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "fixed",
                                   pop_scenario = "High")
                
    settings, args, yield_information, population_information, penalty_methods, \
    status, all_durations, exp_incomes, crop_alloc_best, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = "trend",
                                   pop_scenario = "fixed")            
                    
    # settings, args, yield_information, population_information, penalty_methods, \
    # status, all_durations, exp_incomes, crop_alloc_fixed, meta_sol, \
    # crop_allocF, meta_solF, crop_allocS, meta_solS, \
    # crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
    #             FS.LoadFullResults(k_using = cl,
    #                                yield_projection = "fixed",
    #                                pop_scenario = "fixed")    
                
    def _getAreas(year, crops):
        areas = [crops[year,0,0], 
                crops[year,1,0], 
                args["max_areas"][0] - np.sum(crops[year,:,0])]
        return(areas)
    
    pos = letter.index(cluster_letters[cl-1]) + 1
    
    ax_tmp = fig4.add_subplot(3, 10, pos + 1)
    areas_outer = _getAreas(3, crop_alloc_worst)
    areas_inner = _getAreas(3, crop_alloc_best)
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"),
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8),
               startangle = 180, counterclock = False)
    ax_tmp.set_title(cluster_letters[cl-1], fontsize = 18)
        
    
    ax_tmp = fig4.add_subplot(3, 10, pos + 11)
    areas_outer = _getAreas(13, crop_alloc_worst)
    areas_inner = _getAreas(13, crop_alloc_best)
    ax_tmp.pie(areas_outer, radius = 1.2, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w"), 
               startangle = 180, counterclock = False)
    ax_tmp.pie(areas_inner, radius = 1.2-size, colors = colors,
               wedgeprops = dict(width = size, edgecolor = "w", alpha = 0.8), 
               startangle = 180, counterclock = False)
  
ax = fig4.add_subplot(3, 1, 3, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

legend_elements = [Patch(color = colors[0], alpha = 0.9, label='Rice'),
                   Patch(color = colors[1], alpha = 0.9, label='Maize'),
                   Patch(color = colors[2], alpha = 0.9, label='Not used'),
                   Patch(color = "w", label='Inner circle: best case scenario'),
                   Patch(color = "w", label='Outer circle: worst case scenario')]
ax.legend(handles = legend_elements, fontsize = 18,
          loc = "center", ncol = 2)


fig4.savefig("Figures/PublicationPlots/Figure4_CropAreas.jpg", 
                bbox_inches = "tight", pad_inches = 0.2, format = "jpg")
