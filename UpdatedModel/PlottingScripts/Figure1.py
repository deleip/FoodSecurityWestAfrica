# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 19:23:58 2021

@author: leip
"""
# set the right directory
import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gs
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ModelCode.GeneralSettings import figsize


# %% ################## PANEL A - CULTIVATED AREA OVER TIME  ##################

# All clusters as subolts (no cooperation)
# For each cluster all yield and population scenarios 
# Different crops shown by linecolor 

# markers = [".", "X", "v", "p", "s", "*", "2", "_"]

# panelA = plt.figure(figsize = figsize)
# ax = panelA.add_subplot(1,1,1)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.set_yticks([])
# ax.set_xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# inner = gs.GridSpecFromSubplotSpec(3, 3, ax, wspace = 0.2, hspace = 0.3)
cols = ["red", "green"]

print("Plotting in progress ...", flush = True)
for cl in range(1, 10):
    print("             ... cluster " + str(cl))
    panelA  = plt.figure(figsize = figsize)
    ax = panelA.axes
    # ax_tmp = plt.Subplot(panelA, inner[cl - 1])
    idx = -1

    for (y, p) in [("fixed", "High"), ("trend", "fixed")]:
        idx = idx + 1
        
        # if cl == 1:
        #     ax.scatter([], [], marker = markers[idx], 
        #          label = "yield " + y + ", population trend " + p.lower(),
        #          color = "black")
        
        # get results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p)
                    
        sim_start = settings["sim_start"]
        T = settings["T"]
        years = range(sim_start, sim_start + T)
        ticks = np.arange(sim_start, sim_start + T + 0.1, 3)
        
        # crop_ratio = np.sum(crop_alloc, axis = 2)[:,1]/np.sum(crop_alloc, axis = (1, 2))
        # colors = list(zip((255 - crop_ratio * (255 - 23))/255,
        #                   (236 - crop_ratio * (236 - 86))/255,
        #                   (31 - crop_ratio * (31 - 30))/255))
        
        # plt.fill_between(years, np.repeat(0, T), np.repeat(args["max_areas"], T),
        #                  color = "gainsboro")
        plt.plot([years[0] - 0.5, years[-1] + 0.5], np.repeat(args["max_areas"], 2),
                color = "dimgrey", lw = 5, alpha = 0.4)
        plt.plot(years, crop_alloc[:,0,0], color = cols[idx], lw = 2,
                 label = "yield " + y + ", population trend " + p.lower())
        plt.plot(years, crop_alloc[:,1,0], ls = "--", lw = 2, color = cols[idx])
    
    plt.xlabel("Year", fontsize = 24)
    plt.ylabel("Area", fontsize = 24)
    plt.title("Cluster " + str(cl), fontsize = 30) 
    plt.xlim(years[0] - 0.5, years[-1] + 0.5)
    plt.ylim((-args["max_areas"] * 0.03, args["max_areas"]*1.06))
    plt.xticks(ticks)    
    
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.yaxis.offsetText.set_fontsize(12)
    ax.xaxis.offsetText.set_fontsize(12)
    
    legend_elements1 = [Line2D([0], [0], color ='dimgrey',  lw = 5,
                             label="Available arable area"),
                        Line2D([0], [0], color ='red', lw = 2, 
                             label='Fixed yield, high population growth'),
                       Line2D([0], [0], color ='green', lw = 2, 
                             label='Yield trends, fixed population'),]
    legend_elements2 = [Line2D([0], [0], color ='black',
                             label='Rice', lw = 2),
                       Line2D([0], [0], color ='black', lw = 2,
                             label='Maize', ls = "--")]
    
    legend1 = plt.legend(handles = legend_elements1,
                 fontsize = 14, loc = 1)
    plt.gca().add_artist(legend1)
    plt.legend(handles = legend_elements2,
                 fontsize = 14, loc = 4)
    
    panelA.savefig("Figures/PublicationPlots/Fig1PanelA_cl" + str(cl) + ".jpg", 
                    bbox_inches = "tight", pad_inches = 1, format = "jpg")
    plt.close(panelA)
    # panelA.add_subplot(ax_tmp)
    
    
# axins = inset_axes(ax, width = "50%", height = "50%",
#                    loc = "lower left", bbox_to_anchor = (1.02, 0.5, 1, 1))
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1, 236/255, 31/255), (23/255, 86/255, 20/255)])
# m = ax.pcololormesh([], [], cmap = cmap, levels = [0, 1])
# panelA.colorbar(cm.ScalarMappable(norm = None, cmap = cmap), ax = ax, shrink = 0.5, orientation = "horizontal")
# ax.legend(bbox_to_anchor = (1.02, 0.5), loc = 'center left', fontsize = 14)
# panelA.savefig("Figures/PublicationPlots/Fig1PanelA.jpg", 
#                bbox_inches = "tight", pad_inches = 1, format = "jpg")
# plt.close(panelA)
            
            
# %% ################ PANEL B,C - FOOD PRODUCTION DISTRIBUTION ################

# Middle of the road scenario (yield trends, medium population growth)
# Changing probability for food security
# Year 2030 (link to SDGs, Agenda 2030)
# Each cluster as separate plot (one good and one bad in main text, rest in SI)

p = "fixed"
y = "fixed"

for cl in range(1, 10):
    panelB = plt.figure(figsize = figsize)
    
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        # get results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p,
                                       probF = alpha)
                    
        year_rel = 2030 - settings["sim_start"]
        plt.hist(meta_sol["food_supply"].flatten(), bins = 200, alpha = 0.5,
                 density = True, label = str(alpha * 100) + "%")
    
    plt.axvline(args["demand"][0], color = "blue", linestyle = "dashed", alpha = 0.6, label = "Food demand")
    plt.xlabel(r"Food production in cluster [$10^{12}\,kcal$]")
    plt.title("Distribution of food production for cluster " + str(cl))
    plt.legend()
        
    panelB.savefig("Figures/PublicationPlots/Fig1PanelB_cl" + str(cl) + ".jpg", 
               bbox_inches = "tight", pad_inches = 1, format = "jpg")
            
    # plt.close(panelB)


# %% ######################## PANEL A* - PARETO FRONT #########################

# Input probability for food security vs. resulting total cultivation costs
# Front for different scenarios

panelAs = plt.figure(figsize = figsize)
alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995]

for (y, p) in [("fixed", "fixed"), 
                ("trend", "Medium"),
                ("fixed", "High"),
                ("trend", "fixed")]:
    
    costs = []
    
    for alpha in alphas:
        
        print("alpha:" + str(alpha) + ", yield " + y + ", population " + p)
        
        tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                          out_type = "agg_sum",
                                          sizes = 1,
                                          yield_projection = y,
                                          pop_scenario = p,
                                          probF = alpha)
        
        costs.append(tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0])

    
    plt.scatter(alphas, costs, label =  "yield " + y + ", population trend " + p.lower())

plt.xlabel("Input probability for food security")
plt.ylabel("Total cultivation costs")
plt.legend()
plt.title("Trade-off between food security probability and cultivation costs")

panelAs.savefig("Figures/PublicationPlots/Fig1PanelA_ParetoFront.jpg", 
           bbox_inches = "tight", pad_inches = 1, format = "jpg")

plt.close(panelAs)



##############################################################################
##############################################################################



# %% #####################

# All clusters as subolts (no cooperation)
# For each cluster all yield and population scenarios 
# Different crops shown by linecolor 

panelA = plt.figure(figsize = figsize)
linestyles = ["-", "--", "-.", ":"]

print("Plotting in progress ...", flush = True)
for cl in range(1, 3):
    print("             ... cluster " + str(cl))
    ax = panelA.add_subplot(3, 3, cl)
    idx = -1
    
    for p in ["fixed", "Low", "Medium", "High"]:
        idx = idx + 1
        for y in ["fixed"]:
            
            # get results
            settings, args, yield_information, population_information, \
            status, all_durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vss, meta_sol_vss, VSS_value, validation_values = \
                        FS.LoadFullResults(k_using = cl,
                                           yield_projection = y,
                                           pop_scenario = p)
                        
            sim_start = settings["sim_start"]
            T = settings["T"]
            years = range(sim_start, sim_start + T)
            ticks = np.arange(sim_start, sim_start + T + 0.1, 3)
            
            # plt.fill_between(years, np.repeat(0, T), np.repeat(args["max_areas"], T),
            #                  color = "gainsboro")
            plt.plot([years[0] - 0.5, years[-1] + 0.5],
                     np.repeat(args["max_areas"], 2), color = "dimgrey")
            plt.plot(years, np.sum(crop_alloc, axis = (1, 2)),
                     ls = linestyles[idx], color = "green")
            
            ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
            ax.set_ylim((-args["max_areas"] * 0.03, args["max_areas"]*1.06))
            ax.set_xticks(ticks)
            
            
            
            
###


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

N = 10
np.random.seed(101)
x = np.random.rand(N)
y = np.random.rand(N)
fig, ax = plt.subplots()

path = mpath.Path(np.column_stack([x, y]))
verts = path.interpolated(steps=3).vertices
x, y = verts[:, 0], verts[:, 1]
z = np.linspace(0, 1, len(x))
colorline(years, np.sum(crop_alloc, axis = (1,2)), cmap=plt.get_cmap('jet'), linewidth=2)

plt.show()