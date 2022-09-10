# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 22:51:26 2022

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
import cartopy.crs as ccrs

from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters

from ModelCode.PlotMaps import PlotClusterGroups

if not os.path.isdir("Figures/PublicationPlots/SI"):
    os.mkdir("Figures/PublicationPlots/SI")
    
    
# %% ######################### FIG 1 - MAP OF CLUSTERS ########################

PlotClusterGroups(k = 9, title = "", plot_cmap = False, 
                  basecolors = list(publication_colors.values()),
                  close_plt = False, figsize = (20, 15),
                  file = "Figures/PublicationPlots/Figure1_MapClusters")

PlotClusterGroups(k = 9, title = "", figsize = (20, 15), 
                  basecolors = list(publication_colors.values()),
                  file = "Figures/PublicationPlots/Figure1_MapClusters_wCmap")


# %% ######################## SI? - CLUSTER GROUOPINGS ########################


metric = "medoids"
aim = "Similar"
adj = "Adj"
adj_text = "True"


metric = "equality"
aim = "Similar"
adj = ""
adj_text = "False"

fig = plt.figure(figsize = (12,11))

for (met, aim, adj, idx) in[("medoids", "Similar", "Adj", 1),
                            ("equality", "Similar", "", 2)]:
    for (idx2, s) in enumerate([2,3,5]):
        ax = fig.add_subplot(3, 2, 2 * idx2 + idx, projection = ccrs.PlateCarree())
        BestGrouping, BestCosts, valid = \
                FS.GroupingClusters(k = 9, size = s, aim = aim, \
                    adjacent = adj, metric = met, title = None)
        PlotClusterGroups(grouping = BestGrouping, k = 9, title = "Group size " + str(s),
                          plot_cmap = False, close_plt = False, ax = ax)
        
fig.savefig("Figures/PublicationPlots/SI/ClusterGroups.jpg", bbox_inches = "tight", pad_inches = 1)

plt.close()
        