#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:19:25 2021

@author: debbora
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from ModelCode.GeneralSettings import figsize

# %% ########################## PLOTTING FUNCTIONS ############################  

def PlotModelOutput(PlotType = "CropAlloc", cols = None, cols_b = None, \
                    figsize = figsize, title = None, file = None, **kwargs):
    """
    Creating different types of plots based on the model in- and output

    Parameters
    ----------
    PlotType : str, optional
        Specifying what kind of information should be plotted. The default is 
        "CropAlloc".
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    title : str
        Basis of title for for the plot (will be completed with information on
        the clusters). The default is None.
    file : str
        Filename to save resulting figure. If None, figure will not be saved.
        The default is None.    
    **kwargs
        Additional parameters passed along to the different plotting functions.

    Returns
    -------
    None.

    """
    
    # defining colors
    if cols is None:            
        cols = ["royalblue", "darkred", "grey", "gold", \
                "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                "indigo"]
    if cols_b is None:
        cols_b = ["dodgerblue", "red", "darkgrey", "khaki", \
              "lime", "cyan", "orchid", "lightseagreen", "mediumpurple"]      
            
    # plotting the specified information
    if PlotType == "CropAlloc":
        PlotCropAlloc(cols = cols, cols_b = cols_b, figsize = figsize, \
                      title = title, file = file, **kwargs)
    
    return()

def PlotCropAlloc(crop_alloc, k, k_using, max_areas, cols = None, cols_b = None, \
                  figsize = figsize, title = None, file = None, sim_start = 2017):
    """
    Plots crop area allocations over the years for all clusters. Should be 
    called through PlotModelOutput(PlotType = "CropAlloc").

    Parameters
    ----------
    crop_alloc : np.array of size (T*num_crops*len(k_using),)
        Gives allocation of area to each crop in each cluster.
    k : int, optional
        Number of clusters in which the area is to be devided. 
    k_using :  "all" or a list of int
        Specifies which of the clusters are to be considered in the model. 
        The default is "all".
    max_areas : np.array of size (len(k_using),) 
        Upper limit of area available for agricultural cultivation in each
        cluster
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    title : str
        Basis of title for for the plot (will be completed with information on
        the clusters). The default is None.
    file : str
        Filename to save resulting figure. If None, figure will not be saved.
        The default is None.
    sim_start : int
        The first year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    
    if title is None:
        title = ""
    else:
        title = " - " + title
        
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    [T, J, K] = crop_alloc.shape
    years = range(sim_start, sim_start + T)
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.95,
                    wspace=0.15, hspace=0.35)
    
    # overview of all crop area allocations in one plot
    if K > 1:
        ax = fig.add_subplot(1,2,1)
        l0, = plt.plot(years, np.sum(crop_alloc, axis = (1,2)), color = "k", \
                 lw = 2, alpha = 0.7, ls = "-.")
    else:
        ax = fig.add_subplot(1,1,1)
    for cl in range(0, K):
        l1, = plt.plot(years, np.repeat(max_areas[cl], len(years)), \
                 color = cols_b[k_using[cl]-1], lw = 5, alpha = 0.4)
        l2, = plt.plot(years, crop_alloc[:,0,cl], color = cols[k_using[cl]-1], \
                 lw = 1.8, linestyle = "--")
        l3, = plt.plot(years, crop_alloc[:,1,cl], color = cols[k_using[cl]-1], \
                 lw = 1.8, label = "Cluster " + str(k_using[cl]))
    plt.xlim(years[0] - 0.5, years[-1] + 0.5)
    # plt.ylim(-0.05 * np.max(max_areas), 1.1 * np.max(max_areas))
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.yaxis.offsetText.set_fontsize(24)
    ax.xaxis.offsetText.set_fontsize(24)
    plt.xlabel("Years", fontsize = 26)
    plt.ylabel(r"Crop area in [$10^6$ ha]", fontsize = 26)
    if K > 2:
        plt.suptitle(str(k_using) + title, fontsize = 30)
    elif K == 1:
        plt.title("Cluster " + str(k_using[0]) + title, \
                  fontsize = 26, pad = 10)
    elif K == 2:
        plt.title("Cluster " + str(k_using[0]) + " and " + str(k_using[1]) + title, \
                 fontsize = 26, pad = 10)
    if K == 1:
        legend1 = plt.legend([l1, l2, l3], ["Maximum available area", \
                                            "Area Rice", "Area Maize"], \
                             fontsize = 20, loc = 2)
    else:
        legend1 = plt.legend([l0, l1, l2, l3], ["Total cultivated area", \
                                                "Maximum available area", \
                                                "Area Rice", "Area Maize"], \
                             fontsize = 20, loc = 2)
        plt.legend(loc='lower left', ncol=K, borderaxespad=0.6, fontsize = 16)
    plt.gca().add_artist(legend1)
    
    # crop area allocations in separate subplots per cluster
    if K > 2:
        rows = math.ceil(K/2)
        whichplots = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
        for cl in range(0, K):
            ax = fig.add_subplot(rows, 4, whichplots[cl])
            plt.plot(years, np.repeat(max_areas[cl], len(years)), \
                     color = cols_b[k_using[cl]-1], lw = 5, alpha = 0.4)
            plt.plot(years, crop_alloc[:,0,cl], color = cols[k_using[cl]-1], \
                     lw = 1.8, linestyle = "--")
            plt.plot(years, crop_alloc[:,1,cl], color = cols[k_using[cl]-1], \
                     lw = 1.8, label = "Cluster " + str(k_using[cl]))
            plt.ylim(-0.05 * np.max(max_areas), 1.1 * np.max(max_areas))
            plt.xlim(years[0] - 0.5, years[-1] + 0.5)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.yaxis.offsetText.set_fontsize(16)
            ax.xaxis.offsetText.set_fontsize(16)
            # ax.text(0.05, 0.91, "Cluster " + str(int(k_using[cl])), \
            #         fontsize = 16, transform = ax.transAxes, \
            #         verticalalignment = 'top', bbox = props)
            # if cl == 0:
            #     plt.title("                        Separate clusters", \
            #               fontsize = 32, pad = 8) 
    
    
    if file is not None:
        if not os.path.isdir("Figures/CropAllocs/" + str(K) + "clusters"):
            os.mkdir("Figures/CropAllocs/" + str(K) + "clusters") 
        fig.savefig("Figures/CropAllocs/" + str(K) + "clusters/CropAlloc_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    return()