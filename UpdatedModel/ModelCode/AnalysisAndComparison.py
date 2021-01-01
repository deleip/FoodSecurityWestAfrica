#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:07:33 2021

@author: Debbora Leip
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from ModelCode.GeneralSettings import figsize
from ModelCode.Auxiliary import filename
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import printing
from ModelCode.CompleteModelCall import FoodSecurityProblem

# %% ############################# ANALYSIS ###################################

def CompareCropAllocs(CropAllocs, MaxAreas, labels, title, legend_title, \
                      comparing = "clusters", cols = None, cols_b = None, filename = None, \
                      figsize = figsize, fig = None, ax = None, subplots = False, \
                      fs = "big", sim_start = 2017):
    """
    Creates a plot of the crop areas over time, either all in one plot or as
    separate subplots.

    Parameters
    ----------
    CropAllocs : list
        List of crop areas.
    MaxAreas : list
        List of maximum available areas.
    labels : list
        List of labels for the plot.
    title : str
        Title of the plot.
    legend_title : str
        Titlel of the legend.
    comparing : str, optional
        What cahnges between different results. Mainly relevant, as "clusters" 
        is treated differently. Any other string will be treated the same. The
        default is "clusters".
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    filename : str, optional
        Filename to save the resulting plot. If None, plot is not saved. The
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    fig : figure, optional
        If the function should create a plot within an already existing
        figure, the figure has to be passed to the function. The default is 
        None.
    ax : AxesSubplot, optional
        If the function should create a plot as a specific subplot of a figure, 
        the AxesSubplot has to be passed to the function. The default is None.
    subplots : boolean, optional
        If True, for each setting a new subplot is used. If False, all are 
        plotted in the same plot. The default is False.
    fs : str, optional
        Defines the fontsize, if fs is "big", larger fontsize is chosen, for 
        any other string smaller fontsizes are chosen. the smaller dontsizes
        should be used for figures with many plots. The default is "big".
    sim_start : int, optional
        First year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    
    if fs == "big":
        fs_axis = 24
        fs_label = 28
        fs_title = 30
        fs_sptitle = 18
        titlepad = 40
    else:
        fs_axis = 16
        fs_label = 18
        fs_title = 20
        fs_sptitle = 14
        titlepad = 30
        
    
    if cols is None:            
        cols = ["royalblue", "darkred", "grey", "gold", \
                "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                "indigo"]
    if cols_b is None:
        cols_b = ["dodgerblue", "red", "darkgrey", "y", \
              "lime", "cyan", "orchid", "lightseagreen", "mediumpurple"]  
    
    if fig is None:
        fig = plt.figure(figsize = figsize)
        
    if ax is None:
        ax = fig.add_subplot(1,1,1)
    if subplots:            
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        inner = gs.GridSpecFromSubplotSpec(subplots[0], subplots[1], ax, wspace=0.3, hspace=0.3)
        which = -1
        
    idx_col = -1
    for idx, cr in enumerate(CropAllocs):
        [T, J, K] = cr.shape
        years = range(sim_start, sim_start + T)
        for k in range(0, K):
            if comparing == "clusters":
                idx_col = labels[idx][k] - 1
                label = str(labels[idx][k])
            else:
                idx_col += 1
                label = labels[idx_col]
            if subplots:
                if (comparing == "clusters") and (subplots == (3,3)):
                    which = labels[idx][k] - 1
                else:
                    which += 1
                axTmp = plt.Subplot(fig, inner[which])
            else:
                axTmp = ax
            axTmp.plot(years, np.repeat(MaxAreas[idx][k], len(years)), \
                     color = cols_b[idx_col], lw = 5, alpha = 0.4)
            axTmp.plot(years, CropAllocs[idx][:,0,k], color = cols[idx_col], \
                     lw = 2, linestyle = "--")
            axTmp.plot(years, CropAllocs[idx][:,1,k], color = cols[idx_col], \
                     lw = 2, label = label)
            if subplots:
                axTmp.set_title(legend_title + str(label), fontsize = fs_sptitle)
                axTmp.set_xlim(years[0] - 0.5, years[-1] + 0.5)
                axTmp.xaxis.set_tick_params(labelsize=12)
                axTmp.yaxis.set_tick_params(labelsize=12)
                axTmp.yaxis.offsetText.set_fontsize(12)
                axTmp.xaxis.offsetText.set_fontsize(12)
                fig.add_subplot(axTmp)
                    
    if not subplots:
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.xaxis.set_tick_params(labelsize=fs_axis)
        ax.yaxis.set_tick_params(labelsize=fs_axis)
        ax.yaxis.offsetText.set_fontsize(fs_axis)
        ax.xaxis.offsetText.set_fontsize(fs_axis)
        ax.legend(title = legend_title, fontsize = fs_axis, title_fontsize = fs_label, loc = 7)
        
    ax.set_title(title, fontsize = fs_title, pad = titlepad)
    ax.set_xlabel("Years", fontsize = fs_label, labelpad = 30)
    ax.set_ylabel(r"Crop area in [$10^6$ ha]", fontsize = fs_label, labelpad = 30)
    
    if filename is not None:
        if subplots:
            filename = filename + "_sp"
        fig.savefig("Figures/CompareCropAllocs/" + filename + \
                    ".jpg", bbox_inches = "tight", pad_inches = 1, format = "jpg")
    plt.close()
    
def CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, MaxAreasPool, MaxAreasIndep, 
                                labelsPool, labelsIndep, title = None, cols = None, cols_b = None, 
                                subplots = False, filename = None, figsize = figsize, 
                                sim_start = 2017):
    """
    Given two list of crop areas, max available areas and labels, this plots 
    the crop areas for comparison, either with subplots for each cluster of 
    each list, or all cluster per list in one plot.

    Parameters
    ----------
    CropAllocsPool : list
        List of crop areas of setting 1.
    CropAllocsIndep : list
        List of crop areas of setting 2.
    MaxAreasPool : list
        List of maximum available area for setting 1.
    MaxAreasIndep : list
        List of maximum available area for setting 2.
    labelsPool : list
        Labels for setting 1.
    labelsIndep : list
        Labels for setting 2.
    title : str, optional
        Title of the plot. The default is None.
    cols : list, optional
        List of colors to use. If None, a standard set of colors is used. The 
        default is None.
    cols_b : list, optional
        List of colors in a lighter shade to use. If None, a standard set of
        colors is used. The default is None.
    subplots : TYPE, optional
        DESCRIPTION. The default is False.
    filename : str, optional
        Filename to save the resulting plot. If None, plot is not saved. The
        default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    sim_start : int, optional
        First year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,2,1)
    CompareCropAllocs(CropAllocsPool, MaxAreasPool, labelsPool, "Risk pooling", "Cluster: ", \
                      fig = fig, ax = ax, subplots = subplots, fs = "small")
    ax = fig.add_subplot(1,2,2)
    CompareCropAllocs(CropAllocsIndep, MaxAreasIndep, labelsIndep, "Independent", "Cluster: ", \
                      fig = fig, ax = ax, subplots = subplots, fs = "small")
    
    if title is not None:
        fig.suptitle(title, fontsize = 30)
        
    if filename is not None:
        if subplots:
            filename = filename + "_sp"
        fig.savefig("Figures/CompareCropAllocsRiskPooling/" + filename + \
                    ".jpg", bbox_inches = "tight", pad_inches = 1)

def GetResultsToCompare(ResType = "k_using", PenMet = "prob", probF = 0.95, \
                       probS = 0.95, rhoF = None, rhoS = None, prints = True, \
                       groupSize = "", groupAim = "", adjacent = False, \
                       validation = None, **kwargs):
    """
    Function that loads results from different model runs, with one setting 
    changing while the others stay the same (e.g. different clusters for same
    settings).

    Parameters
    ----------
    ResType : str, optional
        Which setting will be changed to compare different results. Needs to 
        be the exact name of that setting. The default is "k_using".
    PenMet : "prob" or "penalties", optional
        "prob" if desired probabilities were given. "penalties" if penalties 
         were given directly. The default is "prob".
    probF : float, optional
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    probS : float, optional
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float or None, optional 
        Value that will be used as penalty for shortciómings of the food 
        demand (only relevant if PenMet == "penalties"). The default is None.
    rhoS : float or None, optional 
        Value that will be used as penalty for insolvency of the government 
        fund (only relevant if PenMet == "penalties"). The default is None.
    groupSize : int, optional
        The size of the groups. If we load reults for different cluster groups,
        this is relevant for the output filename. The default is "".
    groupAim : str ("Similar" or "Dissimilar"), optional
        The aim when grouping clusters. If we load reults for different cluster 
        groups, this is relevant for the output filename. The default is "".
    adjacent : boolean, optional
        Whether clusters within a group are adjacent. If we load reults for 
        different cluster groups, this is relevant for the output filename.
        The default is False.
    validation : None or int, optional
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.  
    **kwargs
        settings for the model, passed to DefaultSettingsExcept()  

    Returns
    -------
    CropAllocs : list
         List of crop allocations for the different settings.
    MaxAreas : list
         List of maximum available agricultural areas for the different 
         settings.
    labels : list
        List of labels for plots (given information on the setting that is 
        changing).
    fnIterate : str
        Filename to be used as basis for saving figures using this data.

    """

    settingsIterate = DefaultSettingsExcept(**kwargs)
    fnIterate = filename(settingsIterate, PenMet, validation, probF, probS, \
                     rhoF, rhoS, groupSize = groupSize, groupAim = groupAim, \
                     adjacent = adjacent)
    
    if type(kwargs["k_using"]) is tuple: 
        settingsIterate["k_using"] = [settingsIterate["k_using"]]
    if type(kwargs["k_using"] is list) and (sum([type(i) is int for \
                    i in kwargs["k_using"]]) == len(kwargs["k_using"])):
        settingsIterate["k_using"] = kwargs["k_using"]
        
        
    settingsIterate["probF"] = probF
    settingsIterate["probS"] = probS
    settingsIterate["rhoF"] = rhoF
    settingsIterate["rhoS"] = rhoS
    settingsIterate["validation"] = validation
    
    ToIterate = settingsIterate[ResType]
    
    if type(ToIterate) is not list:
        ToIterate = [ToIterate]
    
    CropAllocs = []
    MaxAreas = []
    labels = []
    
    for val in ToIterate:
        printing(ResType + ": " + str(val), prints = prints)
        if ResType == "k_using":
            if type(val) is int:
                val = [val]
            if type(val) is tuple:
                val = list(val)
                val.sort()
        settingsIterate[ResType] = val
        
        crop_alloc, meta_sol, status, durations, settings, args, \
        rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
        validation_values, fn = FoodSecurityProblem(PenMet = PenMet,
                                    prints = prints, **settingsIterate)
        
        CropAllocs.append(crop_alloc)
        MaxAreas.append(args["max_areas"])
        labels.append(val)
        
    return(CropAllocs, MaxAreas, labels, fnIterate)