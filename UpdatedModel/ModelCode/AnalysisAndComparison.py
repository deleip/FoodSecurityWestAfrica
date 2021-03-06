#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:07:33 2021

@author: Debbora Leip
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
import pickle

from ModelCode.Auxiliary import printing
from ModelCode.Auxiliary import GetFilename
from ModelCode.CompleteModelCall import LoadModelResults
from ModelCode.PandaHandling import ReadFromPandaSingleClusterGroup
from ModelCode.SettingsParameters import DefaultSettingsExcept

# %% ############################# ANALYSIS ###################################

def CompareCropAllocs(CropAllocs, MaxAreas, labels, title, legend_title, \
                      comparing = "clusters", cols = None, cols_b = None, filename = None, \
                      figsize = None, fig = None, ax = None, subplots = False, \
                      fs = "big", sim_start = 2017, plot_total_area = False, legends = True, plt_labels = True):
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
    if figsize is None:
        from ModelCode.GeneralSettings import figsize        
    
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
        titlepad = 20
        
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
        
    if (not subplots) and plot_total_area:
        total_area = np.nansum(CropAllocs[0], axis = (1,2))
        if len(CropAllocs) > 1:
            for cr in CropAllocs[1:]:
                total_area = total_area + np.nansum(cr, axis = (1,2))
        years = range(sim_start, sim_start + CropAllocs[0].shape[0])
        
    idx_col = -1
    total_K = 0
    for idx, cr in enumerate(CropAllocs):
        [T, J, K] = cr.shape
        total_K = total_K + K
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
            l1, = axTmp.plot(years, np.repeat(MaxAreas[idx][k], len(years)), \
                     color = cols_b[idx_col], lw = 5, alpha = 0.4)
            l2, = axTmp.plot(years, CropAllocs[idx][:,0,k], color = cols[idx_col], \
                     lw = 2, linestyle = "--")
            l3, = axTmp.plot(years, CropAllocs[idx][:,1,k], color = cols[idx_col], \
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
        if legends:
            plt.legend(loc='lower left', ncol=total_K, borderaxespad=0.6, fontsize = 9)
            # ax.legend(title = legend_title, fontsize = fs_axis, title_fontsize = fs_label, loc = 7)
             
    if plot_total_area:
            ax2 = axTmp.twinx()
            l0, = ax2.plot(years, total_area, color = "k", \
                 lw = 2.5, alpha = 0.9, ls = "-.")
            a_lim = np.nanmax(total_area)
            ax2.set_ylim([-0.01*a_lim, 1.05*a_lim])
            ax2.yaxis.set_tick_params(labelsize=fs_axis)
            ax2.yaxis.offsetText.set_fontsize(fs_axis)
        
    if legends:
        if plot_total_area:
            legend1 = ax2.legend([l0, l1, l2, l3], ["Total cultivated area", \
                                                    "Maximum available area", \
                                                "Area Rice", "Area Maize"], \
                                 fontsize = 9, loc = 1, handlelength = 2.5)
        else:
            legend1 = plt.legend([l1, l2, l3], ["Maximum available area", \
                                                "Area Rice", "Area Maize"], \
                                 fontsize = 9, loc = 1)            
        plt.gca().add_artist(legend1)
    
        
    ax.set_title(title, fontsize = fs_title, pad = titlepad)
    if plt_labels:
        ax.set_xlabel("Years", fontsize = fs_label, labelpad = 25)
        ax.set_ylabel(r"Crop area in [$10^6\,$ha]", fontsize = fs_label, labelpad = 25)
    
    if filename is not None:
        if subplots:
            filename = filename + "_sp"
        fig.savefig("Figures/CompareCropAllocs/" + filename + \
                    ".jpg", bbox_inches = "tight", pad_inches = 1, format = "jpg")
 #   plt.close()
    
def CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, MaxAreasPool, MaxAreasIndep, 
                                labelsPool, labelsIndep, title = None, plot_total_area = True,  
                                cols = None, cols_b = None, 
                                subplots = False, filename = None, figsize = None, 
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
    if figsize is None:
        from ModelCode.GeneralSettings import figsize     
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,2,1)
    CompareCropAllocs(CropAllocsPool, MaxAreasPool, labelsPool, \
                      "Risk pooling", "Cluster: ", fig = fig, ax = ax, \
                      subplots = subplots, fs = "small", \
                      plot_total_area = plot_total_area, legends = False, plt_labels = False)
    ax = fig.add_subplot(1,2,2)
    CompareCropAllocs(CropAllocsIndep, MaxAreasIndep, labelsIndep, \
                      "Independent", "Cluster: ", fig = fig, ax = ax, \
                      subplots = subplots, fs = "small", \
                      plot_total_area = plot_total_area, plt_labels = False)
        
    fig.add_subplot(111, frame_on = False)
    plt.tick_params(bottom = False, left = False, which = "both",
                    labelbottom = False, labelleft = False)
    plt.xlabel("Years", fontsize = 22, labelpad = 30)
    plt.ylabel(r"Crop area in [$10^6\,$ha]", fontsize = 22, labelpad = 30)
    
    if title is not None:
        fig.suptitle(title, fontsize = 24)
        
    if filename is not None:
        if subplots:
            filename = filename + "_sp"
        fig.savefig("Figures/CompareCropAllocsRiskPooling/" + filename + \
                    ".jpg", bbox_inches = "tight", pad_inches = 1)

# def GetResultsToCompare(ResType = "k_using", PenMet = "prob", probF = 0.99, \
#                        probS = 0.95, rhoF = None, rhoS = None, console_output = None, \
#                        groupSize = "", groupAim = "", adjacent = False, \
#                        validation = None, **kwargs):
#     """
#     Function that loads results from different model runs, with one setting 
#     changing while the others stay the same (e.g. different clusters for same
#     settings).

#     Parameters
#     ----------
#     ResType : str, optional
#         Which setting will be changed to compare different results. Needs to 
#         be the exact name of that setting. The default is "k_using".
#     PenMet : "prob" or "penalties", optional
#         "prob" if desired probabilities were given. "penalties" if penalties 
#          were given directly. The default is "prob".
#     probF : float, optional
#         demanded probability of keeping the food demand constraint (only 
#         relevant if PenMet == "prob"). The default is 0.95.
#     probS : float, optional
#         demanded probability of keeping the solvency constraint (only 
#         relevant if PenMet == "prob"). The default is 0.95.
#     rhoF : float or None, optional 
#         Value that will be used as penalty for shortciómings of the food 
#         demand (only relevant if PenMet == "penalties"). The default is None.
#     rhoS : float or None, optional 
#         Value that will be used as penalty for insolvency of the government 
#         fund (only relevant if PenMet == "penalties"). The default is None.
#     console_output : boolean, optional
#         Specifying whether the progress should be documented thorugh console 
#         outputs. The default is defined in ModelCode/GeneralSettings.
#     groupSize : int, optional
#         The size of the groups. If we load reults for different cluster groups,
#         this is relevant for the output filename. The default is "".
#     groupAim : str ("Similar" or "Dissimilar"), optional
#         The aim when grouping clusters. If we load reults for different cluster 
#         groups, this is relevant for the output filename. The default is "".
#     adjacent : boolean, optional
#         Whether clusters within a group are adjacent. If we load reults for 
#         different cluster groups, this is relevant for the output filename.
#         The default is False.
#     validation : None or int, optional
#         if not None, the objevtice function will be re-evaluated for 
#         validation with a higher sample size as given by this parameter. 
#         The default is None.  
#     **kwargs
#         settings for the model, passed to DefaultSettingsExcept()  

#     Returns
#     -------
#     CropAllocs : list
#          List of crop allocations for the different settings.
#     MaxAreas : list
#          List of maximum available agricultural areas for the different 
#          settings.
#     labels : list
#         List of labels for plots (given information on the setting that is 
#         changing).
#     fnIterate : str
#         Filename to be used as basis for saving figures using this data.

#     """
    
#     if console_output is None:
#         from ModelCode.GeneralSettings import console_output
    
#     print(kwargs)
#     settingsIterate = DefaultSettingsExcept(**kwargs)
#     fnIterate = filename(settingsIterate, PenMet, validation, probF, probS, \
#                      rhoF, rhoS, groupSize = groupSize, groupAim = groupAim, \
#                      adjacent = adjacent)
    
#     if type(kwargs["k_using"]) is tuple: 
#         settingsIterate["k_using"] = [settingsIterate["k_using"]]
#     if type(kwargs["k_using"] is list) and (sum([type(i) is int for \
#                     i in kwargs["k_using"]]) == len(kwargs["k_using"])):
#         settingsIterate["k_using"] = kwargs["k_using"]
        
        
#     settingsIterate["probF"] = probF
#     settingsIterate["probS"] = probS
#     settingsIterate["rhoF"] = rhoF
#     settingsIterate["rhoS"] = rhoS
#     settingsIterate["validation"] = validation
#     del settingsIterate["import"]
    
#     ToIterate = settingsIterate[ResType]
    
#     if type(ToIterate) is not list:
#         ToIterate = [ToIterate]
    
#     CropAllocs = []
#     MaxAreas = []
#     labels = []
    
#     for val in ToIterate:
#         printing(ResType + ": " + str(val), console_output = console_output)
#         if ResType == "k_using":
#             if type(val) is int:
#                 val = [val]
#             if type(val) is tuple:
#                 val = list(val)
#                 val.sort()
                
#         crop_alloc, meta_sol, status, durations, settings, args, \
#         yield_information, population_information, rhoF, rhoS, \
#         VSS_value, crop_alloc_vss, meta_sol_vss, \
#         validation_values, fn = FoodSecurityProblem(PenMet = PenMet,
#                                     console_output = console_output, **settingsIterate)
        
#         CropAllocs.append(crop_alloc)
#         MaxAreas.append(args["max_areas"])
#         labels.append(val)
        
#     return(CropAllocs, MaxAreas, labels, fnIterate)

def GetResultsToCompare(ResType = "k_using", panda_file = "current_panda", \
                               console_output = None, **kwargs):
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
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
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
    
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    if "PenMet" not in kwargs.keys():
        kwargs["PenMet"] = "prob"
    if kwargs["PenMet"] == "penalties":
        kwargs["probF"] = None
        kwargs["probS"] = None
    
    # prepare cluster groups
    if "k_using" in kwargs.keys():
        k_using = kwargs["k_using"]
        if ResType == "k_using":
            if (type(k_using) is list) and (type(k_using[0]) is tuple):
                k_using = [sorted(list(k_using_tmp)) for k_using_tmp in k_using]
            elif (type(k_using) is list) and (type(k_using[0]) is int):
                k_using = [[k_using_tmp] for k_using_tmp in k_using]
        else:
            if type(k_using) is tuple:
                k_using = sorted(list(k_using))
            elif type(k_using) is int:
                k_using = [k_using]
        kwargs["k_using"] = k_using

    ToIterate = kwargs[ResType]
    
    if type(ToIterate) is not list:
        ToIterate = [ToIterate]
        
    kwargs_tmp = kwargs.copy()
    del kwargs_tmp["PenMet"]    
    
    panda_filenames = pd.DataFrame()
    for it in ToIterate:
        kwargs_tmp[ResType] = it
        panda_filenames = panda_filenames.append(\
                        ReadFromPandaSingleClusterGroup(file = panda_file, 
                                          output_var = "Filename for full results",
                                          **kwargs_tmp))
     
    panda_filenames = panda_filenames.reset_index()
        
    CropAllocs = []
    MaxAreas = []
    labels = []  
    
    printing("  Fetching data", console_output = console_output)
    for idx, val in enumerate(ToIterate):
        printing("     " + ResType + ": " + str(val), console_output = console_output)
        
        settings, args, AddInfo_CalcParameters, yield_information, \
        population_information, status, durations, crop_alloc, meta_sol, \
        crop_alloc_vs, meta_sol_vss, VSS_value, validation_values = \
            LoadModelResults(panda_filenames.at[idx, "Filename for full results"])

        CropAllocs.append(crop_alloc)
        MaxAreas.append(args["max_areas"])
        labels.append(val)
        
    return(CropAllocs, MaxAreas, labels)

# def PlotCropAllClusterForGrouping(grouping, 
#                                 groupSize = "",
#                                 groupAim = "",
#                                 adjacent = False,
#                                 panda_file = "current_panda",
#                                 console_output = None,
#                                 **kwargs):

#     if console_output is None:
#         from ModelCode.GeneralSettings import console_output
      
#     if adjacent:
#         add = ", adjacent"
#     else:
#         add = ""
        
#     settingsIterate = DefaultSettingsExcept(k_using = grouping, **kwargs)
#     fnPlot = GetFilename(settingsIterate, groupSize = groupSize, groupAim = groupAim, \
#                       adjacent = adjacent)
    
#     CropAllocs, MaxAreas, labels = GetResultsToCompare(ResType="k_using",\
#                                             panda_file = panda_file,
#                                             console_output = console_output,
#                                             k_using = grouping, 
#                                             **kwargs)
    
#     # return(CropAllocs, MaxAreas, labels)
#     printing("  Plotting", console_output = console_output)
#     CompareCropAllocs(CropAllocs = CropAllocs,
#                       MaxAreas = MaxAreas,
#                       labels = labels,
#                       title = "Groups of size " + str(groupSize) + " (" + groupAim.lower() + add + ")",
#                       legend_title = "Cluster: ",
#                       comparing = "clusters", 
#                       filename = fnPlot, 
#                       subplots = (3,3)) 
    
#     return(None)
    
    
# def CropAreasDependingOnColaborationOld(panda_file = "current_panda", 
#                                     groupAim = "Dissimilar",
#                                     adjacent = False,
#                                     console_output = None,
#                                     **kwargs):
    
#     if console_output is None:
#         from ModelCode.GeneralSettings import console_output
        
#     if adjacent:
#         add = "Adj"
#     else:
#         add = ""
        
#     for size in [1,2,3,5,9]:
#     # for size in [2]:
#         printing("\nGroup size " + str(size), console_output = console_output)
#         with open("InputData/Clusters/ClusterGroups/GroupingSize" \
#                       + str(size) + groupAim + add + ".txt", "rb") as fp:
#                 BestGrouping = pickle.load(fp)
#         PlotCropAllClusterForGrouping(BestGrouping, 
#                                     groupSize = size,
#                                     groupAim = groupAim,
#                                     adjacent = adjacent,
#                                     panda_file = panda_file,
#                                     console_output = console_output,
#                                     **kwargs)
        
#     return(None)

def CropAreasDependingOnColaboration(panda_file = "current_panda", 
                                    groupAim = "Dissimilar",
                                    adjacent = False,
                                    console_output = None,
                                    **kwargs):
    
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
        
    if adjacent:
        add = "Adj"
        add_title = ", adjacent"
    else:
        add = ""
        add_title = ""
        
    printing("\nGroup size " + str(1), console_output = console_output)
    with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                  + str(1) + groupAim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)    
                
    settingsIterate = DefaultSettingsExcept(k_using = BestGrouping, **kwargs)
    fnPlot = GetFilename(settingsIterate, groupSize = 1, groupAim = groupAim, \
                      adjacent = adjacent)
    
    CropAllocs, MaxAreas, labels = GetResultsToCompare(ResType="k_using",\
                                            panda_file = panda_file,
                                            console_output = console_output,
                                            k_using = BestGrouping, 
                                            **kwargs)
    
    # return(CropAllocs, MaxAreas, labels)
    printing("  Plotting", console_output = console_output)
    CompareCropAllocs(CropAllocs = CropAllocs,
                      MaxAreas = MaxAreas,
                      labels = labels,
                      title = "Groups of size " + str(1) + " (" + groupAim.lower() + add_title + ")",
                      legend_title = "Cluster: ",
                      comparing = "clusters", 
                      filename = fnPlot, 
                      subplots = (3,3))    
        
    
    for size in [2,3,5,9]:
        printing("\nGroup size " + str(size), console_output = console_output)
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + groupAim + add + ".txt", "rb") as fp:
                    BestGrouping = pickle.load(fp)
         
        settingsIterate = DefaultSettingsExcept(k_using = BestGrouping, **kwargs)
        fnPlot = GetFilename(settingsIterate, groupSize = str(size), groupAim = groupAim, \
                          adjacent = adjacent)
            
        CropAllocs_pooling, MaxAreas_pooling, labels_pooling = \
                GetResultsToCompare(ResType="k_using",\
                                           panda_file = panda_file,
                                           console_output = console_output,
                                           k_using = BestGrouping, 
                                           **kwargs)  
                    
        printing("  Plotting", console_output = console_output)
        CompareCropAllocs(CropAllocs = CropAllocs_pooling,
                          MaxAreas = MaxAreas_pooling,
                          labels = labels_pooling,
                          title = "Groups of size " + str(size) + " (" + groupAim.lower() + add_title + ")",
                          legend_title = "Cluster: ",
                          comparing = "clusters", 
                          filename = fnPlot, 
                          subplots = (3,3)) 
        
        CompareCropAllocRiskPooling(CropAllocs_pooling, CropAllocs, 
                                        MaxAreas_pooling, MaxAreas, 
                                        labels_pooling, labels, 
                                        filename = fnPlot,
                                        title = str(BestGrouping),
                                        subplots = False)   
        
    return(None)


# def CompareAllCooperationLevel(panda_file = "current_panda",
#                                 groupAim = "Dissimilar",
#                                 adjacent = False,
#                                 console_output = None,
#                                 **kwargs):

#     if console_output is None:
#         from ModelCode.GeneralSettings import console_output
      
#     if adjacent:
#         add = "Adj"
#     else:
#         add = ""
        
    
#     printing("\nGroup size " + str(1), console_output = console_output)
#     with open("InputData/Clusters/ClusterGroups/GroupingSize" \
#                   + str(1) + groupAim + add + ".txt", "rb") as fp:
#                 BestGrouping = pickle.load(fp)
    
        
#     CropAllocs, MaxAreas, labels = GetResultsToCompare(ResType="k_using",\
#                                             panda_file = panda_file,
#                                             console_output = console_output,
#                                             k_using = BestGrouping, 
#                                             **kwargs)    
        
#     for size in [2,3,5,9]:
#     # for size in [2]:
#         printing("\nGroup size " + str(size), console_output = console_output)
#         with open("InputData/Clusters/ClusterGroups/GroupingSize" \
#                       + str(size) + groupAim + add + ".txt", "rb") as fp:
#                     BestGrouping = pickle.load(fp)
         
#         settingsIterate = DefaultSettingsExcept(k_using = BestGrouping, **kwargs)
#         fnPlot = GetFilename(settingsIterate, groupSize = str(size), groupAim = groupAim, \
#                           adjacent = adjacent)
            
#         CropAllocs_pooling, MaxAreas_pooling, labels_pooling = \
#                 GetResultsToCompare(ResType="k_using",\
#                                            panda_file = panda_file,
#                                            console_output = console_output,
#                                            k_using = BestGrouping, 
#                                            **kwargs)    
    
#         printing("  Plotting", console_output = console_output)
#         CompareCropAllocRiskPooling(CropAllocs_pooling, CropAllocs, 
#                                         MaxAreas_pooling, MaxAreas, 
#                                         labels_pooling, labels, 
#                                         filename = fnPlot,
#                                         title = str(BestGrouping),
#                                         subplots = False)        
    
#     return(None)