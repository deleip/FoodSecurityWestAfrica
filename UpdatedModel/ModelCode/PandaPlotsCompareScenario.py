# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:32:24 2021

@author: Debbora Leip
"""
import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
from textwrap import wrap

from ModelCode.PandaPlotsSingleScenario import PandaToPlot_GetResultsSingScen

# %% ############### PLOTTING FUNCTIONS USING RESULTS PANDA CSV ###############

def PandaToPlot_GetResultsMultScen(file = "current_panda", 
                                   output_var = None,
                                   out_type = "agg", # or median, or all
                                   grouping_aim = "Dissimilar",
                                   adjacent = False,
                                   **kwargs):
    
    fulldict = kwargs.copy()
    fulldict["file"] = file
    fulldict["out_type"] = out_type
    fulldict["grouping_aim"] = grouping_aim
    fulldict["adjacent"] = adjacent

    l = []
    keys_list = []
    for key in fulldict.keys():
        if type(fulldict[key]) is list:
            l.append(len(fulldict[key]))
            keys_list.append(key)
            
    if (len(l) > 0) and (not all(ls == l[0] for ls in l)):
        sys.exit("All settings over which should be iterated must be " +
                     "lists of the same length!")
     
    if len(l) == 0:
        res = [PandaToPlot_GetResultsSingScen(output_var = output_var, **fulldict)]
    else:
        res = []
        for idx in range(0, l[0]):
            fulldict_tmp = fulldict.copy()
            for key in keys_list:
                fulldict_tmp[key] = fulldict[key][idx]
            try:
                res.append(PandaToPlot_GetResultsSingScen(output_var = output_var, **fulldict_tmp))
            except SystemExit:
                res.append(None)
                
    return(res)

def PlotPandaMedianScenarios(panda_file = "current_panda", 
                    output_var = None,
                    grouping_aim = "Dissimilar",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_file = None,
                    foldername = None,
                    close_plots = None,
                    cols = None,
                    **kwargs):
    """
    Creates plots visualizing min, max and median of the specified output 
    variables for all cluster group sizes for the given gorouping type.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : list of str or str
        The variables that are reported.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    figsize : tuple, optional
        The figure size. If None, the default as defined in ModelCode/GeneralSettings is used.
    subplots : boolean, optional
        Whether different output variables should be shown in separate figures
        or as subplots of the same figure.
    plt_file : str or None, optional
        If not None, the resluting figure(s) will be saved using this name.
        The default is None.
    foldername : str, optional
        The subfolder of Figures which to use (depends on the grouping type).
        The default is None.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to PandaToPlot_GetResults.

    Returns
    -------
    None.

    """
    if cols is None:
            cols = ["royalblue", "darkred", "grey", "gold"]
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
        
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    res = PandaToPlot_GetResultsMultScen(panda_file, output_var, "median", grouping_aim, adjacent, **kwargs)
    
    if output_var is str:
        output_var = [output_var]
    
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.35)
        rows = int(np.floor(np.sqrt(len(output_var))))
        cols = int(np.ceil(len(output_var)/rows))
    
    
    for idx, var in enumerate(output_var):
        if subplots:
            fig.add_subplot(rows, cols, idx + 1)
            plt.suptitle("Development depending on colaboration of clusters", \
                  fontsize = 24)
        else:
            fig = plt.figure(figsize = figsize)
            plt.title("Development depending on colaboration of clusters", \
                  fontsize = 24, pad = 15)
        for scen in range(0, len(res)):
            if res[scen] is None:
                continue
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Maximum"], color = cols[scen], marker = "^")
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Median"], color = cols[scen], marker = "X")
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Minimum"], color = cols[scen])
            plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
            plt.yticks(fontsize = 16)
            plt.xlabel("Number of different cluster groups", fontsize = 20)
            plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
            # plt.legend(fontsize = 20)
            if (not subplots) and (plt_file is not None):
                fig.savefig("Figures/" + foldername + "/PandaPlots/Median/" + plt_file + str(idx) + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    if subplots and (plt_file is not None):
        fig.savefig("Figures/" + foldername + "/PandaPlots/Median/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    if close_plots:
        plt.close()
        
    return(None)