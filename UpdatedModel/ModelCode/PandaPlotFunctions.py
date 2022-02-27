# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:55:08 2022

@author: leip
"""
import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
from textwrap import wrap

from ModelCode.PandaHandling import ReadFromPanda

# %% ############ PLOTTING FUNCTIONS USING RESULTS FROM PANDA CSV #############



def PlotPenaltyVsProb(panda_file = "current_panda", 
                      grouping_aim = "Dissimilar",
                      grouping_metric = "medoids",
                      adjacent = False,
                      figsize = None,
                      close_plots = None,
                      fn_suffix = None,
                      **kwargs):
    """
    Creates a plot of penalties vs. resulting probabilities using model runs 
    for all cluster groups of all group sizes for a specific grouping type and 
    specific model settings.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    figsize : tuple, optional
        The figure size. If None, the default as defined in ModelCode/GeneralSettings is used.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None.
    **kwargs : 
        Settings specifiying for which model run results shall be plotted.

    Returns
    -------
    None.

    """
    
    # settings
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
        
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
        
    if adjacent:
        add = "Adj"
    else:
        add = ""
    
    foldername = grouping_aim
    if adjacent:
        foldername = foldername + "Adjacent/"
    else:
        foldername = foldername + "NonAdjacent/"
    foldername = foldername + "/PandaPlots"
    
    plt_file = "PenaltiesProbabilities" + fn_suffix
    
    cols = ["royalblue", "darkred", "grey", "gold", "limegreen"]
    markers = ["o", "X", "^", "D", "s"]

    # plot penalties vs. probabilities for all cluster groups for all groupsizes
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for idx, size in enumerate([1, 2, 3, 5, 9]):
        with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric.capitalize() + 
                  "Size"  + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
    
        panda_tmp = ReadFromPanda(file = panda_file, \
                                  output_var = ['Penalty for food shortage', 
                                                'Penalty for insolvency',
                                                'Resulting probability for food security',
                                                'Resulting probability for solvency'], 
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        ax1.scatter(panda_tmp[['Penalty for food shortage']], 
                    panda_tmp[['Resulting probability for food security']],
                    color = cols[idx], marker = markers[idx])
        ax2.scatter(panda_tmp[['Penalty for insolvency']], 
                    panda_tmp[['Resulting probability for solvency']],
                    color = cols[idx], marker = markers[idx], 
                    label = str(size))
        
    # add axis, labels, etc.
    ax1.tick_params(labelsize = 14)
    ax2.tick_params(labelsize = 14)    
    ax1.set_xlabel(r"Penalty for food shortage $\rho_\mathrm{F}$ [$\$/10^3\,$kcal]", fontsize = 18)
    ax1.set_ylabel("Resulting probability for food security", fontsize = 18)
    ax2.set_xlabel(r"Penalty for insolvency $\rho_\mathrm{S}$ [$\$/\$$]", fontsize = 18)
    ax2.set_ylabel("Resulting probability for solvency", fontsize = 18)
    ax2.legend(title = "Groupsizes", fontsize = 16, title_fontsize = 18)
    plt.suptitle("Penalties and resulting probabilities (Aim: " + grouping_aim + \
                 ", Adjacent: " + str(adjacent) + ")", fontsize = 26)
         
    # save plot
    fig.savefig("Figures/" + foldername + "/Other/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)

    # close plot
    if close_plots:
        plt.close()  

    return(None)


def PlotProbDetVsSto(panda_file = "current_panda", 
                     grouping_aim = "Dissimilar",
                     grouping_metric = "medoids",
                     adjacent = False,
                     figsize = None,
                     close_plots = None,
                     fn_suffix = None,
                     **kwargs):
    """
    Creates a plot of probabilities resulting from stochastic solution vs. 
    probabilities resulting from deterministic solution for all cluster groups
    of all group sizes for a specific grouping type and specific model settings.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    figsize : tuple, optional
        The figure size. If None, the default as defined in ModelCode/GeneralSettings is used.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None.
    **kwargs : 
        Settings specifiying for which model run results shall be plotted.

    Returns
    -------
    None.

    """
    
    # settings
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
        
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
        
    if adjacent:
        add = "Adj"
    else:
        add = ""
    
    foldername = grouping_aim
    if adjacent:
        foldername = foldername + "Adjacent/"
    else:
        foldername = foldername + "NonAdjacent/"
    foldername = foldername + "/PandaPlots"
    
    plt_file = "ProbabilitiesDetVsSto" + fn_suffix
    
    cols = ["royalblue", "darkred", "grey", "gold", "limegreen"]
    markers = ["o", "X", "^", "D", "s"]

    # plot sto. probabilities vs det. probabilities
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for idx, size in enumerate([1, 2, 3, 5, 9]):
        with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric.capitalize() + 
                  "Size"  + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
    
        panda_tmp = ReadFromPanda(file = panda_file, \
                                  output_var = ['Resulting probability for food security for VSS', 
                                                'Resulting probability for solvency for VSS',
                                                'Resulting probability for food security',
                                                'Resulting probability for solvency'], 
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        ax1.scatter(panda_tmp[['Resulting probability for food security for VSS']], 
                    panda_tmp[['Resulting probability for food security']],
                    color = cols[idx], marker = markers[idx])
        ax2.scatter(panda_tmp[['Resulting probability for solvency for VSS']], 
                    panda_tmp[['Resulting probability for solvency']],
                    color = cols[idx], marker = markers[idx], 
                    label = str(size))
    
    # add axis, labels, etc.
    ax1.tick_params(labelsize = 14)
    ax2.tick_params(labelsize = 14)    
    ax1.set_xlim([-0.02, 1.02])    
    ax1.set_ylim([-0.02, 1.02])
    ax2.set_xlim([-0.02, 1.02])    
    ax2.set_ylim([-0.02, 1.02])
    ax1.set_xlabel(r"Resulting probability for food security (deterministic solution)", fontsize = 18)
    ax1.set_ylabel("Resulting probability for food security (stochastic solution)", fontsize = 18)
    ax2.set_xlabel(r"Resulting probability for solvency (deterministic solution)", fontsize = 18)
    ax2.set_ylabel("Resulting probability for solvency (stochastic solution)", fontsize = 18)
    ax2.legend(title = "Groupsizes", fontsize = 16, title_fontsize = 18)
    plt.suptitle("Resulting probabilities for deterministic and stochastic solution (Aim: " + grouping_aim + \
                 ", Adjacent: " + str(adjacent) + ")", fontsize = 26)
          
    # save plot
    fig.savefig("Figures/" + foldername + "/Other/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)

    # close plot
    if close_plots:
        plt.close()  

    return(None)


def PlotPandaSingle(panda_file = "current_panda", 
                    output_var = None,
                    scenarionames = None,
                    grouping_aim = "Dissimilar",
                    grouping_metric = "medoids",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_file = None,
                    foldername = None,
                    plt_legend = True,
                    close_plots = None,
                    **kwargs):
    """
    Combines the call to PlotPandaAll and PlotPandaMedian (as they are used
    for the same variable types and just are slightly different visualizations)

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : list of str or str
        The variables that are reported. The default is None.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
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
    plt_legend : boolean
        Whether legend should be plotted (in case with multiple scenarios).
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    **kwargs : 
        Settings specifiying for which mode

    Returns
    -------
    None.

    """
    
    PlotPandaMedian(panda_file = panda_file, 
                    output_var = output_var,
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    figsize = figsize,
                    subplots = subplots,
                    plt_file = plt_file,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    **kwargs)

    PlotPandaAll(panda_file = panda_file, 
                    output_var = output_var,
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    figsize = figsize,
                    subplots = subplots,
                    plt_file = plt_file,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    **kwargs)
    
    return(None)

def PlotPandaMedian(panda_file = "current_panda", 
                    output_var = None,
                    scenarionames = None,
                    grouping_aim = "Dissimilar",
                    grouping_metric = "medoids",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_file = None,
                    foldername = None,
                    plt_legend = True,
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
    scenarionames : list of str, optional
        Added as legend to describe the different scenarios.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
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
    plt_legend : boolean
        Whether legend should be plotted (in case with multiple scenarios).
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    cols : list, optional
        Colors to plot the different scenarios.        
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to _Panda_GetResults.

    Returns
    -------
    None.

    """
    
    
    # settings
    if cols is None:
            cols = ["royalblue", "darkred", "grey", "gold", "turquoise", "darkviolet"]
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
        
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    # get results
    res = Panda_GetResults(file = panda_file, 
                           output_var = output_var,
                           out_type = "median", 
                           grouping_aim = grouping_aim, 
                           grouping_metric = grouping_metric,
                           adjacent = adjacent, 
                           **kwargs)
    
    # make sure the output variable are given as list
    if output_var is str:
        output_var = [output_var]
    
    # set up suplots
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.35)
        num_rows = int(np.floor(np.sqrt(len(output_var))))
        num_cols = int(np.ceil(len(output_var)/num_rows))
    
    # plot each output variable
    for idx, var in enumerate(output_var):
        if subplots:
            fig.add_subplot(num_rows, num_cols, idx + 1)
            plt.suptitle("Development depending on colaboration of clusters", \
                  fontsize = 24)
        else:
            fig = plt.figure(figsize = figsize)
            plt.title("Development depending on colaboration of clusters", \
                  fontsize = 24, pad = 15)
        scatters = []
        for scen in range(0, len(res)):
            if res[scen] is None:
                if scenarionames is not None:
                    scenarionames.pop(scen)
                continue
            plt.fill_between(x = [1, 2, 3, 4, 5], y1 = np.array(res[scen][var + " - Maximum"], dtype = float), 
                                              y2 = np.array(res[scen][var + " - Minimum"], dtype = float), color = cols[scen], alpha = 0.3)
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Maximum"], alpha = 0.7, color = cols[scen], marker = "o")
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Median"], alpha = 1, color = cols[scen], marker = "X", label = "Median")
            plt.plot([1, 2, 3, 4, 5], res[scen][var + " - Mean"], alpha = 0.7, color = cols[scen], marker = "o", linestyle = "solid", label = "Mean")
            sc = plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Minimum"], alpha = 0.7, color = cols[scen])
            scatters.append(sc)
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        plt.legend()
        if (scenarionames is not None) and (plt_legend):
            plt.legend(scatters, scenarionames, fontsize = 18, title = "Scenarios", title_fontsize = 20)
        if (not subplots) and (plt_file is not None):
            fig.savefig("Figures/" + foldername + "/Median/" + plt_file + str(idx) + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    # save plot
    if subplots and (plt_file is not None):
        fig.savefig("Figures/" + foldername + "/Median/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    # close plot
    if close_plots:
        plt.close()
        
    return(None)


def PlotPandaAll(panda_file = "current_panda", 
                 output_var = None,
                 scenarionames = None,
                 grouping_aim = "Dissimilar",
                 grouping_metric = "medoids",
                 adjacent = False,
                 figsize = None,
                 subplots = True,
                 color = "blue",
                 plt_file = None,
                 foldername = None,
                 plt_legend = True,
                 close_plots = None,
                 cols = None,
                 **kwargs):
    """
    Creates plots visualizing the specified output variables for all cluster 
    groups within the grouping of each size for the given gorouping type.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : list of str or str
        The variables that are reported.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
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
    plt_legend : boolean
        Whether legend should be plotted (in case with multiple scenarios).
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    **kwargs : 
        Settings specifiying for which model run results shall be plotted.
    
    Returns
    -------
    None.

    """
    
    # settings
    if cols is None:
            cols = ["royalblue", "darkred", "grey", "gold", "turquoise", "darkviolet"]
            
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
        
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    # get results
    res = Panda_GetResults(file = panda_file, 
                           output_var = output_var,
                           out_type = "all", 
                           grouping_aim = grouping_aim, 
                           grouping_metric = grouping_metric,
                           adjacent = adjacent, 
                           **kwargs)
    
    # make sure the output variable are given as list
    if output_var is str:
        output_var = [output_var]
    
    # set up subplots
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.35)
        num_rows = int(np.floor(np.sqrt(len(output_var))))
        num_cols = int(np.ceil(len(output_var)/num_rows))
    
    # plot each output variable
    for idx, var in enumerate(output_var):
        if subplots:
            fig.add_subplot(num_rows, num_cols, idx + 1)
            plt.suptitle("Development depending on colaboration of clusters", \
                  fontsize = 24)
        else:
            fig = plt.figure(figsize = figsize)
            plt.title("Development depending on colaboration of clusters", \
                  fontsize = 24, pad = 15)
        scatters = []
        for scen in range(0, len(res)):
            if res[scen] is None:
                if scenarionames is not None:
                    scenarionames.pop(scen)
                continue
            plt.scatter(np.repeat(1, len(res[scen].loc[1, var])), res[scen].loc[1, var], alpha = 0.8, color = cols[scen])
            plt.scatter(np.repeat(2, len(res[scen].loc[2, var])), res[scen].loc[2, var], alpha = 0.8, color = cols[scen])
            plt.scatter(np.repeat(3, len(res[scen].loc[3, var])), res[scen].loc[3, var], alpha = 0.8, color = cols[scen])
            plt.scatter(np.repeat(4, len(res[scen].loc[5, var])), res[scen].loc[5, var], alpha = 0.8, color = cols[scen])
            sc = plt.scatter(np.repeat(5, len(res[scen].loc[9, var])), res[scen].loc[9, var], color = cols[scen])
            scatters.append(sc)
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        if (scenarionames is not None) and (plt_legend):
            plt.legend(scatters, scenarionames, fontsize = 18, title = "Scenarios", title_fontsize = 20)
        
    # save plot
    if plt_file is not None:
        fig.savefig("Figures/" + foldername + "/All/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
      
    # close plot
    if close_plots:
        plt.close()
        
    return(None)


def PlotPandaAggregate(panda_file = "current_panda", 
                    agg_type = "agg_sum", # or "agg_avgweight"
                    var_weight = None,
                    weight_title = None,
                    output_var = None,
                    scale_by = None, 
                    scenarionames = None,
                    scenarios_shaded = False,
                    grouping_aim = "Dissimilar",
                    grouping_metric = "medoids",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_legend = True,
                    subplot_titles = None,
                    ylabels = None,
                    plt_title = True,
                    plt_file = None,
                    foldername = None,
                    close_plots = None,
                    cols = None,
                    **kwargs):
    """
    Creates plots visualizing the specified output variables aggregated over
    all cluster groups within the grouping of each size for the given 
    gorouping type.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    agg_type: str
        Either "agg_sum" if values for different cluster groups should be 
        added up, or "agg_avgweiht" if a weighted average should be calculated.
    var_weight: str
        Name of variable that is used as weight. Only necessary for
        agg_type == "agg_avgweight".
    weight_title: str
        How the weight variable should be referred to in title .Only necessary 
        for agg_type == "agg_avgweight".
    output_var : list of str or str
        The variables that are reported.
    scale_by : float or str or list of str
        If float, all output variables will be scaled by this value. If str, all 
        output variables will be scaled by the sum of his scaling variable over
        all clusters (e.g. to scale by total demand). If list, each output
        variable can have a specific scaling variable/value (length of scale_by 
        must length of output_var). If None, no scaling is happening. The
        default is None.
    scenarionames : list of str, optional
        Added as legend to describe the different scenarios, and leads to plots
        being saved in /ComparingScenarios. If None, the folder according
        grouping_aim and adjacent is used. Default is None.
    scenarios_shaded : boolean, optional
        Whether area between two scenarios should be shaded. To be used if
        worst and best case scenario is to be compared for different settings.
        In this case, an even number of scenarios must be given, and always 
        two sequential scenarios are seen as a pair and the area between them
        will be shaded.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    figsize : tuple, optional
        The figure size. If None, the default as defined in ModelCode/GeneralSettings is used.
    subplots : boolean, optional
        Whether different output variables should be shown in separate figures
        or as subplots of the same figure.
    plt_legend : boolean
        Whether legend should be plotted (in case with multiple scenarios).
    subplot_titles : list of str or str of None
        Custom titles if the standard title shouldn't be used
    ylabels : list of str or str or None
        List of custom ylabels if output variables and units should not be used.
        The default is None.
    plt_title : boolean
        Whether the plot title should be added.
    plt_file : str or None, optional
        If not None, the resluting figure(s) will be saved using this name.
        The default is None.
    foldername : str, optional
        The subfolder of Figures which to use (depends on the grouping type).
        The default is None.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    cols : list of colors or None
        Colors to use. If None, default colors are used.
    **kwargs : 
        Settings specifiying for which model run results shall be plotted.

    Returns
    -------
    None.

    """
    
    
    # settings
    if cols is None:
        cols = ["royalblue", "darkred", "grey", "gold", "limegreen"]
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots

    if type(output_var) is str:
        output_var = [output_var]
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
      
    if (type(agg_type) is not list):
        if agg_type == "agg_sum":
            agg_title = " (aggregated by adding up)"
        if agg_type == "agg_avgweight":
            agg_title = " (aggregated by averaging with " + weight_title + " as weight)"
    else:
        agg_title = ""
    
    # make sure the output variable are given as list
    if output_var is str:
        output_var = [output_var]
        
    if (len(subplot_titles) != len(output_var)):
        sys.exit("Not the right number of subplot titles.")
        
    agg_title = []
    for idx, a_t in enumerate(agg_type):
        if a_t == "agg_sum":
            agg_title.append(" (aggregated by adding up)")
        else:
            agg_title.append(" (aggregated by averaging with " + weight_title[idx] + " as weight)")
                
                             
        
    # right number of subplot titles?
    if subplot_titles is not None:
        if type(subplot_titles) is not list:
            subplot_titles = [subplot_titles]
        if (len(subplot_titles) != len(output_var)):
            sys.exit("Not the right number of subplot titles.")
            
    
    # get results
    res = Panda_GetResults(file = panda_file, 
                           output_var = output_var,
                           out_type = agg_type, 
                           var_weight = var_weight,
                           grouping_aim = grouping_aim, 
                           grouping_metric = grouping_metric,
                           adjacent = adjacent, 
                           **kwargs)
    
    # get scaling variable
    if (type(scale_by) is list) and (len(scale_by) == 1):
        scale_by = scale_by[0]
        
    if (type(scale_by) is str) or ((type(scale_by) is list) and (type(scale_by[0]) is str)):
        scaling = Panda_GetResults(file = panda_file, 
                               output_var = scale_by,
                               out_type = "agg_sum", 
                               grouping_aim = grouping_aim, 
                               grouping_metric = grouping_metric,
                               adjacent = adjacent, 
                               **kwargs)
    
    # scale output variables
    plt_values = []
    for scen in range(0, len(res)):
        if scale_by is None:
            plt_values.append(res[scen])
        elif type(scale_by) is str:
            for var in output_var:
                res[scen][var + " - Aggregated over all groups"] = \
                    res[scen][var + " - Aggregated over all groups"]/ \
                        scaling[scen][scale_by + " - Aggregated over all groups"]
            plt_values.append(res[scen])
        elif type(scale_by) is list:
            if len(scale_by) != len(output_var):
                sys.exit("Not the right number of scaling variables")
            if type(scale_by[0]) is str:
                for idx, var in enumerate(output_var):
                    res[scen][var + " - Aggregated over all groups"] = \
                        res[scen][var + " - Aggregated over all groups"]/ \
                            scaling[scen][scale_by[idx] + " - Aggregated over all groups"]
                plt_values.append(res[scen])
            else:
                for idx, var in enumerate(output_var):
                    res[scen][var + " - Aggregated over all groups"] = \
                        res[scen][var + " - Aggregated over all groups"]/ \
                           scale_by[idx]
                plt_values.append(res[scen])                
        else:
            res[scen].iloc[:, 1:] = res[scen].iloc[:, 1:]/scale_by
            plt_values.append(res[scen])
    res = plt_values
        
    
    # set up subplots
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.25, hspace=0.3)
        num_cols = int(np.floor(np.sqrt(len(output_var))))
        num_rows = int(np.ceil(len(output_var)/num_cols))
        
    markers = ["X", "o", "^", "P", "s", "v"]    
    
    # plot each of the oubput variables
    if scenarios_shaded is False:
        for idx, var in enumerate(output_var):
            if subplots:
                fig.add_subplot(num_rows, num_cols, idx + 1)
                if plt_title is True:
                    if subplot_titles is not None:
                        plt.suptitle(subplot_titles[idx], fontsize = 26)
                    else:
                        plt.suptitle("Development depending on colaboration of clusters" + agg_title, \
                              fontsize = 26)
            else:
                fig = plt.figure(figsize = figsize)
                if plt_title is True:
                    plt.title("Development depending on colaboration of clusters" + agg_title, \
                          fontsize = 26, pad = 15)
            scatters = []
            mins = []
            maxs = []
            for scen in range(0, len(res)):
                if res[scen] is None:
                    if scenarionames is not None:
                        scenarionames.pop(scen)
                    continue
                plt.plot([1, 2, 3, 4, 5], res[scen][var + " - Aggregated over all groups"], lw = 2.8)
                sc = plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Aggregated over all groups"], marker = markers[scen], s = 70)
                scatters.append(sc)
                mins.append(min(-res[scen][var + " - Aggregated over all groups"].max()*0.01, res[scen][var + " - Aggregated over all groups"].min()*1.05))
                maxs.append(res[scen][var + " - Aggregated over all groups"].max()*1.1)
            plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
            plt.ylim((min(mins), max(maxs)))
            plt.yticks(fontsize = 16)
            plt.xlabel("Number of risk pools", fontsize = 20)
            if ylabels is not None:
                plt.ylabel("\n".join(wrap(ylabels[idx], width = 40)), fontsize = 20)
            else:
                plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        if (scenarionames is not None) and (plt_legend):
            ax = fig.add_subplot(1, 1, 1, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            ax.legend(scatters, scenarionames, fontsize = 18, title = "Scenarios",
                       title_fontsize = 20, bbox_to_anchor = (0.5, -0.15),
                       loc = "upper center")
        
    else:
        if (scenarionames is None) or (len(scenarionames)%3 != 0):
            sys.exit("Scenarionames not valid for option scenarios_shaded")
        for idx, var in enumerate(output_var):
            if subplots:
                fig.add_subplot(num_rows, num_cols, idx + 1)
                if plt_title is True:
                    if subplot_titles is not None:
                        plt.title("  " + subplot_titles[idx], fontsize = 24, loc = "left", pad = 15)
                    else:
                        plt.title("Development depending on colaboration of clusters" + agg_title, \
                              fontsize = 24, pad = 15)
            else:
                fig = plt.figure(figsize = figsize)
                if plt_title is True:
                    plt.title("Development depending on colaboration of clusters" + agg_title, \
                          fontsize = 24, pad = 15)
            lines = []
            mins = []
            maxs = []
            for scen in range(0, int(len(res)/3)):
                scen1 = 3 * scen
                scen2 = 3 * scen + 1
                scen3 = 3 * scen + 2
                if (res[scen1] is None) or (res[scen2] is None) or (res[scen3] is None):
                    res[scen1] = None
                    res[scen2] = None
                    res[scen3] = None
                    if scenarionames is not None:
                        scenarionames.pop(scen1)
                        scenarionames.pop(scen2)
                        scenarionames.pop(scen3)
                    continue
                l1, = plt.plot([1, 2, 3, 4, 5], res[scen1][var + " - Aggregated over all groups"], lw = 2.8, color = cols[scen], label = scenarionames[scen1])
                l2, = plt.plot([1, 2, 3, 4, 5], res[scen2][var + " - Aggregated over all groups"], lw = 2.8, color = cols[scen], linestyle = "dashdot", label = scenarionames[scen2])
                l3, = plt.plot([1, 2, 3, 4, 5], res[scen3][var + " - Aggregated over all groups"], lw = 2.8, color = cols[scen], linestyle = "--", label = scenarionames[scen3])
                plt.fill_between(x = [1, 2, 3, 4, 5], y1 = np.array(res[scen1][var + " - Aggregated over all groups"], dtype = float), 
                                                      y2 = np.array(res[scen3][var + " - Aggregated over all groups"], dtype = float), 
                                                      color = cols[scen], alpha = 0.3)
                lines = lines + [l1, l2, l3]
                
                mins.append(min(-res[scen1][var + " - Aggregated over all groups"].max()*0.01, res[scen1][var + " - Aggregated over all groups"].min()*1.05))
                mins.append(min(-res[scen2][var + " - Aggregated over all groups"].max()*0.01, res[scen2][var + " - Aggregated over all groups"].min()*1.05))
                mins.append(min(-res[scen3][var + " - Aggregated over all groups"].max()*0.01, res[scen3][var + " - Aggregated over all groups"].min()*1.05))
                maxs.append(res[scen1][var + " - Aggregated over all groups"].max()*1.1)
                maxs.append(res[scen2][var + " - Aggregated over all groups"].max()*1.1)
                maxs.append(res[scen3][var + " - Aggregated over all groups"].max()*1.1)
            plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 18)
            plt.ylim((min(mins), max(maxs)))
            plt.yticks(fontsize = 18)
            plt.xlabel("Number of risk pools", fontsize = 24)
            if ylabels is not None:
                plt.ylabel("\n".join(wrap(ylabels[idx], width = 30)), fontsize = 24)
            else:
                plt.ylabel("\n".join(wrap(var + " " + units[var], width = 30)), fontsize = 24)
        if (scenarionames is not None) and (plt_legend):
            ax = fig.add_subplot(1, 1, 1, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            ax.legend(handles = lines, fontsize = 18, title = "Scenarios",
                       title_fontsize = 20, bbox_to_anchor = (0.5, -0.1),
                       loc = "upper center")

    # save plot
    if plt_file is not None:
        fig.savefig("Figures/" + foldername  + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    
    # close plot
    if close_plots:
        plt.close()
    
    return(fig)


 # %% ########## FUNCTIONS TO GET THE RIGHT SUBSET OF PANDA RRESULTS ###########


def Panda_GetResults(file = "current_panda", 
                     output_var = None,
                     out_type = "agg_sum", # or agg_avgweight, or median, or all
                     var_weight = None,
                     grouping_aim = "Dissimilar",
                     grouping_metric = "medoids",
                     adjacent = False,
                     **kwargs):
    """
    This subsets and aggregates results for all cluster groups for all group
    sizes of specific grouping type, for multiple scenarios, i.e. model settings. 
    The setting over which to iterate must be given as a list. If iteration
    should go over multiple settings, they all must be defined as list of the
    same length (i.e. if two tax rates are to be combined with two risk levels,
    the four resulting combinations must be given by using lists of length
    four for tax rate and risk).

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : list of str or str
        The variables that are reported.
    out_type : str
        Specifies how the output variables should be aggregate over the different
        cluster groups. "agg_avgweight" for weighted average,
        "agg_sum" for summation, "median" for returning minimum, maximum
        and median over the cluster groups, "all" if result for all cluster
        groups should be kept.
    var_weight: str
        Name of variable that is used as weight .Only necessary for
        out_type == "agg_avgweight".
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    **kwargs :
        Settings specifiying for which model runs we want the plots

    Returns
    -------
    None.

    """
    
    # scenario dicts
    scenario_dict = kwargs.copy()
    scenario_dict["grouping_aim"] = grouping_aim
    scenario_dict["grouping_metric"] = grouping_metric
    scenario_dict["adjacent"] = adjacent

    # checking which of the scenario settings are lists
    l_scenario = []
    keys_list_scenario = []
    for key in scenario_dict.keys():
        if type(scenario_dict[key]) is list:
            l_scenario.append(len(scenario_dict[key]))
            keys_list_scenario.append(key)
            
    # checking if the scenario settings which should be iterated over have same length
    if (len(l_scenario) > 0) and (not all(ls == l_scenario[0] for ls in l_scenario)):
        sys.exit("All settings over which should be iterated must be " +
                     "lists of the same length!")
     
    # run _Panda_GetResultsSingScen for each setting combination
    if len(l_scenario) == 0:
        res = [Panda_GetResultsSingScen(output_var = output_var,
                                        out_type = out_type,
                                        var_weight = var_weight,
                                        **scenario_dict,
                                        file = file)]
    else:
        res = []
        for idx in range(0, l_scenario[0]):
            scenario_dict_tmp = scenario_dict.copy()
            for key in keys_list_scenario:
                scenario_dict_tmp[key] = scenario_dict[key][idx]
            try:
                res.append(Panda_GetResultsSingScen(output_var = output_var,
                                                     out_type = out_type,
                                                     var_weight = var_weight,
                                                    **scenario_dict_tmp,
                                                    file = file))
            except SystemExit:
                print("The " + str(idx + 1) + ". scenario is not available", flush = True)
                res.append(None)
                
    return(res)


def Panda_GetResultsSingScen(file = "current_panda", 
                           output_var = None,
                           out_type = "agg_sum", # or agg_avgweight, or median, or all
                           var_weight = None,
                           grouping_aim = "Dissimilar",
                           grouping_metric = "medoids",
                           adjacent = False,
                           sizes = [1, 2, 3, 5, 9],
                           **kwargs):
    """
    For given grouping type and model settings, this subsets the results and 
    aggregates them for all grouping sizes.

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : list of str or str
        The variables that are reported. The defaul is None.
    out_type : str
        Specifies how the output variables should be aggregate over the different
        cluster groups. "agg_avgweight" for weighted average,
        "agg_sum" for summation, "median" for returning minimum, maximum
        and median over the cluster groups, "all" if result for all cluster
        groups should be kept.
    var_weight: str
        Name of variable that is used as weight .Only necessary for
        out_type == "agg_avgweight".
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    sizes: list, optional
        List of group sizes for which results should be loaded.
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to ReadFromPandaSingleClusterGroup.

    Returns
    -------
    res : panda dataframe
        Dataframe of results prepared for each cluster grouping size.

    """
    
    # output variables dict
    output_dict = {"output_var" : output_var,
                   "out_type" : out_type,
                   "var_weight" : var_weight}
    
    # checking which of the output settings are lists
    l_output = []
    keys_list_output = []
    for key in output_dict.keys():
        if type(output_dict[key]) is list:
            l_output.append(len(output_dict[key]))
            keys_list_output.append(key)
    
    # checking if the output settings which should be iterated over have same length
    if (len(l_output) > 0) and (not all(ls == l_output[0] for ls in l_output)):
        sys.exit("All settings over which should be iterated must be " +
                     "lists of the same length!")
        
    add = ""
    if adjacent:
        add = "Adj"
       
    res = pd.DataFrame()
    
    if type(sizes) is not list:
        sizes = [sizes]
    
    # get results for each cluster grouping size
    for size in sizes:
        with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric.capitalize() + 
                  "Size"  + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
        
        panda_tmp = ReadFromPanda(file = file, \
                                  output_var = output_var, \
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        # run _ExtractResPanda for each output var
        if len(l_output) == 0:
            if out_type == "agg_avgweight":
                weight = ReadFromPanda(file = file, \
                                       output_var = var_weight, \
                                       k_using = BestGrouping, \
                                       **kwargs)
                tmp = _ExtractResPanda(sub_panda = panda_tmp, 
                                    out_type = out_type, 
                                    output_var = output_var,
                                    size = size, 
                                    weight = weight, 
                                    var_weight = var_weight)
            else:         
                tmp = _ExtractResPanda(sub_panda = panda_tmp, 
                                    out_type = out_type, 
                                    output_var = output_var,
                                    size = size)
        else:
            for idx in range(0, l_output[0]):
                output_dict_tmp = output_dict.copy()
                for key in keys_list_output:
                    output_dict_tmp[key] = output_dict[key][idx]
                    
                if output_dict_tmp["out_type"] == "agg_avgweight":
                    weight = ReadFromPanda(file = file, \
                                           output_var = output_dict_tmp["var_weight"], \
                                           k_using = BestGrouping, \
                                           **kwargs)
                    tmp2 = _ExtractResPanda(sub_panda = panda_tmp, 
                                        out_type = output_dict_tmp["out_type"], 
                                        output_var = output_dict_tmp["output_var"],
                                        size = size, 
                                        weight = weight, 
                                        var_weight = output_dict_tmp["var_weight"])
                else:         
                    tmp2 = _ExtractResPanda(sub_panda = panda_tmp, 
                                        out_type = output_dict_tmp["out_type"], 
                                        output_var = output_dict_tmp["output_var"],
                                        size = size)
                    
                if idx == 0:
                    tmp = tmp2
                else:
                    tmp = pd.concat([tmp, tmp2], axis = 1)    
                    
        res = res.append(tmp)
    
    return(res)


def _ExtractResPanda(sub_panda, out_type, output_var, size, weight = None, var_weight = None):
    """
    Aggregates results given by ReadFromPanda.

    Parameters
    ----------
    sub_panda : panda dataframe
        A subset of the panda dataframe for all groups of a specific cluster
        grouping.
    out_type : str
        Specifies how the output variables should be aggregate over the different
        cluster groups. "agg_avgweight" weighted average,
        "agg_sum" for summation, "median" for returning minimum, maximum
        and median over the cluster groups, "all" if result for all cluster
        groups should be kept.
    output_var : list of str or str
        The variables that are reported.
    size : int
        The group size for the grouping used.
    weight: panda dataframe
        Panda dataframe with variable that is to be used as weight.
        Only necessary for out_type == "agg_avgweight".
    var_weight: str
        Name of variable that is used as weight .Only necessary for
        out_type == "agg_avgweight".

    Returns
    -------
    res : panda dataframe
        the aggregated results

    """
    
    # make sure output variables are given as list
    if type(output_var) is str:
        output_var = [output_var]
        
    output_var_fct = output_var.copy()
    
    # average results of different clusters with population as weight
    if out_type == "agg_avgweight":
        output_var_fct.insert(0, "Group size")
        res = pd.DataFrame(columns = output_var_fct, index = [size])
        res.iloc[0,0] = size
        # calclate weights out of weight variable
        weight[var_weight] = weight[var_weight]/(weight[var_weight].sum())
        # weighted average
        for i in range(0, len(output_var)):
            res.iloc[0, i + 1] = (sub_panda[output_var[i]] * weight[var_weight]).sum()
        res = res.add_suffix(" - Aggregated over all groups")
        res.rename(columns = {"Group size - Aggregated over all groups": \
                              "Group size"}, inplace = True)
    
    
    # ... or sum up results of different clusters
    elif out_type == "agg_sum":
        output_var_fct.insert(0, "Group size")
        res = pd.DataFrame(columns = output_var_fct, index = [size])
        res.iloc[0,0] = size
        res.iloc[0,1:] = sub_panda[output_var].sum()
        res = res.add_suffix(" - Aggregated over all groups")
        res.rename(columns = {"Group size - Aggregated over all groups": \
                              "Group size"}, inplace = True)
    
    # ... or find min, max, and median
    elif out_type == "median":
        colnames = ["Group size"]
        for var in output_var_fct:
            colnames.append(var + " - Minimum")
            colnames.append(var + " - Median")
            colnames.append(var + " - Mean")
            colnames.append(var + " - Maximum")
        res = pd.DataFrame(columns = colnames, index = [size])
        res.iloc[0,0] = size
        for idx, var in enumerate(output_var_fct):
            res.iloc[0, idx*4 + 1] = sub_panda[var].min()
            res.iloc[0, idx*4 + 2] = sub_panda[var].median()
            res.iloc[0, idx*4 + 3] = sub_panda[var].mean()
            res.iloc[0, idx*4 + 4] = sub_panda[var].max()
    
    # ... or keep all cluster
    elif out_type == "all":
        output_var_fct.insert(0, "Group size")
        res = pd.DataFrame(columns = output_var_fct, index = [size])
        res.iloc[0,0] = size
        for idx, var in enumerate(output_var):
            res.iloc[0, idx + 1] = list(sub_panda[var])
            
    else:
        sys.exit("Invalid aggregation type! Please choose one of \"agg\", \"median\", or \"all\"")
            
    return(res)