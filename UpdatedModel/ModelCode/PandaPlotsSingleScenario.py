# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:28:30 2021

@author: Debbora Leip
"""
import numpy as np
import pandas as pd
import sys
import pickle
import os 
import matplotlib.pyplot as plt
from textwrap import wrap

from ModelCode.PandaHandling import ReadFromPanda
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import GetFilename
from ModelCode.SetFolderStructure import _GroupingPlotFolders

# %% ############### PLOTTING FUNCTIONS USING RESULTS PANDA CSV ###############

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
    
    add = ""
    if adjacent:
        add = "Adj"
       
    res = pd.DataFrame()
    
    if type(sizes) is not list:
        sizes = [sizes]
    
    # get results for each cluster grouping size
    for size in sizes:
        with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric + 
                  "Size"  + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
    
        panda_tmp = ReadFromPanda(file = file, \
                                  output_var = output_var, \
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        if out_type == "agg_avgweight":
            weight = ReadFromPanda(file = file, \
                                   output_var = var_weight, \
                                   k_using = BestGrouping, \
                                   **kwargs)
            res = res.append(_ExtractResPanda(sub_panda = panda_tmp, 
                                              out_type = out_type, 
                                              output_var = output_var,
                                              size = size, 
                                              weight = weight, 
                                              var_weight = var_weight))
        else:         
            res = res.append(_ExtractResPanda(sub_panda = panda_tmp, 
                                              out_type = out_type, 
                                              output_var = output_var,
                                              size = size))
            
    return(res)

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
                    close_plots = close_plots,
                    **kwargs)
    
    return(None)

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
        with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric + 
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
        with open("InputData/Clusters/ClusterGroups/Grouping" + grouping_metric + 
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


def PandaPlotsCooperation(panda_file = "current_panda", 
                          scenarionames = None,
                          folder_comparisons = "unnamed",
                          fn_suffix = None,
                          grouping_aim = "Dissimilar",
                          grouping_metric = "medoids",
                          adjacent = False,
                          close_plots = None,
                          console_output = None,
                          **kwargs):
    """
    
    Calls plotting functions for all main output variables such that all main 
    output plots can be created and saved with one function call for fixed
    model settings.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    scenarionames : list of str, optional
        Added as legend to describe the different scenarios, and leads to plots
        being saved in /ComparingScenarios. If None, the folder according
        grouping_aim and adjacent is used. Default is None.
    folder_comparisons: str
        Subfolder of /ComparingScenarios (i.e. for example
        ComparisonPlots/folder_comparison/AggregatedSum/NecImport.png).
        Only relevant if scenarionames is not None.
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None. Default is "unnamed".
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.        
    **kwargs : 
        Settings specifiying for which model runs we want the plots
    
    Returns
    -------
    None.

    """
    # settings
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    if scenarionames is None:
        foldername = grouping_aim
        if adjacent:
            foldername = foldername + "Adjacent"
        else:
            foldername = foldername + "NonAdjacent"
        foldername = foldername + "/PandaPlots"
    else:
        _GroupingPlotFolders(main = "ComparingScenarios/" + folder_comparisons, a = False)
        foldername = "ComparingScenarios/" + folder_comparisons
        
    if fn_suffix is None:
        settingsIterate = DefaultSettingsExcept(**kwargs)
        settingsIterate["N"] = ""
        settingsIterate["validation_size"] = ""
        settingsIterate["k_using"] = ""
        fn_suffix = "_" + GetFilename(settingsIterate, groupSize = "", groupAim = grouping_aim, \
                          adjacent = adjacent)
            
            
    def _report(i, console_output = console_output, num_plots = 17):
        if console_output:
            sys.stdout.write("\r     Plot " + str(i) + " of " + str(num_plots))
        return(i + 1)
    i = 1         
    
    # plotting:
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Average yearly total cultivated area', \
                                   'Total cultivation costs (sto. solution)'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "TotalAllocArea_TotalCultCosts" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=["Average total cultivation costs", \
                                   "Average total food demand penalty (over samples)", \
                                   "Average solvency penalty (over samples)"],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "CultivationAndSocialCosts" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
    
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum", 
                       output_var=['Average aggregate food shortage excluding solvency constraint', \
                                   'Average aggregate debt after payout (excluding food security constraint)'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "NecImportsPen_NecDebtPen" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Average aggregate food shortage', \
                                   'Average aggregate debt after payout'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "NecImports_NecDebt" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Penalty for food shortage', \
                                'Penalty for insolvency'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "Penalties" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)    

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security', \
                                'Resulting probability for solvency'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "ResProbabilities" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)    

    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Resulting probability for food security', \
                                   'Resulting probability for solvency'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "ResProbabilities" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average aggregate food shortage per capita', \
                                'Average aggregate debt after payout per capita'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "ShortcomingsCapita" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)  
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average aggregate food shortage per capita (including only samples that have shortage)', \
                                'Average aggregate debt after payout per capita (including only samples with negative final fund)'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "ShortcomingsOnlyWhenNeededCapita" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)  
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average aggregate food shortage (without taking into account imports)', \
                                'Average aggregate debt after payout'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "Shortcomings" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i) 
    
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Average aggregate food shortage per capita', \
                                   'Average aggregate debt after payout per capita'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "ShortcomingsCapita" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)   
        
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum", 
                       output_var=['Average food demand penalty (over samples and then years)', \
                                   'Average solvency penalty (over samples)'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "PenaltiesPaied" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Value of stochastic solution', \
                                'VSS as share of total costs (sto. solution)',\
                                'VSS as share of total costs (det. solution)'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "VSScosts" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)    
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['VSS in terms of avg. nec. debt', \
                                'VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution',\
                                'VSS in terms of avg. nec. debt as share of avg. nec. debt of sto. solution'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "VSSdebt" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)  
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['VSS in terms of avg. nec. import', \
                                'VSS in terms of avg. nec. import as share of avg. nec. import of det. solution',\
                                'VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "VSSimport" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)  
    
    
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum", 
                       output_var=['Value of stochastic solution', \
                                   'VSS in terms of avg. nec. debt', \
                                   'VSS in terms of avg. nec. import'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "VSSagg" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    i = _report(i)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security for VSS',\
                                'Resulting probability for solvency for VSS'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "VSSprobabilities" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    i = _report(i)    
    
    
    return(None)

def OtherPandaPlots(panda_file = "current_panda", 
                    grouping_aim = "Dissimilar",
                    grouping_metric = "medoids",
                    adjacent = False,
                    close_plots = None,
                    console_output = None,
                    fn_suffix = None,
                    **kwargs):
    """
    Creates some additional plots (that don't fit into the structure of 
    PandaPlotsCooperation): PlotPenaltyVsProb and PlotProbDetVsSto

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
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.        
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None.
    **kwargs : 
        Settings specifiying for which model runs we want the plots 

    Returns
    -------
    None.

    """
    
    # settings
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
      
    # foldername = grouping_aim
    # if adjacent:
    #     foldername = foldername + "Adjacent"
    # else:
    #     foldername = foldername + "NonAdjacent"
    # foldername = foldername + "/PandaPlots"
        
    if fn_suffix is None:
        settingsIterate = DefaultSettingsExcept(**kwargs)
        settingsIterate["N"] = ""
        settingsIterate["validation_size"] = ""
        settingsIterate["k_using"] = ""
        fn_suffix = "_" + GetFilename(settingsIterate, groupSize = "", groupAim = grouping_aim, \
                          adjacent = adjacent)
            
    # plot penalties vs. probabilities
    PlotPenaltyVsProb(panda_file = panda_file, 
                  grouping_aim = grouping_aim,
                  grouping_metric = grouping_metric,
                  adjacent = adjacent,
                  close_plots = close_plots,
                  fn_suffix = fn_suffix, 
                  **kwargs)
    
    # plot sto. probabilities vs. det. probabilities
    PlotProbDetVsSto(panda_file = panda_file, 
                     grouping_aim = grouping_aim,
                     grouping_metric = grouping_metric,
                     adjacent = adjacent,
                     close_plots = close_plots,
                     fn_suffix = fn_suffix, 
                     **kwargs)
    return(None)

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
    
    # adding settings to dict
    fulldict = kwargs.copy()
    fulldict["file"] = file
    fulldict["out_type"] = out_type
    fulldict["var_weight"] = var_weight
    fulldict["grouping_aim"] = grouping_aim
    fulldict["grouping_metric"] = grouping_metric
    fulldict["adjacent"] = adjacent

    # checking which of the settings are lists
    l = []
    keys_list = []
    for key in fulldict.keys():
        if type(fulldict[key]) is list:
            l.append(len(fulldict[key]))
            keys_list.append(key)
     
    # checking if the settings which should be iterated over have same length
    if (len(l) > 0) and (not all(ls == l[0] for ls in l)):
        sys.exit("All settings over which should be iterated must be " +
                     "lists of the same length!")
     
    # run _Panda_GetResultsSingScen for each setting combination
    if len(l) == 0:
        res = [Panda_GetResultsSingScen(output_var = output_var, **fulldict)]
    else:
        res = []
        for idx in range(0, l[0]):
            fulldict_tmp = fulldict.copy()
            for key in keys_list:
                fulldict_tmp[key] = fulldict[key][idx]
            try:
                res.append(Panda_GetResultsSingScen(output_var = output_var, **fulldict_tmp))
            except SystemExit:
                print("The " + str(idx + 1) + ". scenario is not available", flush = True)
                res.append(None)
                
    return(res)

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
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to _Panda_GetResults.

    Returns
    -------
    None.

    """
    
    # settings
    if cols is None:
            cols = ["royalblue", "darkred", "grey", "gold"]
    
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
        if scenarionames is not None:
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
            cols = ["royalblue", "darkred", "grey", "gold"]
            
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
                           grounping_metric = grouping_metric,
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
        if scenarionames is not None:
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
                    scenarionames = None,
                    grouping_aim = "Dissimilar",
                    grouping_metric = "medoids",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
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
            cols = ["royalblue", "darkred", "grey", "gold"]
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots

    if type(output_var) is str:
        output_var = [output_var]
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
        
    if agg_type == "agg_sum":
        agg_title = " (aggregated by adding up)"
        agg_folder = "/AggregatedSum/"
    if agg_type == "agg_avgweight":
        agg_title = " (aggregated by averaging with " + weight_title + " as weight)"
        agg_folder = "/AggregatedWeightedAvg/"
    
    # get results
    res = Panda_GetResults(file = panda_file, 
                           output_var = output_var,
                           out_type = agg_type, 
                           var_weight = var_weight,
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
        
    markers = ["X", "o", "^", "P", "s", "v"]    
    
    # plot each of the oubput variables
    for idx, var in enumerate(output_var):
        if subplots:
            fig.add_subplot(num_rows, num_cols, idx + 1)
            plt.suptitle("Development depending on colaboration of clusters" + agg_title, \
                  fontsize = 24)
        else:
            fig = plt.figure(figsize = figsize)
            plt.title("Development depending on colaboration of clusters" + agg_title, \
                  fontsize = 24, pad = 15)
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
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        if scenarionames is not None:
            plt.legend(scatters, scenarionames, fontsize = 18, title = "Scenarios",
                       title_fontsize = 20, loc = "best")
        

    # save plot
    if plt_file is not None:
        fig.savefig("Figures/" + foldername + agg_folder + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    
    # close plot
    if close_plots:
        plt.close()
    
    return(None)