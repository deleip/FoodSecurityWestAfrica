# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:28:30 2021

@author: Debbora Leip
"""
import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
from textwrap import wrap

from ModelCode.PandaHandling import ReadFromPanda
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import GetFilename

# %% ############### PLOTTING FUNCTIONS USING RESULTS PANDA CSV ###############

def _ExtractResPanda(sub_panda, out_type, output_var, size):
    """
    Aggregates results given by ReadFromPanda.

    Parameters
    ----------
    sub_panda : panda dataframe
        A subset of the panda dataframe for all groups of a specific cluster
        grouping.
    out_type : str
        Specifies how the output variables should be aggregate over the different
        cluster groups. "agg" for summation, "median" for returning minimum,
        maximum and median over the cluster groups, "all" if result for all
        cluster groups should be kept.
    output_var : list of str or str
        The variables that are reported.
    size : int
        The group size for the grouping used.

    Returns
    -------
    res : panda dataframe
        the aggregated results

    """
    
    # make sure output variables are given as list
    if type(output_var) is str:
        output_var = [output_var]
        
    output_var_fct = output_var.copy()
    
    # sum up results of different clusters
    if out_type == "agg":
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
            colnames.append(var + " - Maximum")
        res = pd.DataFrame(columns = colnames, index = [size])
        res.iloc[0,0] = size
        for idx, var in enumerate(output_var_fct):
            res.iloc[0, idx*3 + 1] = sub_panda[var].min()
            res.iloc[0, idx*3 + 2] = sub_panda[var].median()
            res.iloc[0, idx*3 + 3] = sub_panda[var].max()
    
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
    

def _Panda_GetResultsSingScen(file = "current_panda", 
                           output_var = None,
                           out_type = "agg", # or median, or all
                           grouping_aim = "Dissimilar",
                           adjacent = False,
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
        cluster groups. "agg" for summation, "median" for returning minimum,
        maximum and median over the cluster groups, "all" if result for all
        cluster groups should be kept.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
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
    
    # get results for each cluster grouping size
    for size in [1,2,3,5,9]:
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
    
        panda_tmp = ReadFromPanda(file = file, \
                                  output_var = output_var, \
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        res = res.append(_ExtractResPanda(panda_tmp, out_type, output_var, size))
            
    return(res)

def PlotPandaSingle(panda_file = "current_panda", 
                    output_var = None,
                    scenarionames = None,
                    grouping_aim = "Dissimilar",
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
        Settings specifiying for which mode

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
    
    plt_file = "PenaltiesProbabilities" + fn_suffix
    
    cols = ["royalblue", "darkred", "grey", "gold", "limegreen"]
    markers = ["o", "X", "^", "D", "s"]

    # plot penalties vs. probabilities for all cluster groups for all groupsizes
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for idx, size in enumerate([1, 2, 3, 5, 9]):
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                          + str(size) + grouping_aim + add + ".txt", "rb") as fp:
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
    fig.savefig("Figures/" + foldername + "PandaPlots/Other/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)

    # close plot
    if close_plots:
        plt.close()  

    return(None)


def PlotProbDetVsSto(panda_file = "current_panda", 
                     grouping_aim = "Dissimilar",
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
        Settings specifiying for which mode

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
    
    plt_file = "ProbabilitiesDetVsSto" + fn_suffix
    
    cols = ["royalblue", "darkred", "grey", "gold", "limegreen"]
    markers = ["o", "X", "^", "D", "s"]

    # plot sto. probabilities vs det. probabilities
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for idx, size in enumerate([1, 2, 3, 5, 9]):
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                          + str(size) + grouping_aim + add + ".txt", "rb") as fp:
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
    fig.savefig("Figures/" + foldername + "PandaPlots/Other/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)

    # close plot
    if close_plots:
        plt.close()  

    return(None)


def PandaPlotsCooperation(panda_file = "current_panda", 
                          scenarionames = None,
                          fn_suffix = None,
                          grouping_aim = "Dissimilar",
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
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
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
    
    def _report(i, console_output = console_output, num_plots = 9):
        if console_output:
            sys.stdout.write("\r     Plot " + str(i) + " of " + str(num_plots))
            
    # settings
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    if scenarionames is None:
        foldername = grouping_aim
        if adjacent:
            foldername = foldername + "Adjacent"
        else:
            foldername = foldername + "NonAdjacent"
    else:
        foldername = "ComparingScenarios"
        
    if fn_suffix is None:
        settingsIterate = DefaultSettingsExcept(**kwargs)
        settingsIterate["N"] = ""
        settingsIterate["validation_size"] = ""
        settingsIterate["k_using"] = ""
        fn_suffix = "_" + GetFilename(settingsIterate, groupSize = "", groupAim = grouping_aim, \
                          adjacent = adjacent)
        
    # plotting:
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average yearly total cultivated area', \
                                   'Average total cultivation costs'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "TotalAllocArea_TotalCultCosts" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    _report(1)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average necessary add. import excluding solvency constraint (over samples and then years)', \
                                   'Average necessary debt (excluding food security constraint)'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "NecImportsPen_NecDebtPen" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    _report(2)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average total necessary import (over samples and then years)', \
                                   'Average necessary debt'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "NecImports_NecDebt" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    _report(3)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Penalty for food shortage', \
                                'Penalty for insolvency'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "Penalties" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    _report(4)    

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security', \
                                'Resulting probability for solvency'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "ResProbabilities" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    _report(5)    

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average necessary add. import per capita (over samples and then years)', \
                                'Average necessary debt per capita (over all samples)'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "ShortcomingsCapita" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    _report(6)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average food demand penalty (over samples and then years)', \
                                   'Average solvency penalty (over samples)'],
                       scenarionames = scenarionames,
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "PenaltiesPaied" + fn_suffix,
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    _report(7)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Value of stochastic solution', \
                                'VSS as share of total costs (sto. solution)',\
                                'VSS as share of total costs (det. solution)'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "VSScosts" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    _report(8)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security for VSS',\
                                'Resulting probability for solvency for VSS'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "VSSprobabilities" + fn_suffix,
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    _report(9)    
    
    
    return(None)

def OtherPandaPlots(panda_file = "current_panda", 
                    grouping_aim = "Dissimilar",
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
      
    foldername = grouping_aim
    if adjacent:
        foldername = foldername + "Adjacent"
    else:
        foldername = foldername + "NonAdjacent"
        
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
                  adjacent = adjacent,
                  close_plots = close_plots,
                  fn_suffix = fn_suffix, 
                  **kwargs)
    
    # plot sto. probabilities vs. det. probabilities
    PlotProbDetVsSto(panda_file = panda_file, 
                     grouping_aim = grouping_aim,
                     adjacent = adjacent,
                     close_plots = close_plots,
                     fn_suffix = fn_suffix, 
                     **kwargs)
    return(None)

def Panda_GetResults(file = "current_panda", 
                                   output_var = None,
                                   out_type = "agg", # or median, or all
                                   grouping_aim = "Dissimilar",
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
        cluster groups. "agg" for summation, "median" for returning minimum,
        maximum and median over the cluster groups, "all" if result for all
        cluster groups should be kept.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
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
    fulldict["grouping_aim"] = grouping_aim
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
        res = [_Panda_GetResultsSingScen(output_var = output_var, **fulldict)]
    else:
        res = []
        for idx in range(0, l[0]):
            fulldict_tmp = fulldict.copy()
            for key in keys_list:
                fulldict_tmp[key] = fulldict[key][idx]
            try:
                res.append(_Panda_GetResultsSingScen(output_var = output_var, **fulldict_tmp))
            except SystemExit:
                res.append(None)
                
    return(res)

def PlotPandaMedian(panda_file = "current_panda", 
                    output_var = None,
                    scenarionames = None,
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
    scenarionames : list of str, optional
        Added as legend to describe the different scenarios.
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
    res = Panda_GetResults(panda_file, output_var, "median", grouping_aim, adjacent, **kwargs)
    
    # make sure the output variable are given as list
    if output_var is str:
        output_var = [output_var]
    
    # set up suplots
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.35)
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
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Maximum"], alpha = 0.8, color = cols[scen], marker = "^")
            plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Median"], alpha = 0.8, color = cols[scen], marker = "X")
            sc = plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Minimum"], alpha = 0.8, color = cols[scen])
            scatters.append(sc)
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        if scenarionames is not None:
            plt.legend(scatters, scenarionames, fontsize = 18, title = "Scenarios", title_fontsize = 20)
        if (not subplots) and (plt_file is not None):
            fig.savefig("Figures/" + foldername + "/PandaPlots/Median/" + plt_file + str(idx) + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    # save plot
    if subplots and (plt_file is not None):
        fig.savefig("Figures/" + foldername + "/PandaPlots/Median/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    # close plot
    if close_plots:
        plt.close()
        
    return(None)


def PlotPandaAll(panda_file = "current_panda", 
                 output_var = None,
                 scenarionames = None,
                 grouping_aim = "Dissimilar",
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
    res = Panda_GetResults(panda_file, output_var, "all", grouping_aim, adjacent, **kwargs)
    
    # make sure the output variable are given as list
    if output_var is str:
        output_var = [output_var]
    
    # set up subplots
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.35)
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
        fig.savefig("Figures/" + foldername + "/PandaPlots/All/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
      
    # close plot
    if close_plots:
        plt.close()
        
    return(None)


def PlotPandaAggregate(panda_file = "current_panda", 
                    output_var = None,
                    scenarionames = None,
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
    Creates plots visualizing the specified output variables aggregated over
    all cluster groups within the grouping of each size for the given 
    gorouping type.

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
        Settings specifiying for which mode

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
    
    # get results
    res = Panda_GetResults(panda_file, output_var, "agg", grouping_aim, adjacent, **kwargs)
    
    # make sure the output variable are given as list
    if output_var is str:
        output_var = [output_var]
    
    # set up subplots
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.35)
        num_rows = int(np.floor(np.sqrt(len(output_var))))
        num_cols = int(np.ceil(np.sqrt(len(output_var))))
    
    # plot each of the oubput variables
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
        mins = []
        maxs = []
        for scen in range(0, len(res)):
            if res[scen] is None:
                if scenarionames is not None:
                    scenarionames.pop(scen)
                continue
            sc = plt.scatter([1, 2, 3, 4, 5], res[scen][var + " - Aggregated over all groups"], marker = "X")
            scatters.append(sc)
            mins.append(min(-res[scen][var + " - Aggregated over all groups"].max()*0.01, res[scen][var + " - Aggregated over all groups"].min()*1.05))
            maxs.append(res[scen][var + " - Aggregated over all groups"].max()*1.1)
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.ylim((min(mins), max(maxs)))
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        if scenarionames is not None:
            plt.legend(scatters, scenarionames, fontsize = 18, title = "Scenarios", title_fontsize = 20)
        
    # save plot
    if plt_file is not None:
        fig.savefig("Figures/" + foldername + "/PandaPlots/Aggregated/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    
    # close plot
    if close_plots:
        plt.close()
    
    return(None)