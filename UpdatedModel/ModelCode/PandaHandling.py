# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:15:20 2021

@author: Debbora Leip
"""
import numpy as np
import pandas as pd
import os
import sys
import pickle
import matplotlib.pyplot as plt
from textwrap import wrap

from ModelCode.CompleteModelCall import LoadModelResults
from ModelCode.PandaGeneration import OpenPanda
from ModelCode.PandaGeneration import CreateEmptyPanda
from ModelCode.PandaGeneration import write_to_pandas
from ModelCode.PandaGeneration import SetUpPandaDicts

# %% ############## FUNCTIONS DEALING WITH THE RESULTS PANDA CSV ##############

def UpdatePandaWithAddInfo(OldFile = "current_panda", console_output = None):
    """
    If additional variables were included in write_to_pandas and the 
    dictionaries in SetUpPandaDicts were extended accordingly, this function
    rewrites the panda object (including the new variables) by loading the full
    model results (using "Filename for full results") row for row

    Parameters
    ----------
    OldFile : str, optional
        Name of the file containing the panda csv that needs to be updated.
        The default is "current_panda".
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.

    Returns
    -------
    None.

    """
    
    os.remove("ModelOutput/Pandas/ColumnUnits.txt")
    os.remove("ModelOutput/Pandas/ColumnNames.txt")
    os.remove("ModelOutput/Pandas/ColumnTypes.txt")
    SetUpPandaDicts()
    
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    # load the panda that should be updated
    oldPanda = OpenPanda(file = OldFile)
    
    # temporarily move the current_panda to tmp to be able to work on a
    # current_panda
    CreateEmptyPanda(OldFile + "_updating")
    
    for i in range(0, len(oldPanda)):
        if console_output:
            sys.stdout.write("\r     Updating row " + str(i + 1) + " of " + str(len(oldPanda)))
            
        filename = oldPanda.at[i,'Filename for full results']
        
        settings, args, AddInfo_CalcParameters, yield_information, \
        population_information, status, durations, crop_alloc, meta_sol, \
        crop_alloc_vs, meta_sol_vss, VSS_value, validation_values = \
            LoadModelResults(filename)
            
        write_to_pandas(settings, args, AddInfo_CalcParameters, yield_information, \
                        population_information, crop_alloc, \
                        meta_sol, meta_sol_vss, VSS_value, validation_values, \
                        filename, console_output = False, file = OldFile + "_updating")

    # remove old panda file
    os.remove("ModelOutput/Pandas/" + OldFile + ".csv")
    
    # rename _update file to OldFile
    os.rename("ModelOutput/Pandas/" + OldFile + "_updating.csv", "ModelOutput/Pandas/" + OldFile + ".csv")

    return(None)

def ReadFromPandaSingleClusterGroup(file = "current_panda", 
                                    output_var = None,
                                    probF = 0.99,
                                    probS = 0.95, 
                                    rhoF = None,
                                    rhoS = None,
                                    k = 9,     
                                    k_using = [3],
                                    yield_projection = "fixed",   
                                    sim_start = 2017,
                                    pop_scenario = "fixed",
                                    risk = 0.05,       
                                    tax = 0.01,       
                                    perc_guaranteed = 0.9,
                                    ini_fund = 0,            
                                    N = None, 
                                    validation_size = None,
                                    T = 20,
                                    seed = 201120):
    """
    Function returning a specific line (depending on the settings) of the 
    given panda csv for specific output variables. If N is not specified, it 
    uses the model run for the settings that used the highest sample size.

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : str or list of str
        A subset of the columns of the panda csv which should be returned.
    probF : float or None, optional
        demanded probability of keeping the food demand constraint. The default is 0.99.
    probS : float or None, optional
        demanded probability of keeping the solvency constraint. The default is 0.95.
    rhoF : float or None, optional
        The penalty for food shortages. The default is None.
    rhoS : float or None, optional
        The penalty for insolvency. The default is None.
    k : int, optional
        Number of clusters in which the area is devided. 
        The default is 9.
    k_using : list of int i\in{1,...,k}, optional
        Specifies the clusters considered in the model. 
        The default is the representative cluster [3].
    yield_projection : "fixed" or "trend", optional
        Specifies the yield projection used in the model. The default is "fixed".
    sim_start : int, optional
        The first year of the simulation. The default is 2017.
    pop_scenario : str, optional
        Specifies the population scenario used in the model.
        The default is "fixed".
    risk : int, optional
        The risk level that is covered by the government. The default is 5%.
    tax : float, optional
        Tax rate to be paied on farmers profits. The default is 1%.
    perc_guaranteed : float, optional
        The percentage that determines how high the guaranteed income is 
        depending on the expected income of farmers in a scenario excluding
        the government. The default is 90%.
    ini_fund : float
        Initial fund size. The default is 0.    
    N : int, optional
        Number of yield samples used to approximate the expected value
        in the original objective function. The default is 10000.
    validation_size : None or int, optional
        The sample size used for validation. The default is None.
    T : int, optional
        Number of years to cover in the simulation. The default is 20.
    seed : int, optional
        Seed used for yield generation. The default is 201120.
        
    Returns
    -------
    sub_panda : panda dataframe
        subset of the panda dataframe according to the settings and the 
        specified output variables.

    """
        
    if output_var is None:
        sys.exit("Please provide an output variable.")
    
    # open data frame
    panda = OpenPanda(file = file)
    
    # either settings sepcify the probabilites or the penalties
    if (probF is not None) and (probS is not None):
        panda = panda[:][list((panda.loc[:, "Input probability food security"] == probF) & \
                     (panda.loc[:, "Input probability solvency"] == probS))]
    elif (rhoF is not None) and (rhoS is not None):
        panda = panda[:][list((panda.loc[:, "Penalty for food shortage"] == rhoF) & \
                     (panda.loc[:, "Penalty for insolvency"] == rhoS))]
        
    
    # cannot compare with list over full column -> as string
    panda["Used clusters"] = panda["Used clusters"].apply(str)

    # make sure the output_variables are given as list
    if type(output_var) is str:
        output_var = [output_var]
    
    # subset the data frame according to the settings and the output_variables,
    # keeping sample size and sample size for validation ...
    output_var_fct = output_var.copy()
    output_var_fct.insert(0, "Used clusters")
    tmp = output_var_fct.copy()
    tmp.append("Sample size")
    tmp.append("Sample size for validation")
    sub_panda = panda[tmp]\
                    [list((panda.loc[:, "Number of clusters"] == k) & \
                     (panda.loc[:, "Used clusters"] == str(k_using)) & \
                     (panda.loc[:, "Yield projection"] == yield_projection) & \
                     (panda.loc[:, "Simulation start"] == sim_start) & \
                     (panda.loc[:, "Population scenario"] == pop_scenario) & \
                     (panda.loc[:, "Risk level covered"] == risk) & \
                     (panda.loc[:, "Tax rate"] == tax) & \
                     (panda.loc[:, "Share of income that is guaranteed"] == perc_guaranteed) & \
                     (panda.loc[:, "Initial fund size"] == ini_fund) & \
                     (panda.loc[:, "Number of covered years"] == T))]
                  
    # no results for these settings?
    if sub_panda.empty:
        sys.exit("Requested data is not available.")
        
    # ... subset sample size here if given ...
    if N is not None:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size"] == N]
        # nor results for right sample size?
        if sub_panda.empty:
            sys.exit("Requested data is not available.")
        return(sub_panda)
        
    # ... else use the results for highest sample size for these settings
    sub_panda = sub_panda[sub_panda["Sample size"] == max(sub_panda["Sample size"])]
    # if multiple runs for highest sample size, find highest validation sample size
    if (len(sub_panda) > 1) and (validation_size is None):
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size for validation"] == \
                                          max(sub_panda["Sample size for validation"])]
    elif validation_size is not None:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size for validation"] == \
                                                              validation_size]
        if sub_panda.empty:
            sys.exit("Requested data is not available.")
    else:
        sub_panda = sub_panda[output_var_fct]
                       
    
    # turn used clusters back from strings to lists
    def __ConvertListsInts(arg):
        arg = arg.strip("][").split(", ")
        res = []
        for j in range(0, len(arg)):
            res.append(int(arg[j]))
        return(res)

    sub_panda["Used clusters"] = sub_panda["Used clusters"].apply(__ConvertListsInts)
        
    return(sub_panda)
    
def ReadFromPanda(file = "current_panda", 
                  output_var = None,
                  k_using = [3],
                  **kwargs):
    """
    Function returning a subset of the given panda dataframe, according to the
    cluster groups, settings, and output variables specified

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : str or list of str
        A subset of the columns of the panda csv which should be returned.
    k_using : tuple of ints, list of tuples of ints, int, list of ints, optional
        Either one group of clusters or a set of different cluster groups for
        which results shall be returned. The default is [3].
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to ReadFromPandaSingleClusterGroup.

    Returns
    -------
    subpanda : panda dataframe
        subset of the given panda dataframe, according to the cluster groups, 
        settings, and output variables specified

    """
    
    # check given output_variables
    if output_var is None:
        sys.exit("Please probide an output variable.")
    elif type(output_var) is str:
        output_var = [output_var]
        
    # prepare cluster groups
    if type(k_using) is tuple:
       k_using = [sorted(list(k_using))]
    elif (type(k_using) is list) and (type(k_using[0]) is not int):
        k_using = [sorted(list(k_using_tmp)) for k_using_tmp in k_using]
    elif type(k_using) is int:
        k_using = [[k_using]]
    else:
        k_using = [sorted(k_using)]
    
    # set up panda cluster group per cluster grop
    sub_panda = pd.DataFrame()
    for k_using_tmp in k_using:
        sub_panda = sub_panda.append(ReadFromPandaSingleClusterGroup(file = file, \
                                                        output_var = output_var, \
                                                        k_using = k_using_tmp, \
                                                        **kwargs))
        
    return(sub_panda)
            

def __ExtractResPanda(sub_panda, out_type, output_var, size):
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

    if type(output_var) is str:
        output_var = [output_var]
        
    output_var_fct = output_var.copy()
    
    if out_type == "agg":
        output_var_fct.insert(0, "Group size")
        res = pd.DataFrame(columns = output_var_fct, index = [size])
        res.iloc[0,0] = size
        res.iloc[0,1:] = sub_panda[output_var].sum()
        res = res.add_suffix(" - Aggregated over all groups")
        res.rename(columns = {"Group size - Aggregated over all groups": \
                              "Group size"}, inplace = True)
    
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
    
    elif out_type == "all":
        output_var_fct.insert(0, "Group size")
        res = pd.DataFrame(columns = output_var_fct, index = [size])
        res.iloc[0,0] = size
        for idx, var in enumerate(output_var):
            res.iloc[0, idx + 1] = list(sub_panda[var])
            
    return(res)
    

def PandaToPlot_GetResults(file = "current_panda", 
                           output_var = None,
                           out_type = "agg", # or median, or all
                           grouping_aim = "Dissimilar",
                           adjacent = False,
                           **kwargs):
    """
    For a given grouping type, this subsets the results and aggregates them
    for all grouping sizes.

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
    
    for size in [1,2,3,5,9]:
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
    
        panda_tmp = ReadFromPanda(file = file, \
                                  output_var = output_var, \
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        res = res.append(__ExtractResPanda(panda_tmp, out_type, output_var, size))
            
    return(res)

def PlotPandaMedian(panda_file = "current_panda", 
                    output_var = None,
                    grouping_aim = "Dissimilar",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_file = None,
                    foldername = None,
                    close_plots = None,
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
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
        
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    res = PandaToPlot_GetResults(panda_file, output_var, "median", grouping_aim, adjacent, **kwargs)
    
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
        plt.scatter([1, 2, 3, 4, 5], res[var + " - Maximum"], marker = "^", label = "Maximum")
        plt.scatter([1, 2, 3, 4, 5], res[var + " - Median"], marker = "X", label = "Median")
        plt.scatter([1, 2, 3, 4, 5], res[var + " - Minimum"], label = "Minimum")
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        plt.legend(fontsize = 20)
        if (not subplots) and (plt_file is not None):
            fig.savefig("Figures/" + foldername + "/PandaPlots/Median/" + plt_file + str(idx) + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    if subplots and (plt_file is not None):
        fig.savefig("Figures/" + foldername + "/PandaPlots/Median/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    if close_plots:
        plt.close()
        
    return(None)

def PlotPandaAll(panda_file = "current_panda", 
                 output_var = None,
                 grouping_aim = "Dissimilar",
                 adjacent = False,
                 figsize = None,
                 subplots = True,
                 plt_file = None,
                 foldername = None,
                 close_plots = None,
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
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
        
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    res = PandaToPlot_GetResults(panda_file, output_var, "all", grouping_aim, adjacent, **kwargs)
    
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
        plt.scatter(np.repeat(1, len(res.loc[1, var])), res.loc[1, var])
        plt.scatter(np.repeat(2, len(res.loc[2, var])), res.loc[2, var])
        plt.scatter(np.repeat(3, len(res.loc[3, var])), res.loc[3, var])
        plt.scatter(np.repeat(4, len(res.loc[5, var])), res.loc[5, var])
        plt.scatter(np.repeat(5, len(res.loc[9, var])), res.loc[9, var])
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        # plt.legend(fontsize = 20)
        
    if plt_file is not None:
        fig.savefig("Figures/" + foldername + "/PandaPlots/All/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
      
    if close_plots:
        plt.close()
        
    return(None)


def PlotPandaAggregate(panda_file = "current_panda", 
                    output_var = None,
                    grouping_aim = "Dissimilar",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_file = None,
                    foldername = None,
                    close_plots = None,
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
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots

    if type(output_var) is str:
        output_var = [output_var]
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    res = PandaToPlot_GetResults(panda_file, output_var, "agg", grouping_aim, adjacent, **kwargs)
    
    if output_var is str:
        output_var = [output_var]
    
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.35)
        rows = int(np.floor(np.sqrt(len(output_var))))
        cols = int(np.ceil(np.sqrt(len(output_var))))
    
    for idx, var in enumerate(output_var):
        if subplots:
            fig.add_subplot(rows, cols, idx + 1)
            plt.suptitle("Development depending on colaboration of clusters", \
                  fontsize = 24)
        else:
            fig = plt.figure(figsize = figsize)
            plt.title("Development depending on colaboration of clusters", \
                  fontsize = 24, pad = 15)
        plt.scatter([1, 2, 3, 4, 5], res[var + " - Aggregated over all groups"], marker = "X")
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        min_y = min(-res[var + " - Aggregated over all groups"].max()*0.01, res[var + " - Aggregated over all groups"].min()*1.05)
        plt.ylim((min_y, res[var + " - Aggregated over all groups"].max()*1.1))
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel("\n".join(wrap(var + " " + units[var], width = 50)), fontsize = 20)
        
    if plt_file is not None:
        fig.savefig("Figures/" + foldername + "/PandaPlots/Aggregated/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    
    if close_plots:
        plt.close()
    
    return(None)

def PlotPandaSingle(panda_file = "current_panda", 
                    output_var = None,
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
    
    PlotPandaMedian(panda_file = panda_file, 
                    output_var = output_var,
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
                      **kwargs):
    """
    Creates a plot of penalties vs. resulting probabilities using model runs 
    for all cluster groups for a specific grouping type and specific model 
    settings.

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
    **kwargs : 
        Settings specifiying for which mode

    Returns
    -------
    None.

    """
    
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
    
    plt_file = grouping_aim + add + "_PenaltiesProbabilities"
    
    cols = ["royalblue", "darkred", "grey", "gold", "limegreen"]
    markers = ["o", "X", "^", "D", "s"]

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
        
    ax1.tick_params(labelsize = 14)
    ax2.tick_params(labelsize = 14)    
    ax1.set_xlabel(r"Penalty for food shortage $\rho_\mathrm{F}$ [$\$/10^3\,$kcal]", fontsize = 18)
    ax1.set_ylabel("Resulting probability for food security", fontsize = 18)
    ax2.set_xlabel(r"Penalty for insolvency $\rho_\mathrm{S}$ [$\$/\$$]", fontsize = 18)
    ax2.set_ylabel("Resulting probability for solvencyy", fontsize = 18)
    ax2.legend(title = "Groupsizes", fontsize = 16, title_fontsize = 18)
    plt.suptitle("Penalties and resulting probabilities (Aim: " + grouping_aim + \
                 ", Adjacent: " + str(adjacent) + ")", fontsize = 26)
            
    fig.savefig("Figures/" + foldername + "PandaPlots/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)

    if close_plots:
        plt.close()  

    return(None)

def MainPandaPlotsFixedSettings(panda_file = "current_panda", 
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
        Settings specifiying for which mode
    
    Returns
    -------
    None.

    """
    
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    def __report(i, console_output = console_output, num_plots = 13):
        if console_output:
            sys.stdout.write("\r     Plot " + str(i) + " of " + str(num_plots))
      
    foldername = grouping_aim
    if adjacent:
        foldername = foldername + "Adjacent"
    else:
        foldername = foldername + "NonAdjacent"
        
    if adjacent:
        add = "Adj"
    else:
        add = ""
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average yearly total cultivated area', \
                                   'Average total cultivation costs'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = grouping_aim + add + "_TotalAllocArea_TotalCultCosts",
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    __report(1)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Necessary add. import (excluding solvency constraint, including theoretical export)', \
                                   'Necessary debt (excluding food security constraint)'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = grouping_aim + add + "_NecImportsPen_NecDebtPen",
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    __report(2)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Total necessary import when including solvency constraint', \
                                   'Necessary debt (including food security constraint)'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = grouping_aim + add + "_NecImports_NecDebt",
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    __report(3)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Penalty for food shortage', \
                                'Penalty for insolvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_Penalties",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(4)    

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security', \
                                'Resulting probability for solvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_ResProbabilities",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(5)    

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average food shortcomings (over all years and samples with shortcomings)', \
                                'Average food shortcomings per capita (over all years and samples with shortcomings)'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_FoodShortcomings",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(6)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average final fund (over all samples with negative final fund)',
                                'Averge final fund (over all samples with negative final fund) scaled with probability of insolvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_FinalFund",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(7)    
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Averge final fund per capita (over all samples with negative final fund)',
                                'Averge final fund per capita (over all samples with negative final fund) scaled with probability of insolvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_FinalFundPerCapita",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(8)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Averge final fund as share of guaranteed income (over all samples with negative final fund)',
                                'Averge final fund as share of guaranteed income (over all samples with negative final fund) scaled with probability of insolvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_FinalFundShareGovInc",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(9)    
    
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average food demand penalty (over years and samples)', \
                                   'Average solvency penalty (over samples)'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = grouping_aim + add + "_PenaltiesPaied",
                       foldername = foldername,
                       close_plots = close_plots,
                       **kwargs)
    __report(10)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Value of stochastic solution', \
                                'VSS as share of total costs (sto. solution)',\
                                'VSS as share of total costs (det. solution)'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_VSScosts",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(11)    
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security for VSS',\
                                'Resulting probability for solvency for VSS'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = grouping_aim + add + "_VSSprobabilities",
                    foldername = foldername,
                    close_plots = close_plots,
                    **kwargs)
    __report(12)    
        
    PlotPenaltyVsProb(panda_file = panda_file, 
                      grouping_aim = grouping_aim,
                      adjacent = adjacent,
                      close_plots = close_plots,
                      **kwargs)
    __report(13)    
    
    return(None)
