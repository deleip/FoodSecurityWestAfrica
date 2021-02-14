# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:15:20 2021

@author: leip
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
    CreateEmptyPanda(OldFile + "_updated")
    
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
                        filename, console_output = False, file = OldFile + "_updated")

    # remove old panda file
    os.remove("ModelOutput/Pandas/" + OldFile + ".csv")
    
    # rename _update file to OldFile
    os.rename("ModelOutput/Pandas/" + OldFile + "_updated.csv", "ModelOutput/Pandas/" + OldFile + ".csv")

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
        
    if output_var is None:
        sys.exit("Please probide an output variable.")
    
    panda = OpenPanda(file = file)
    # cannot compare with list over full column -> as string
    panda["Used clusters"] = panda["Used clusters"].apply(str)

    
    if type(output_var) is str:
        output_var = [output_var]
    
    output_var_fct = output_var.copy()
    output_var_fct.insert(0, "Used clusters")
    tmp = output_var_fct.copy()
    tmp.append("Sample size")
    tmp.append("Sample size for validation")
    sub_panda = panda[tmp]\
                    [list((panda.loc[:, "Input probability food security"] == probF) & \
                     (panda.loc[:, "Input probability solvency"] == probS) & \
                     (panda.loc[:, "Number of clusters"] == k) & \
                     (panda.loc[:, "Used clusters"] == str(k_using)) & \
                     (panda.loc[:, "Yield projection"] == yield_projection) & \
                     (panda.loc[:, "Simulation start"] == sim_start) & \
                     (panda.loc[:, "Population scenario"] == pop_scenario) & \
                     (panda.loc[:, "Risk level covered"] == risk) & \
                     (panda.loc[:, "Tax rate"] == tax) & \
                     (panda.loc[:, "Share of income that is guaranteed"] == perc_guaranteed) & \
                     (panda.loc[:, "Initial fund size"] == ini_fund) & \
                     (panda.loc[:, "Number of covered years"] == T))]
                  
    # no results for these settings
    if sub_panda.empty:
        sys.exit("Requested data is not available.")
        
    # finding right sample size
    if N is not None:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size"] == N]
        # nor results for right sample siize
        if sub_panda.empty:
            sys.exit("Reyuested data is not available.")
        return(sub_panda)
        
    # results for highest sample size for these settings
    sub_panda = sub_panda[sub_panda["Sample size"] == max(sub_panda["Sample size"])]
    # if multiple runs for highest sample size, find highest validation sample size
    if len(sub_panda) == 1:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size for validation"] == \
                                          max(sub_panda["Sample size for validation"])]
    else:
        sub_panda = sub_panda[output_var_fct]
                
    
    def __ConvertListsInts(arg):
        print(arg, flush = True)
        arg = arg.strip("][").split(", ")
        res = []
        for j in range(0, len(arg)):
            res.append(int(arg[j]))
        return(res)

    # turn back to lists
    sub_panda["Used clusters"] = sub_panda["Used clusters"].apply(__ConvertListsInts)
        
    return(sub_panda)
    
def ReadFromPanda(file = "current_panda", 
                 output_var = None,
                 k_using = [3],
                 **kwargs):
    
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
    
    sub_panda = pd.DataFrame()
    for k_using_tmp in k_using:
        sub_panda = sub_panda.append(ReadFromPandaSingleClusterGroup(file = file, \
                                                        output_var = output_var, \
                                                        k_using = k_using_tmp, \
                                                        **kwargs))
        
    return(sub_panda)
            

def __ExtractResPanda(sub_panda, out_type, output_var, size):

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
        return(res)
    
    if out_type == "median":
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
        return(res)
    
    if out_type == "all":
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
                    close_plots = None,
                    **kwargs):
    
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
        plt.ylabel(var + " " + units[var], fontsize = 20)
        plt.legend(fontsize = 20)
        
    if plt_file is not None:
        fig.savefig("Figures/PandaPlots/" + plt_file + "_Median.jpg", bbox_inches = "tight", pad_inches = 1)
        
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
                 close_plots = None,
                 **kwargs):
    
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
        plt.ylabel(var + " " + units[var], fontsize = 20)
        # plt.legend(fontsize = 20)
        
    if plt_file is not None:
        fig.savefig("Figures/PandaPlots/" + plt_file + "_All.jpg", bbox_inches = "tight", pad_inches = 1)
      
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
                    close_plots = None,
                    **kwargs):
    
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
        fig.savefig("Figures/PandaPlots/" + plt_file + "_Agg.jpg", bbox_inches = "tight", pad_inches = 1)
    
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
                    close_plots = None,
                    **kwargs):
    
    PlotPandaMedian(panda_file = panda_file, 
                    output_var = output_var,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    figsize = figsize,
                    subplots = subplots,
                    plt_file = plt_file,
                    close_plots = close_plots,
                    **kwargs)

    PlotPandaAll(panda_file = panda_file, 
                    output_var = output_var,
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    figsize = figsize,
                    subplots = subplots,
                    plt_file = plt_file,
                    close_plots = close_plots,
                    **kwargs)
    
    return(None)

def MainPandaPlotsFixedSettings(panda_file = "current_panda", 
                                grouping_aim = "Dissimilar",
                                adjacent = False,
                                **kwargs):
    
    if adjacent:
        add = "Adj"
    else:
        add = ""
    
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average yearly total cultivated area', \
                                   'Average total cultivation costs'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_TotalAllocArea_TotalCultCosts",
                       close_plots = True,
                       **kwargs)
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Necessary add. import (excluding solvency constraint, including theoretical export)', \
                                   'Necessary debt (excluding food security constraint)'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_NecImportsPen_NecDebtPen",
                       close_plots = True,
                       **kwargs)
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Total necessary import when including solvency constraint', \
                                   'Necessary debt (including food security constraint)'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_NecImports_NecDebt",
                       close_plots = True,
                       **kwargs)
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Penalty for food shortage', \
                                'Penalty for insolvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_Penalties",
                    close_plots = True,
                    **kwargs)

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security', \
                                'Resulting probability for solvency'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_ResProbabilities",
                    close_plots = True,
                    **kwargs)

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average food shortcomings (over all years and samples with shortcomings)', \
                                'Average final fund (over all samples with negative final fund)'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_ShortcomingConstraints",
                    close_plots = True,
                    **kwargs)
        
    PlotPandaAggregate(panda_file = panda_file,
                       output_var=['Average food demand penalty (over years and samples)', \
                                   'Average solvency penalty (over samples)'],
                       grouping_aim = grouping_aim,
                       adjacent = adjacent,
                       plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_PenaltiesPaied",
                       close_plots = True,
                       **kwargs)
        
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Value of stochastic solution', \
                                'Resulting probability for food security for VSS',\
                                'Resulting probability for solvency for VSS'],
                    grouping_aim = grouping_aim,
                    adjacent = adjacent,
                    plt_file = "DevelopmentColaboration/" + grouping_aim + add + "_VSS",
                    close_plots = True,
                    **kwargs)
        
    return(None)

def PlotPenaltyVsProb(panda_file = "current_panda", 
                      grouping_aim = "Dissimilar",
                      adjacent = False,
                      figsize = None,
                      **kwargs):
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
        
    if adjacent:
        add = "Adj"
    else:
        add = ""
    
    plt_file = "PenaltiesProbabilities_" + grouping_aim + add
    
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
            
    fig.savefig("Figures/PandaPlots/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)

    plt.close()    

    return(None)