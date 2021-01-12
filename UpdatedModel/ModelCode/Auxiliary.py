#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:09:14 2021

@author: Debbora Leip
"""
import itertools as it
import numpy as np
import pandas as pd
import os

# %% ############################### AUXILIARY ################################  

def flatten(ListOfLists):
    """
    Turn list of lists into a list.

    Parameters
    ----------
    ListOfLists : list
        A lits with lists as elements.

    Returns
    -------
    FlatList : list
        List made out of single elements.

    """
    return(list(it.chain(*ListOfLists)))

def MakeList(grouping):
    """
    Function to make a single list out of the clusters in a grouping (which 
    is given by a list of tuples)

    Parameters
    ----------
    grouping : list
        all groups (given by tuples of clusters within a group)

    Returns
    -------
    clusters : list
        all clusters within that grouping

    """
    res = []
    for gr in grouping:
        if type(gr) is int:
            res.append(gr)
        else:
            for i in range(0, len(gr)):
                res.append(gr[i])
    return(res)        
    
def printing(content, console_output = None, flush = True, logs_on = None):
    """
    Function to only print progress report to console if chosen.

    Parameters
    ----------
    content : str
        Message that is to be printed.
    console_output : boolean, optional
        Whether message should be printed to console. 
        The default is defined in ModelCode/GeneralSettings.
    flush : bpolean, optional
        Whether to forcibly flush the stream. The default is True.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    None.

    """
    
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    if logs_on is None:
        from ModelCode.GeneralSettings import logs_on
    
    # output to consoole
    if console_output:
        print(content, flush = flush)
    
    # output to log file
    if logs_on:
        log = open("ModelLogs/tmp.txt", "a")
        log.write("\n" + content)
        log.close()
        
    return(None)
    
def filename(settings, groupSize = "", groupAim = "", \
             adjacent = False, allNames = False):
    """
    Combines all settings to a single file name to save results.

    Parameters
    ----------
    settings : dict
        Input settings for the model.
    groupSize : int
        in case loading data for e.g. all groups from a specific cluster 
        grouping, this is the size of the groups (relevant for filename of
        figures)
    groupSize : int
        in case loading data for e.g. all groups from a specific cluster 
        grouping, this is the aim of the grouping (relevant for filename of
        figures)
    adjacent : boolean
        in case loading data for e.g. all groups from a specific cluster 
        grouping, this states whether clusters within a group had to be 
        adjacent (relevant for filename of figures)
    allNames : boolean
        if True, also the names  for SettingsAffectingRhoF etc are returned.
        Else only the filename for model outputs. Default is False.
        
    Returns
    -------
    fn : str
        Filename combining all settings.

    """
        
    settingsTmp = settings.copy()
    if type(settingsTmp["k_using"]) is tuple:
        settingsTmp["k_using"] = list(settingsTmp["k_using"])
    if type(settingsTmp["k_using"]) is list:
        settingsTmp["k_using"] = MakeList(settingsTmp["k_using"])
        
    for key in settingsTmp.keys():
        if type(settingsTmp[key]) is not list:
            settingsTmp[key] = [settingsTmp[key]]
        
    if type(settings["validation_size"]) is not list:
        validationTmp = [settings["validation_size"]]
    else:
        validationTmp = settings["validation_size"]
    
    if settings["PenMet"] == "prob":
        if type(settings["probF"]) is not list:
            probFTmp = [settings["probF"]]
        else:
            probFTmp = settings["probF"]
        if type(settings["probS"]) is not list:
            probSTmp = [settings["probS"]]
        else:
            probSTmp = settings["probS"]
        fn = "pF" + '_'.join(str(n) for n in probFTmp) + \
             "pS" + '_'.join(str(n) for n in probSTmp)
    else:
        rhoFTmp = settings["rhoF"].copy()
        rhoSTmp = settings["rhoS"].copy()
        if type(rhoFTmp) is not list:
            rhoFTmp = [rhoFTmp]
        if type(rhoSTmp) is not list:
            rhoSTmp = [rhoSTmp]
        fn = "rF" + '_'.join(str(n) for n in rhoFTmp) + \
             "rS" + '_'.join(str(n) for n in rhoSTmp)
     
    if groupSize != "":    
        groupSize = "GS" + str(groupSize)
   
    if adjacent:
        ad = "Adj"
    else:
        ad = ""     
        
    fn = fn + "K" + '_'.join(str(n) for n in settingsTmp["k"]) + \
        "using" +  '_'.join(str(n) for n in settingsTmp["k_using"]) + \
        groupAim + groupSize + ad + \
        "Yield" + '_'.join(str(n).capitalize() for n in settingsTmp["yield_projection"]) + \
        "Pop" + '_'.join(str(n).capitalize() for n in settingsTmp["pop_scenario"]) +  \
        "Risk" + '_'.join(str(n) for n in settingsTmp["risk"]) + \
        "N" + '_'.join(str(n) for n in settingsTmp["N"]) + \
        "M" + '_'.join(str(n) for n in validationTmp) + \
        "Tax" + '_'.join(str(n) for n in settingsTmp["tax"]) + \
        "PercIgov" + '_'.join(str(n) for n in settingsTmp["perc_guaranteed"])
    
    
    if allNames:
        # all settings that affect the calculation of rhoF
        SettingsBasics = "k" + str(settings["k"]) + \
                "using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "num_crops" + str(settings["num_crops"]) + \
                "yield_projection" + str(settings["yield_projection"]) + \
                "sim_start" + str(settings["sim_start"]) + \
                "pop_scenario" + str(settings["pop_scenario"]) +  \
                "T" + str(settings["T"])
        SettingsMaxProbF = SettingsBasics + "N" + str(settings["N"])
        SettingsAffectingRhoF = SettingsBasics + "probF" + str(settings["probF"]) + \
                "N" + str(settings["N"])
        
        # all settings that affect the calculation of rhoS
        SettingsBasics = SettingsBasics + \
                "risk" + str(settings["risk"]) + \
                "tax" + str(settings["tax"]) + \
                "perc_guaranteed" + str(settings["perc_guaranteed"])
        SettingsMaxProbS = SettingsBasics + "N" + str(settings["N"])
        SettingsAffectingRhoS = SettingsBasics + "probS" + str(settings["probS"]) + \
                "N" + str(settings["N"])
        return(fn, SettingsMaxProbF, SettingsAffectingRhoF, \
               SettingsMaxProbS, SettingsAffectingRhoS)
    
    return(fn)


def write_to_pandas(settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, crop_alloc, \
                    meta_sol, meta_sol_vss, VSS_value, validation_values, \
                    console_output):
    """
    Adds information on the model run to the current pandas csv.
    
    Parameters
    ----------
    settings : dict
        The model input settings that were given by user. 
    args : dict
        Dictionary of arguments needed as direct model input.
    AddInfo_CalcParameters : dict
        Additional information from calculatings expected income and penalties
        which are not needed as model input.
    yield_information : dict
        Information on the yield distributions for the considered clusters.
    population_information : dict
        Information on the population in the considered area.
        DESCRIPTION.
    crop_alloc :  np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'S', 'exp_incomes', 'profits', 'exp_shortcomings', 
        'fd_penalty', 'avg_fd_penalty', 'sol_penalty', 'final_fund', 
        'prob_staying_solvent', 'prob_food_security', 'payouts', 
        'yearly_fixed_costs', 'num_years_with_losses')
    meta_sol_vss : dict
        additional information on the deterministic solution 
    VSS_value : float
        VSS calculated as the difference between total costs using 
        deterministic solution for crop allocation and stochastic solution
        for crop allocation       
    validation_values : dict
        total costs and penalties for the model result and a higher sample 
        size for validation ("sample_size", "total_costs", "total_costs_val", 
        "fd_penalty", "fd_penalty_val", "sol_penalty", "sol_penalty_val", 
        "total_penalties", "total_penalties_val", "deviation_penalties")
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.

    Returns
    -------
    None.

    """
 
    
    printing("\nAdding results to pandas", console_output = console_output)
    if settings["PenMet"] == "prob":
        dict_for_pandas = {"Input probability food security": settings["probF"],
                           "Input probability solvency": settings["probS"],
                           "Number of clusters": settings["k"],
                           "Used clusters": settings["k_using"],
                           "Yield projection": settings["yield_projection"],
                           "Simulation start": settings["sim_start"],
                           "Population scenario": settings["pop_scenario"],
                           "Risk level covered": settings["risk"],
                           "Tax rate": settings["tax"],
                           "Share of income that is guaranteed": settings["perc_guaranteed"],
                           "Initial fund size": settings["ini_fund"],
                           "Sample size": settings["N"],
                           "Sample size for validation": settings["validation_size"],
                           "Number of covered years": settings["T"],
                           "Average food demand": np.mean(args["demand"]),
                           "Import (excluding solvency constraint)": args["import"],
                           "Import (excluding solvency constraint, including theoretical export)": AddInfo_CalcParameters["import"],
                           "Additional import needed when including solvency constraint": meta_sol["add_needed_import"],
                           "Expected income (to calculate guaranteed income)": AddInfo_CalcParameters["expected_incomes"],
                           "Penalty for food shortage": args["rhoF"],
                           "Penalty for insolvency": args["rhoS"],
                           "Necessary debt (excluding food security constraint)": AddInfo_CalcParameters["necessary_debt"],
                           "Necessary debt (including food security constraint)": meta_sol["necessary_debt"],
                           "Probability for a catastrophic year": yield_information["prob_cat_year"],
                           "Share of samples with no catastrophe": yield_information["share_no_cat"],
                           "Share of years/clusters with unprofitable rice yields": yield_information["share_rice_np"],
                           "Share of years/clusters with unprofitable maize yields": yield_information["share_maize_np"],
                           "Share of West Africa's population that is living in currently considered region (2015)": \
                               population_information["pop_area_ratio2015"],
                           "On average cultivated area per cluster": np.nanmean(crop_alloc, axis = (0,1)),
                           "Average food demand penalty (over years and samples)": np.nanmean(meta_sol["fd_penalty"]),
                           "Average solvency penalty (over samples)": np.mean(meta_sol["sol_penalty"]),
                           "Average cultivation costs per cluster (over years and samples)": np.nanmean(meta_sol["yearly_fixed_costs"], axis = (0,1)),
                           "Expected total costs": meta_sol["exp_tot_costs"],
                           "Average food shortcomings (over years and samples)": np.nanmean(meta_sol["shortcomings"]),
                           "Number of occurrences per cluster where farmers make losses": meta_sol["num_years_with_losses"],
                           "Average income per cluster in final run (over years and samples)": np.nanmean(meta_sol["profits"], axis = (0,1)),
                           "Average government payouts per cluster (over samples)": np.nanmean(np.nansum(meta_sol["payouts"], axis = 1), axis = 0),
                           "Resulting probability for food security": meta_sol["probF"],
                           "Resulting probability for solvency": meta_sol["probS"],
                           "Resulting probability for food security for VSS": meta_sol_vss["probF"],
                           "Resulting probability for solvency for VSS": meta_sol_vss["probS"],
                           "Value of stochastic solution": VSS_value,
                           "Validation value (deviation of total penalty costs)": validation_values["deviation_penalties"]}
        
        current_panda = pd.read_csv("ModelOutput/Pandas/current_panda.csv")
        current_panda = current_panda.append(dict_for_pandas, ignore_index = True)
        current_panda.to_csv("ModelOutput/Pandas/current_panda.csv", index = False)
        
    return(None)

def SetUpNewPandas(name_old_pandas):
    """
    Renames the current pandas csv according to the given name and sets up a
    new current pandas csv.

    Parameters
    ----------
    name_old_pandas : str
        filenme for the csv.

    Returns
    -------
    None.

    """
    
    # save old panda
    current_panda = pd.read_csv("ModelOutput/Pandas/current_panda.csv")
    current_panda.to_csv("ModelOutput/Pandas/" + name_old_pandas + ".csv", index = False)
    
    # create new empty panda
    os.remove("ModelOutput/Pandas/current_panda.csv")
    CreateEmptyPanda()
    
    return(None)

def CreateEmptyPanda():
    """
    Creating a new empty pandas object with the correct columns.

    Returns
    -------
    None.

    """
    
    colnames = ['Input probability food security', 
                'Input probability solvency', 
                'Number of clusters', 
                'Used clusters', 
                'Yield projection', 
                'Simulation start', 
                'Population scenario', 
                'Risk level covered', 
                'Tax rate', 
                'Share of income that is guaranteed', 
                'Initial fund size', 
                'Sample size', 
                'Sample size for validation', 
                'Number of covered years', 
                'Average food demand', 
                'Import (excluding solvency constraint)', 
                'Import (excluding solvency constraint, including theoretical export)', 
                'Additional import needed when including solvency constraint', 
                'Expected income (to calculate guaranteed income)', 
                'Penalty for food shortage', 
                'Penalty for insolvency', 
                'Necessary debt (excluding food security constraint)', 
                'Necessary debt (including food security constraint)', 
                'Probability for a catastrophic year', 
                'Share of samples with no catastrophe', 
                'Share of years/clusters with unprofitable rice yields', 
                'Share of years/clusters with unprofitable maize yields', 
                'Share of West Africa\'s population that is living in currently considered region (2015)', 
                'On average cultivated area per cluster', 
                'Average food demand penalty (over years and samples)', 
                'Average solvency penalty (over samples)', 
                'Average cultivation costs per cluster (over years and samples)', 
                'Expected total costs', 
                'Average food shortcomings (over years and samples)', 
                'Number of occurrences per cluster where farmers make losses', 
                'Average income per cluster in final run (over years and samples)', 
                'Average government payouts per cluster (over samples)', 
                'Resulting probability for food security', 
                'Resulting probability for solvency', 
                'Resulting probability for food security for VSS', 
                'Resulting probability for solvency for VSS', 
                'Value of stochastic solution', 
                'Validation value (deviation of total penalty costs)']
    
    new_panda = pd.DataFrame(columns = colnames)
    new_panda.to_csv("ModelOutput/Pandas/current_panda.csv", index = False)

    return(new_panda)