#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:09:14 2021

@author: Debbora Leip
"""
import itertools as it

from ModelCode.GeneralSettings import logs_on

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
    
def printing(content, prints = True, flush = True):
    """
    Function to only print progress report to console if chosen.

    Parameters
    ----------
    content : str
        Message that is to be printed.
    prints : boolean, optional
        Whether message should be printed to console. The default is True.
    LogFile : boolean, optional
        Whether message should be printed to log file. If None, the same value
        as prints is used. The default is None.
    flush : bpolean, optional
        Whether to forcibly flush the stream. The default is True.

    Returns
    -------
    None.

    """
    # output to consoole
    if prints:
        print(content, flush = flush)
    
    # output to log file
    if logs_on:
        log = open("ModelLogs/tmp.txt", "a")
        log.write("\n" + content)
        log.close()
        
    return(None)
    
def filename(settings, PenMet, validation, probF = 0.95, probS = 0.95, \
             rhoF = None, rhoS = None, groupSize = "", groupAim = "", \
             adjacent = False, allNames = False):
    """
    Combines all settings to a single file name to save results.

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.
    PenMet : "prob" or "penalties"
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly.
    validation : None or int
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.
    probF : float
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float
        The penalty for shortcomings of the food demand (only relevant if 
        PenMet == "penalties"). The default is None.
    rhoS : float
        The penalty for insolvency (only relevant if PenMet == "penalties").
        The default is None.
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
        
    if type(validation) is not list:
        validationTmp = [validation]
    else:
        validationTmp = validation
    
    if PenMet == "prob":
        if type(probF) is not list:
            probFTmp = [probF]
        else:
            probFTmp = probF
        if type(probS) is not list:
            probSTmp = [probS]
        else:
            probSTmp = probS
        fn = "pF" + '_'.join(str(n) for n in probFTmp) + \
             "pS" + '_'.join(str(n) for n in probSTmp)
    else:
        rhoFTmp = rhoF.copy()
        rhoSTmp = rhoS.copy()
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
        SettingsAffectingRhoF = SettingsBasics + "probF" + str(probF) + \
                "N" + str(settings["N"])
        
        # all settings that affect the calculation of rhoS
        SettingsBasics = SettingsBasics + \
                "risk" + str(settings["risk"]) + \
                "tax" + str(settings["tax"]) + \
                "perc_guaranteed" + str(settings["perc_guaranteed"])
        SettingsMaxProbS = SettingsBasics + "N" + str(settings["N"])
        SettingsAffectingRhoS = SettingsBasics + "probS" + str(probS) + \
                "N" + str(settings["N"])
        return(fn, SettingsMaxProbF, SettingsAffectingRhoF, \
               SettingsMaxProbS, SettingsAffectingRhoS)
    
    return(fn)