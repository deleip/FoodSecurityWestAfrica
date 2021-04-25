#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:09:14 2021

@author: Debbora Leip
"""
import itertools as it

# %% ############################### AUXILIARY ################################  


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
    Function the prints progress to console and/or to log file if chosen.

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
        import ModelCode.GeneralSettings as GS
        log = open("ModelLogs/" + GS.fn_log + ".txt", "a")
        log.write("\n" + content)
        log.close()
        
    return(None)
    
def GetFilename(settings, groupSize = "", groupAim = "", \
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
    groupAim : str
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
    SettingsMaxProbF : str
        Only if allNames is True.
    SettingsAffectingRhoF : str
        Only if allNames is True.
    SettingsMaxProbS : str
        Only if allNames is True.
    SettingsAffectingRhoS : str
        Only if allNames is True.
    """
        
    settingsTmp = settings.copy()
    if type(settingsTmp["k_using"]) is tuple:
        settingsTmp["k_using"] = list(settingsTmp["k_using"])
    if type(settingsTmp["k_using"]) is list:
        settingsTmp["k_using"] = MakeList(settingsTmp["k_using"])
    
    if sorted(settingsTmp["k_using"]) == list(range(1, settingsTmp["k"] + 1)):
        settingsTmp["k_using"] = ["All"]
    
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
    
    
    fn = fn + "K" + '_'.join(str(n) for n in settingsTmp["k"])
    if settingsTmp["k_using"] != [""]:
        fn = fn + "using" +  '_'.join(str(n) for n in settingsTmp["k_using"]) 
    fn = fn + groupAim + groupSize + ad + \
        "Yield" + '_'.join(str(n).capitalize() for n in settingsTmp["yield_projection"]) + \
        "Pop" + '_'.join(str(n).capitalize() for n in settingsTmp["pop_scenario"]) +  \
        "Risk" + '_'.join(str(n) for n in settingsTmp["risk"])
    if settingsTmp["N"] != [""]:
        fn = fn + "N" + '_'.join(str(n) for n in settingsTmp["N"])
    if validationTmp != [""]:
        fn = fn + "M" + '_'.join(str(n) for n in validationTmp)
    fn = fn + "Tax" + '_'.join(str(n) for n in settingsTmp["tax"]) + \
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


def GetDefaults(PenMet, probF, probS, rhoF, rhoS, k, k_using,
                num_crops, yield_projection, sim_start, pop_scenario,
                risk, N, validation_size, T, seed, tax, perc_guaranteed,
                ini_fund, food_import):
                                    
    if PenMet == "default":
        from ModelCode.DefaultModelSettings import PenMet
    if probF == "default":
        from ModelCode.DefaultModelSettings import probF
    if probS == "default":
        from ModelCode.DefaultModelSettings import probS
    if rhoF == "default":
        from ModelCode.DefaultModelSettings import rhoF
    if rhoS == "default":
        from ModelCode.DefaultModelSettings import rhoS
    if k == "default":
        from ModelCode.DefaultModelSettings import k
    if k_using == "default":
        from ModelCode.DefaultModelSettings import k_using
    if num_crops == "default":
        from ModelCode.DefaultModelSettings import num_crops
    if yield_projection == "default":
        from ModelCode.DefaultModelSettings import yield_projection
    if sim_start == "default":
        from ModelCode.DefaultModelSettings import sim_start
    if pop_scenario == "default":
        from ModelCode.DefaultModelSettings import pop_scenario
    if risk == "default":
        from ModelCode.DefaultModelSettings import risk
    if N == "default":
        from ModelCode.DefaultModelSettings import N
    if validation_size == "default":
        from ModelCode.DefaultModelSettings import validation_size
    if T == "default":
        from ModelCode.DefaultModelSettings import T
    if seed == "default":
        from ModelCode.DefaultModelSettings import seed
    if tax == "default":
        from ModelCode.DefaultModelSettings import tax
    if perc_guaranteed == "default":
        from ModelCode.DefaultModelSettings import perc_guaranteed
    if ini_fund == "default":
        from ModelCode.DefaultModelSettings import ini_fund
    if food_import == "default":
        from ModelCode.DefaultModelSettings import food_import
        
    return(PenMet, probF, probS, rhoF, rhoS, k, k_using,
          num_crops, yield_projection, sim_start, pop_scenario,
          risk, N, validation_size, T, seed, tax, perc_guaranteed,
          ini_fund, food_import)