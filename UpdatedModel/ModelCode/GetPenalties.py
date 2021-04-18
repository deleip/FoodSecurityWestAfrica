#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:05:11 2021

@author: Debbora Leip
"""
import numpy as np
import pickle
import os
# import sys
import matplotlib.pyplot as plt

# from ModelCode.MetaInformation import GetMetaInformation
from ModelCode.Auxiliary import printing
from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy
from ModelCode.SettingsParameters import DefaultSettingsExcept

# %% ########################## WRAPPING FUNCTION #############################

def GetPenalties(settings, args, console_output = None,  logs_on = None):
    """
    Given the probabilities probF and probS this either loads or calculates
    the corresponding penalties. Penalties are calculated with the respective
    other penalty set to zero.

    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    args : dict
        Dictionary of arguments needed as model input.  
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs.  If None, the default as defined in ModelCode/GeneralSettings is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    rhoF : float
        The correct penalty rhoF to reach the probability probF (or the highest
        possible probF).
    rhoS : float
        The correct penalty rhoF to reach the probability probS (or the highest
        possible probS).
    necessary_debt : float
        The necessary debt to cover the payouts in probS of the cases (when 
        rhoF = 0).
    needed_import : float
        Amount of food that needs to imported to reach the probability for
        food seecurity probF (when using only rhoF and setting rhoS = 0)
    maxProbFareaF : float
        Maximum possible probability for food security for the given settings.
    maxProbSareaF : float
        Probability for solvency for areas to reach maximum porbability for food 
        security.
    maxProbFareaS : float
        Probability for food security for areas to reach maximum porbability 
        for solvency.
    maxProbSareaS : float
        Maximum possible probability for solvency for the given settings.
    """
    # extract some settings (that were originally passed on directly...)
    probF = args["probF"]
    probS = args["probS"]
    rhoFini = settings["rhoF"]
    rhoSini = settings["rhoS"]
    
    if probF == 0:
        rhoF = 0
    else:   
        # all settings that affect the calculation of rhoF
        SettingsBasics = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Yield" + str(settings["yield_projection"]).capitalize() + \
                "Start" + str(settings["sim_start"]) + \
                "Pop" + str(settings["pop_scenario"]).capitalize() + \
                "T" + str(settings["T"])
        SettingsFirstGuess =  SettingsBasics + "ProbF" + str(probF)
        SettingsAffectingRhoF = SettingsFirstGuess + "N" + str(settings["N"])
                        
        # get dictionary of settings for which rhoF has been calculated already
        with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
            dict_rhoFs = pickle.load(fp)
        with open("PenaltiesAndIncome/resProbFs.txt", "rb") as fp:    
            dict_probFs = pickle.load(fp)
        with open("PenaltiesAndIncome/resAvgImport.txt", "rb") as fp:    
            dict_import = pickle.load(fp)
            
        # if this setting was already calculated, fetch rhoF
        if SettingsAffectingRhoF in dict_rhoFs.keys():
            rhoF = dict_rhoFs[SettingsAffectingRhoF]
            printing("Fetching rhoF: " + str(rhoF), console_output = console_output, logs_on = logs_on)
            probF_onlyF = dict_probFs[SettingsAffectingRhoF]
            avg_nec_import = dict_import[SettingsAffectingRhoF]
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoF calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 1)
            if rhoFini is None:
                rhoFini, checkedGuess = GetInitialGuess(dict_rhoFs, SettingsFirstGuess, settings["N"])
            # calculating rhoF
            printing("Calculating rhoF and import", console_output = console_output, logs_on = logs_on)
            
            rhoF, meta_solF = \
                GetRhoWrapper(args, probF, rhoFini, checkedGuess, "F",
                  SettingsAffectingRhoF, console_output = None, logs_on = None)
                
            probF_onlyF = meta_solF["probF"] 
            avg_nec_import = meta_solF["avg_nec_import"]
            
            dict_rhoFs[SettingsAffectingRhoF] = rhoF
            dict_probFs[SettingsAffectingRhoF] = probF_onlyF
            dict_import[SettingsAffectingRhoF] = avg_nec_import
        
        # saving updated dicts
        with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:    
             pickle.dump(dict_rhoFs, fp)
        with open("PenaltiesAndIncome/resProbFs.txt", "wb") as fp:    
             pickle.dump(dict_probFs, fp)
        with open("PenaltiesAndIncome/resAvgImport.txt", "wb") as fp:     
             pickle.dump(dict_import, fp)
            
  
    if probS == 0:
        rhoS = 0
    else:         
        # all settings that affect the calculation of rhoS
        SettingsBasics = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Yield" + str(settings["yield_projection"]).capitalize() + \
                "Start" + str(settings["sim_start"]) + \
                "Pop" + str(settings["pop_scenario"]).capitalize() +  \
                "T" + str(settings["T"])
        # SettingsMaxProbF = SettingsBasics + "N" + str(settings["N"])
        SettingsBasics = SettingsBasics + \
                "Risk" + str(settings["risk"]) + \
                "Tax" + str(settings["tax"]) + \
                "PercGuar" + str(settings["perc_guaranteed"])
        # SettingsMaxProbS = SettingsBasics + "N" + str(settings["N"])
        SettingsFirstGuess = SettingsBasics + "ProbS" + str(probS)
        SettingsAffectingRhoS = SettingsFirstGuess + \
                "N" + str(settings["N"])
                     
        # get dictionary of settings for which rhoS has been calculated already
        with open("PenaltiesAndIncome/RhoSs.txt", "rb") as fp:    
            dict_rhoSs = pickle.load(fp)
        with open("PenaltiesAndIncome/resProbSs.txt", "rb") as fp:    
            dict_probSs = pickle.load(fp)
        with open("PenaltiesAndIncome/resAvgDebt.txt", "rb") as fp:    
            dict_debt = pickle.load(fp)
           
        # if this setting was already calculated, fetch rhoS
        if SettingsAffectingRhoS in dict_rhoSs.keys():
            rhoS = dict_rhoSs[SettingsAffectingRhoS]
            printing("\nFetching rhoS: " + str(rhoS), console_output = console_output, logs_on = logs_on)
            probS_onlyS = dict_probSs[SettingsAffectingRhoS]
            avg_nec_debt = dict_debt[SettingsAffectingRhoS]           
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoS calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 100)
            if rhoSini is None:
                rhoSini, checkedGuess = GetInitialGuess(dict_rhoSs, SettingsFirstGuess, settings["N"])
            # calculating rhoS
            printing("\nCalculating rhoS", console_output = console_output, logs_on = logs_on)
            rhoS, meta_solS = \
                GetRhoWrapper(args, probS, rhoSini, checkedGuess, "S",
                  SettingsAffectingRhoS, console_output = None, logs_on = None)
                
            probS_onlyS = meta_solS["probS"]
            avg_nec_debt = meta_solS["avg_nec_debt"]
            
            dict_rhoSs[SettingsAffectingRhoS] = rhoS
            dict_probSs[SettingsAffectingRhoS] = probS_onlyS
            dict_debt[SettingsAffectingRhoS] = avg_nec_debt
        
        # saving updated dict
        with open("PenaltiesAndIncome/RhoSs.txt", "wb") as fp:    
             pickle.dump(dict_rhoSs, fp)
        with open("PenaltiesAndIncome/resProbSs.txt", "wb") as fp:    
             pickle.dump(dict_probSs, fp)
        with open("PenaltiesAndIncome/resAvgDebt.txt", "wb") as fp:     
             pickle.dump(dict_debt, fp)
             
    return(rhoF, rhoS, probF_onlyF, probS_onlyS, avg_nec_import, avg_nec_debt)

def GetRhoWrapper(args, prob, rhoIni, checkedGuess, objective,
                  file, console_output = None, logs_on = None):
    """
    Finding the correct rhoS given the probability probS, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Information on the yield distributions.
    probF : float
        demanded probability of keeping the food security constraint (only 
        relevant if PenMet == "prob").
    rhoFini : float or None 
        Initial guess for rhoF.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed.
    file : str
        String combining all settings affecting rhoF, used to save a plot 
        of rhoF vs. necessary import in MinimizeNecessaryImport. 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    rhoS : float
        The correct penalty rhoF to reach the given probF (or the highest
        possible probF).
    maxProbFareaF : float
        Maximum possible probability for food security for the given settings.
    maxProbSareaF : float
        Probability for solvency for areas to reach maximum porbability for food 
        security.
    needed_import : float
        Amount of food that needs to imported to reach the probability for
        food seecurity probF (when using only rhoF and setting rhoS = 0)
    crop_alloc : np.array
        Crop area allocations for the resulting penalty.
    meta_sol : dict
        Dictionary on model outputs for the resulting penalty.
    """
    if objective == "F":
        from ModelCode.GeneralSettings import accuracyF as accuracy
        from ModelCode.GeneralSettings import shareDiffF as shareDiff
    elif objective == "S":
        from ModelCode.GeneralSettings import accuracyS as accuracy
        from ModelCode.GeneralSettings import shareDiffS as shareDiff
    
    # find the highest possible probF (and probS when using area to get the max
    # probF)
    maxProb, crop_alloc_conv =  CheckOptimalProb(args, prob, objective, accuracy,
                      console_output = console_output, logs_on = logs_on)
        
    # if probF can be reached find lowest rhoF that gives probF
    if maxProb >= prob:
        printing("     Finding corresponding penalty\n", console_output)
        rho, meta_sol = RhoProbability(args, prob, rhoIni, checkedGuess, \
               objective, shareDiff, accuracy, console_output, logs_on)
    # if probF cannot be reached find rhoF that minimizes the import that is
    # necessary for to provide the food demand in probF of the samples
    else:
        if objective == "F":
            printing("     Finding lowest penalty minimizing total import\n", console_output)
        elif objective == "S":
            printing("     Finding lowest penalty minimizing total debt\n", console_output)
        rho, meta_sol = RhoCropAreas(args, prob, rhoIni,
                checkedGuess, objective, crop_alloc_conv, shareDiff, accuracy, file, \
                console_output = console_output, logs_on = console_output)
        
    printing("\n     Final rho" + objective + ": " + str(rho), console_output = console_output, logs_on = logs_on)
    
    return(rho, meta_sol)

    
def CheckOptimalProb(args, prob, objective, accuracy,
                      console_output = None, logs_on = None):
    """
    Function to find the highest probF possible under the given settings, and
    calculating the amount of import needed to increase this probabtility to 
    the probF desired.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Information on the yield distributions.
    probF : float
        The desired probability for food security.
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probF. 
        The default is defined in ModelCode/GeneralSettings.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    maxProbF : float
        Maximum probability for food security that can be reached under these 
        settings.
    maxProbS : float
        Probability for solvency for the settings that give the maxProbF.
    needed_import : float
        Amount of food that needs to imported to reach the probability for
        food seecurity probF.    
    """
    
    if objective == "F":
        rhoF = 1e12
        rhoS = 0
    elif objective == "S":
        rhoS = 1e12
        rhoF = 0
    
    
    # try for rhoF = 1e9 (as a proxy for rhoF = inf)
    status, crop_alloc, meta_sol, sto_prob, durations = \
         SolveReducedcLinearProblemGurobiPy(args, rhoF, rhoS, console_output = False, logs_on = False)  
       
    # get resulting probabilities
    maxProb = meta_sol["prob" + objective]
    printing("     maxProb" + objective + ": " + str(np.round(maxProb * 100, accuracy - 1)) + "%", \
              console_output = console_output, logs_on = logs_on)
    
    # check if it is high enough 
    if maxProb >= prob:
        printing("     Desired pro" + objective + " (" + str(np.round(prob * 100, accuracy - 1)) \
                             + "%) can be reached\n", console_output = console_output, logs_on = logs_on)
    else:
        printing("     Desired pro" + objective + " (" + str(np.round(prob * 100, accuracy - 1)) \
                             + "%) cannot be reached\n", console_output = console_output, logs_on = logs_on)
            
    return(maxProb, crop_alloc)


def RhoProbability(args, prob, rhoIni, checkedGuess, objective, shareDiff, accuracy, \
            console_output = None, logs_on = None):
    """
    Finding the correct rhoF given the probability probF, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    probF : float
        demanded probability of keeping the food demand constraint 
    rhoFini : float or None 
        Initial guess for rhoF.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed
    shareDiff : float
        The share of the final rhoF that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoF, for rhoF
        the current best guess for the correct penalty, then we use rhoF).
    accuracy : int
        Desired decimal places of accuracy of the obtained probF. 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    rhoF : float
        The resulting penalty rhoF    
    crop_alloc_out : np.array
        The optimal crop allocations using the penalty rhoF and setting rhoS 
        to zero in the given settings.
    meta_sol_out : dict
        Dictionary of meta information to the optimal crop allocations
    """
    def __setGuesses(nextGuess, objective = objective):
        if objective == "F":
            rhoF = nextGuess
            rhoS = 0
        elif objective == "S":
            rhoS = nextGuess
            rhoF = 0
        return(rhoF, rhoS)
    
   
    # accuracy information
    printing("     accuracy that we demand for prob" + objective + ": " + str(accuracy - 2) + " decimal places",\
             console_output = console_output, logs_on = logs_on)
    printing("     accuracy that we demand for rho" + objective + ": 1/" + str(shareDiff) + " of final rho" + objective + "\n", \
             console_output = console_output, logs_on = logs_on)
    
    # check if rhoF from run with smaller N works here as well:
    # if we get the right probF for our guess, and a lower probF for rhoFcheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rhoF is in that interval and can return our guess
    rho, crop_alloc, meta_sol = checkIniGuess(rhoIni, 
                                              args,
                                              checkedGuess,
                                              prob = prob,
                                              objective = objective,
                                              shareDiff = shareDiff,
                                              accuracy = accuracy,
                                              console_output = console_output,
                                              logs_on = logs_on)
    if rho is not None:
        return(rho, meta_sol)
    
    # else we start from scratch
    if objective == "F":
        rhoIni = 1
    elif objective == "S":
        rhoIni = 100
    rhoFini, rhoSini = __setGuesses(rhoIni, objective)
    
    # initialize values for search algorithm
    rhoLastDown = np.inf
    rhoLastUp = 0
    lowestCorrect = np.inf
    meta_sol_lowestCorrect = []
    # crop_alloc_lowestCorrect = []
    
    # calculate initial guess
    status, crop_alloc, meta_sol, sto_prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFini, rhoSini, console_output = False, logs_on = False)
    
    # update information
    if np.round(meta_sol["prob"+objective], accuracy) == prob:
        lowestCorrect = rhoIni
                
    # remember guess
    rhoOld = rhoIni
    
    # report
    accuracy_int = lowestCorrect - rhoLastUp
    ReportProgressFindingRho(rhoOld, meta_sol, accuracy, durations, \
                             objective, accuracy_int = accuracy_int,
                             console_output = console_output, logs_on = logs_on)

    while True:   
        # find next guess
        rhoNew, rhoLastDown, rhoLastUp = \
                    UpdatedRhoGuess(rhoLastUp, rhoLastDown, rhoOld, 
                                    meta_sol = meta_sol, prob = prob,
                                    objective = objective, accuracy = accuracy)
        rhoFnew, rhoSnew = __setGuesses(rhoNew, objective)
       
        # solve model for guess
        status, crop_alloc, meta_sol, sto_prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFnew, rhoSnew, console_output = False, logs_on = False)
        
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if np.round(meta_sol["prob"+objective], accuracy) == prob:
            accuracy_int = rhoNew - rhoLastUp
            if accuracy_int < rhoNew/shareDiff:
                rho = rhoNew
                meta_sol_out = meta_sol
                # crop_alloc_out = crop_alloc
                break
        elif np.round(meta_sol["prob"+objective], accuracy) < prob:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoNew
                if accuracy_int < lowestCorrect/shareDiff:
                    rho = lowestCorrect
                    meta_sol_out = meta_sol_lowestCorrect
                    # crop_alloc_out = crop_alloc_lowestCorrect
                    break
            else:
                accuracy_int = rhoLastDown - rhoNew
        elif np.round(meta_sol["prob"+objective], accuracy) > prob:
            accuracy_int = rhoNew - rhoLastUp
            
        # report
        ReportProgressFindingRho(rhoNew, meta_sol, accuracy, durations, \
                                 objective, accuracy_int = accuracy_int, 
                                 console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoOld = rhoNew
        if np.round(meta_sol["prob"+objective], accuracy) == prob \
            and lowestCorrect > rhoNew:
            lowestCorrect = rhoNew
            meta_sol_lowestCorrect = meta_sol
            # crop_alloc_lowestCorrect = crop_alloc
    
    # last report
    ReportProgressFindingRho(rhoNew, meta_sol, accuracy, durations, \
               objective, accuracy_int = accuracy_int, console_output = console_output, 
               logs_on = logs_on)    
            
    return(rho, meta_sol_out)


def RhoCropAreas(args, prob, rhoIni, checkedGuess, objective, \
                crop_alloc_conv, shareDiff, accuracy, file, \
                console_output = None, logs_on = None):
    """
    In cases where the probability probF cannot be reached, we instead look
    for the lowest penalty that minimizes the necessary import to cover the
    food demand in probF of the cases.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    probF : float
        demanded probability of keeping the food demand constraint 
    rhoFini : float or None 
        Initial guess for rhoF.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed
    necessary_import : float
        Import that is needed to cover food demand in probF of the cases when 
        using all agricultural area with the more productive crop in all years.
    shareDiff : float
        accuracy of the penalties given thorugh size of the accuracy interval;
        the size needs to be smaller than final rho / shareDiff
    accuracy : int
        Desired decimal places of accuracy of the obtained probF.
    file : str
        String combining all settings affecting rhoF, used to save a plot 
        of rhoF vs. necessary import. 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    rhoF : float
        The resulting penalty rhoF    
    crop_alloc_out : np.array
        The optimal crop allocations using the penalty rhoF and setting rhoS 
        to zero in the given settings.
    meta_sol_out : dict
        Dictionary of meta information to the optimal crop allocations

    """
    
    from ModelCode.GeneralSettings import accuracy_areas
    
    def __setGuesses(nextGuess, objective = objective):
        if objective == "F":
            rhoF = nextGuess
            rhoS = 0
        elif objective == "S":
            rhoS = nextGuess
            rhoF = 0
        return(rhoF, rhoS)
    
    def __right_area(crop_alloc):
        tol = args["max_areas"] * accuracy_areas
        s = np.sum(np.abs(crop_alloc - crop_alloc_conv) > tol)
        return(s == 0) 
    
    def __get_necessary_help(meta_sol, objective = objective):
        if objective == "F":
            return(meta_sol["avg_nec_import"])
        elif objective == "S":
            return(meta_sol["avg_nec_debt"])
    
    # accuracy information
    printing("     accuracy that we demand for rho" + objective + ": 1/" + 
             str(shareDiff) + " of final rho" + objective, console_output = console_output, logs_on = logs_on)
    
    # check if rhoF from run with smaller N works here as well:
    rho, crop_alloc, meta_sol = checkIniGuess(rhoIni, 
                                              args,
                                              checkedGuess,
                                              area = crop_alloc_conv,
                                              objective = objective,
                                              shareDiff = shareDiff,
                                              accuracy = accuracy,
                                              accuracy_areas = accuracy_areas,
                                              console_output = console_output,
                                              logs_on = logs_on)
   
  
    # else we start from scratch
    if objective == "F":
        rhoIni = 1
    elif objective == "S":
        rhoIni = 100
    rhoFini, rhoSini = __setGuesses(rhoIni, objective)
    
    # initialize values for search algorithm
    rhoLastDown = np.inf
    rhoLastUp = 0
    lowestCorrect = np.inf
    meta_sol_lowestCorrect = []
    # crop_alloc_lowestCorrect = []
    crop_allocs = []
    rhos_tried = []
    probabilities = []
    necessary_help = []
    
    # calculate initial guess
    status, crop_alloc, meta_sol, sto_prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFini, rhoSini, console_output = False, logs_on = False)
    crop_allocs.append(crop_alloc)
    rhos_tried.append(rhoIni)
    probabilities.append(meta_sol["prob"+objective])
    necessary_help.append(__get_necessary_help(meta_sol))
    
    # update information
    right_area = __right_area(crop_alloc)
    if right_area:
        lowestCorrect = rhoIni
                
    # remember guess
    rhoOld = rhoIni
    
    # report
    accuracy_int = lowestCorrect - rhoLastUp
    ReportProgressFindingRho(rhoOld, meta_sol, accuracy, durations, \
                            objective, method = "area", testPassed = right_area, 
                            accuracy_int = accuracy_int, 
                            console_output = console_output, logs_on = logs_on)

    while True:   
        # find next guess
        rhoNew, rhoLastDown, rhoLastUp = UpdatedRhoGuess(rhoLastUp, 
                     rhoLastDown, rhoOld, crop_alloc = crop_alloc,
                     crop_alloc_conv = crop_alloc_conv, max_areas = args["max_areas"])
        rhoFnew, rhoSnew = __setGuesses(rhoNew, objective)
        
        # solve model for guess
        status, crop_alloc, meta_sol, sto_prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFnew, rhoSnew,
                                                   console_output = False, logs_on = False)
        crop_allocs.append(crop_alloc)
        rhos_tried.append(rhoNew)
        probabilities.append(meta_sol["prob"+objective])
        necessary_help.append(__get_necessary_help(meta_sol))
        right_area = __right_area(crop_alloc)
        
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if right_area:
            accuracy_int = rhoNew - rhoLastUp
            if accuracy_int < rhoNew/shareDiff:
                rho = rhoNew
                meta_sol_out = meta_sol
                # crop_alloc_out = crop_alloc
                break
        else:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoNew
                if accuracy_int < lowestCorrect/shareDiff:
                    rho = lowestCorrect
                    meta_sol_out = meta_sol_lowestCorrect
                    # crop_alloc_out = crop_alloc_lowestCorrect
                    break
            else:
                accuracy_int = rhoLastDown - rhoNew
            
        # report
        ReportProgressFindingRho(rhoNew, meta_sol, accuracy, durations, \
                             objective, method = "area", testPassed = right_area, 
                             accuracy_int = accuracy_int, console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoOld = rhoNew
        if  __right_area(crop_alloc) and lowestCorrect > rhoNew:
            lowestCorrect = rhoNew
            meta_sol_lowestCorrect = meta_sol
            # crop_alloc_lowestCorrect = crop_alloc
    
    # last report
    ReportProgressFindingRho(rhoNew, meta_sol, accuracy, durations, \
                         objective, method = "area", testPassed = right_area, 
                         accuracy_int = accuracy_int, console_output = console_output, logs_on = logs_on)


    # ploting of information
    if args["T"] > 1:
        folder_path = "Figures/GetPenaltyFigures/rho" + objective + "/" + file
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        file = file.split("Crop")[0]
        
        s = sorted(zip(rhos_tried, crop_allocs, probabilities, necessary_help))
        rhos_tried_order = rhos_tried.copy()
        rhos_tried = [x for x,_,_,_ in s]
        crop_allocs = [x for _,x,_,_ in s]
        probabilities = [x for _,_,x,_ in s]
        necessary_help = [x for _,_,_,x in s]
        
        nrows = int(np.floor(np.sqrt(len(crop_allocs))))
        if nrows * nrows >= len(crop_allocs):
            ncols = nrows
        elif nrows * (nrows + 1) >= len(crop_allocs):
            ncols = nrows + 1
        else:
            nrows = nrows + 1
            ncols = nrows
        
        cols = ["royalblue", "darkred", "grey", "gold", \
                    "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                    "indigo"]
            
        from ModelCode.GeneralSettings import figsize
        
        if objective == "F":
            helptype = "import"
        elif objective == "S":
            helptype = "debt"
        
        T = crop_allocs[0].shape[0]
        
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95,
                    wspace=0.15, hspace=0.4)
        years = range(2017, 2017 + T)
        ticks = np.arange(2017, 2017 + T + 0.1, 3)
        for idx, crops in enumerate(crop_allocs):
            ax = fig.add_subplot(nrows, ncols, idx + 1)
            [T, num_crops, num_cluster] = crops.shape
            for cl in range(0, num_cluster):
                if cl == num_cluster -1:
                    ax.plot(years, crops[:,0,cl], ls = "-", color = cols[cl], label = "Rice")
                    ax.plot(years, crops[:,1,cl], ls = "--", color = cols[cl], label = "Maize")
                else:
                    ax.plot(years, crops[:,0,cl], ls = "-", color = cols[cl])
                    ax.plot(years, crops[:,1,cl], ls = "--", color = cols[cl])
                    
            ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
            ax.set_xticks(ticks)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_title("rho" + objective + ": " + "{:.2e}".format(rhos_tried[idx]) + 
                         ", prob" + objective + ": " + str(round(probabilities[idx], 2)) + 
                         ", " + helptype + ": " + str(round(necessary_help[idx], 2)), fontsize = 14)
            if rhos_tried[idx] == rho:
                idx_rho = idx
                ax.set_title("rho" + objective + ": " + "{:.2e}".format(rhos_tried[idx]) + 
                         ", prob" + objective + ": " + str(round(probabilities[idx], 2)) + 
                         ", " + helptype + ": " + str(round(necessary_help[idx], 2)), fontsize = 14, color = "red")
            if (idx % ncols) == 0:
                ax.set_ylabel("Crop areas", fontsize = 14)
            if idx >= len(crop_allocs) - ncols:
                ax.set_xlabel("Years", fontsize = 14)
        plt.legend()
        plt.suptitle("Crop areas for different penalties rho" + objective, fontsize = 22)
        fig.savefig(folder_path + "/CropAreas_" + file + ".jpg", \
                    bbox_inches = "tight", pad_inches = 1)
        plt.close() 
        
        fig = plt.figure(figsize = figsize)
        plt.scatter(rhos_tried, probabilities, color = "royalblue")
        plt.scatter(rhos_tried[idx_rho], probabilities[idx_rho], color = "red")
        plt.scatter(rhos_tried[np.argmin(probabilities)], min(probabilities), color = "green")
        plt.xlabel("rho" + objective, fontsize = 16)
        plt.ylabel("prob" + objective, fontsize = 16)
        plt.title("Probability of meeting objective for different penalties", fontsize = 24, pad = 10)
        fig.savefig(folder_path + "/Prob_" + file + ".jpg", \
                    bbox_inches = "tight", pad_inches = 1)
        plt.close() 
        
        fig = plt.figure(figsize = figsize)
        plt.scatter(rhos_tried, probabilities, color = "royalblue")
        plt.scatter(rhos_tried[idx_rho], probabilities[idx_rho], color = "red")
        plt.scatter(rhos_tried[np.argmin(probabilities)], min(probabilities), color = "green")
        plt.xlabel("rho" + objective, fontsize = 16)
        plt.ylabel("prob" + objective, fontsize = 16)
        plt.title("Probability of meeting objective for different penalties", fontsize = 24, pad = 10)
        plt.xscale("log", base = 4)
        fig.savefig(folder_path + "/ProbLog_" + file + ".jpg", \
                    bbox_inches = "tight", pad_inches = 1)
        plt.close() 
        
            
        fig = plt.figure(figsize = figsize)
        plt.scatter(rhos_tried, necessary_help, color = "royalblue")
        plt.scatter(rhos_tried[idx_rho], necessary_help[idx_rho], color = "red")
        plt.scatter(rhos_tried[np.argmin(necessary_help)], min(necessary_help), color = "green")
        plt.xlabel("rho" + objective, fontsize = 16)
        plt.ylabel("prob" + objective, fontsize = 16)
        plt.title("Necessary " + helptype + " to reach objective", fontsize = 24, pad = 10)
        fig.savefig(folder_path + "/Nec" + helptype.capitalize() + 
                    "_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        plt.close() 
        
        fig = plt.figure(figsize = figsize)
        plt.scatter(rhos_tried, necessary_help, color = "royalblue")
        plt.scatter(rhos_tried[idx_rho], necessary_help[idx_rho], color = "red")
        plt.scatter(rhos_tried[np.argmin(necessary_help)], min(necessary_help), color = "green")
        plt.xlabel("rho" + objective, fontsize = 16)
        plt.ylabel("prob" + objective, fontsize = 16)
        plt.title("Necessary " + helptype + " to reach objective", fontsize = 24, pad = 10)
        plt.xscale("log", base = 4)
        fig.savefig(folder_path + "/Nec" + helptype.capitalize() 
                    + "Log_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        plt.close() 
        
        dictData = {"final_rho": rho,
                    "rhos_tried_order": rhos_tried_order,
                    "rhos_tried": rhos_tried,
                    "crop_allocs": crop_allocs,
                    "probabilities": probabilities,
                    "necessary_help": necessary_help}
        
        with open(folder_path + "/DictData_" + file + ".txt", "wb") as fp:    
             pickle.dump(dictData, fp)
        
    return(rho, meta_sol_out)


# %% ######################### AUXILIARY FUNCTIONS ############################

def GetInitialGuess(dictGuesses, name, N):
    """
    Checks if same settings have already been run for a lower sample size. If
    so, the penalties of that run can be use as an initial guess to calculate
    the correct penalties for the current run.

    Parameters
    ----------
    dictGuesses : dict
        Dictionary of all settings for which penalties have been calculated so 
        far and the corresponsing value of the penalty.
    name : str
        Combining all settings that influence the expected income, used to 
        save the result for further runs.
    N : int
        current sample size

    Returns
    -------
    rhoBest : float or None
        The value of the penalty for the same settings with a lower sample 
        size if existing, None else.
    checked : boolean
        If we have a rhoBest, which was already the correct rho for two 
        sample sizes N1 and N2 with N2 >= N1 and N > N1, then we assume the 
        guess to be already checked and return True, else False.

    """
    
    # initialize values
    Ns = []
    Files = []
    rhos = []
    rhoBest = None
    checked = False
    
    # check for cases with same settings but different N
    for file in dictGuesses.keys():
        if file.startswith(name + "N"):
            Ns.append(int(file[len(name)+1:]))
            Files.append(file)
            rhos.append(dictGuesses[file])
                
    # get rho from the case with the highest N
    if len(Files) != 0:
        Files = [f for _,f in sorted(zip(Ns, Files))]
        rhos = [rho for _,rho in sorted(zip(Ns, rhos))]
        Ns = sorted(Ns)
        rhoBest = rhos[-1]
    
    # if the difference between the lowest N and highest N giving rhoBest 
    # is big enough (at least double), we assume that the penalty won't chenge
    # anymore and we don't do any checks.
    for i in range(0, len(rhos)):
        if rhos[i] == rhoBest:
            if (Ns[-1]/Ns[i] >= 2) and (Ns[i] < N):
                checked = True
            break
        
    return(rhoBest, checked)




def UpdatedRhoGuess(rhoLastUp, 
                    rhoLastDown, 
                    rhoOld, 
                    meta_sol = None,
                    prob = None, 
                    objective = None,
                    accuracy = None,
                    crop_alloc = None,
                    crop_alloc_conv = None,
                    max_areas = None):
    """
    For GetRhoF and GetRhoS (which have the same structure), this provides
    the next guess for the penalty.

    Parameters
    ----------
    meta_sol : dict
        Dictionary of meta information to the optimal crop allocations for the 
        last penalty guess.
    rhoLastUp : float
        The last (and thereby highest) penalty guess for which the probability 
        was too low.
    rhoLastDown : float
        The last (and thereby lowest) penalty guess for which the probability 
        was too high (or exactly right as we are searching for the lowest 
        penalty giving the right probability).
    rhoOld : float
        The last penalty that we tried.
    prob : float
        The probability for which we aim.
    accuracy : int
        Desired decimal places of accuracy of the obtained probability. 
    objective : string, "F" or "S"
        Specifies whether the function is called to find the next guess for 
        rhoS or for rhoF.
    max_areas : np.array
        Maximum available agricultural area per cluster

    Returns
    -------
    rhoNew : float
        The next penalty guess.
    rhoLastDown : float
        Updated version of rhoLastDown
    rhoLastUp : float
        Updated version of rhoFLastUp

    """
    
    if prob is not None:
        # specifiy which probability to use
        if objective == "F":
            currentProb = meta_sol["probF"]
        elif objective == "S":
            currentProb = meta_sol["probS"]
        # find next guess
        if np.round(currentProb, accuracy) < prob:
            rhoLastUp = rhoOld
            if rhoLastDown == np.inf:
                rhoNew = rhoOld * 4
            else:
                rhoNew = (rhoOld + rhoLastDown) / 2 
        else:
            rhoLastDown = rhoOld
            if rhoLastUp == 0:
                rhoNew = rhoOld / 4
            else:
                rhoNew = (rhoOld + rhoLastUp) / 2    
        
    elif crop_alloc_conv is not None:
        from ModelCode.GeneralSettings import accuracy_areas
        # find next guess
        tol = max_areas * accuracy_areas
        s = np.sum(np.abs(crop_alloc - crop_alloc_conv) > tol)
        if s > 0:
            rhoLastUp = rhoOld
            if rhoLastDown == np.inf:
                rhoNew = rhoOld * 4
            else:
                rhoNew = (rhoOld + rhoLastDown) / 2 
        elif s == 0:
            rhoLastDown = rhoOld
            if rhoLastUp == 0:
                rhoNew = rhoOld / 4
            else:
                rhoNew = (rhoOld + rhoLastUp) / 2
    
    return(rhoNew, rhoLastDown, rhoLastUp)


def checkIniGuess(rhoIni, 
                  args,
                  checkedGuess,
                  prob = None,
                  area = None,
                  objective = "F",
                  shareDiff = None,
                  accuracy = None,
                  accuracy_areas = None,
                  console_output = None,
                  logs_on = None):
    
    if rhoIni is not None:
        
        if objective  == "F":
            rhoFguess = rhoIni
            rhoFcheck =  rhoIni - rhoIni/shareDiff
            rhoSguess = 0
            rhoScheck = 0
            rhoCheck = rhoFcheck
        elif objective  == "S":
            rhoSguess = rhoIni
            rhoScheck =  rhoIni - rhoIni/shareDiff
            rhoFguess = 0
            rhoFcheck = 0
            rhoCheck = rhoScheck
            
        if prob is not None:
            method = "prob"
        elif area is not None:
            method = "area"
      
        def __test(crop_alloc, meta_sol, method = method, area = area, prob = prob, objective = objective):
            if method == "prob":
                tmp = "prob" + objective
                testPassed = (np.round(meta_sol[tmp], accuracy) == prob)
            elif method == "area":
                tol = args["max_areas"] * accuracy_areas
                s = np.sum(np.abs(crop_alloc - area) > tol)
                testPassed = (s == 0)
            return(testPassed)
  
        # check if rhoF from run with smaller N works here as well:
        printing("     Checking guess from run with other N", console_output = console_output, logs_on = logs_on)
        status, crop_alloc, meta_sol, sto_prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFguess, rhoSguess, console_output = False, logs_on = False) 
        testPassed = __test(crop_alloc, meta_sol)
        ReportProgressFindingRho(rhoIni, meta_sol, accuracy, durations, \
                                 objective, method, testPassed, prefix = "Guess: ", console_output = console_output, \
                                 logs_on = logs_on)
        if checkedGuess:
            printing("     We have a rhoF from a different N that was already double-checked!", console_output = console_output, logs_on = logs_on)
            return(rhoIni, crop_alloc, meta_sol)
        elif testPassed:    
            status, crop_alloc_check, meta_sol_check, sto_prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, rhoFcheck, rhoScheck,
                                                       console_output = False, logs_on = False) 
            testPassed = __test(crop_alloc_check, meta_sol_check)
            ReportProgressFindingRho(rhoCheck, meta_sol_check, accuracy, durations, \
                                objective, method, testPassed, prefix = "Check: ", console_output = console_output, \
                                 logs_on = logs_on)
            if not testPassed:
                printing("     Cool, that worked!", console_output = console_output, logs_on = logs_on)
                return(rhoIni, crop_alloc, meta_sol)
        printing("     Oops, that guess didn't work - starting from scratch\n", \
                 console_output = console_output, logs_on = logs_on)
            
    return(None, None, None)
  
    
def ReportProgressFindingRho(rho,
                            meta_sol, 
                            accuracy,
                            durations, 
                            objective,
                            method = "prob",
                            testPassed = False,
                            accuracy_int = False, 
                            prefix = "", 
                            console_output = None,
                            logs_on = None):
   
    """
    Function to report progress in finding the correct pealty to the console.

    Parameters
    ----------
    rhoOld : float
        Last penalty that was tried.
    meta_sol : dict
        Meta information of the model output for the last penalty that was 
        tried.
    accuracy : int
        Desired decimal places of accuracy of the obtained probability. 
    durations : list
        Time that was needed for setting up the model, for solving the model,
        and total time used (in sec.)
    objective : string,F" or "S"
        Specifies whether the function while searching for rhoF or rhoS.
    accuracy_int : float or False, optional
        Size of the current interval for which we know that the correct 
        penalty has to be within it. If false, the interval size will not be
        reported to the console. The default is False.
    debt : float or False
        Necessary debt for the government being able to provide payouts in 
        probS of the samples. Only relevant when called from 
        MinimizeNecessaryDebt. If False the debt is not reported. 
        The default is False.
    imports : float or False
        Necessary import to cover food demand in probF of the cases. Only 
        relevant when called from MinimizeNecessaryImport. If False the import
        is not reported.         
    prefix : float or False
        Used for additional information before the rest of the text. Used e.g. 
        when there are two next guesses when searching for the correct rhoS 
        within MinimizeNecessaryDebt. The default is "".
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs.If None, the default as defined in ModelCode/GeneralSettings
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    None.

    """
    # get correct probability and unit
    if objective == "F":
        currentProb = meta_sol["probF"]
        unit = " $/10^3kcal"
    elif objective == "S":
        currentProb = meta_sol["probS"]
        unit = " $/$"
        
    # if length of accuracy interval is given create corresponding text piece
    if accuracy_int:
        accuracy_text = " (current accuracy interval: " + str(np.round(accuracy_int, 3)) + ")"
    else:
        accuracy_text = ""
        
    # if we are trying to find the lowest penatly that gives the right crop areas
    if method == "area":
        if testPassed:
            ra_text = ", final areas"
        else:
            ra_text = ", diff. areas"
    else:
        ra_text = ""
        
    # print information (if console_output = True)
    printing("     " + prefix + "rho" + objective + ": " + str(rho) + unit + \
          ", prob" + objective + ": " + str(np.round(currentProb * 100, \
                                                    accuracy -1)) + \
          "%" + ra_text + ", time: " + str(np.round(durations[2], 2)) + "s" + accuracy_text, \
              console_output = console_output, logs_on = logs_on)
        
    return(None)

def LoadPenaltyStuff(objective, **kwargs):
    settings = DefaultSettingsExcept(**kwargs)
    if objective == "F":
        file = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Yield" + str(settings["yield_projection"]).capitalize() + \
                "Start" + str(settings["sim_start"]) + \
                "Pop" + str(settings["pop_scenario"]).capitalize() + \
                "T" + str(settings["T"]) + \
                "ProbF" + str(settings["probF"]) + \
                "N" + str(settings["N"])    
    elif objective == "S":
        file = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Yield" + str(settings["yield_projection"]).capitalize() + \
                "Start" + str(settings["sim_start"]) + \
                "Pop" + str(settings["pop_scenario"]).capitalize() +  \
                "T" + str(settings["T"]) + \
                "Risk" + str(settings["risk"]) + \
                "Tax" + str(settings["tax"]) + \
                "PercGuar" + str(settings["perc_guaranteed"]) + \
                "ProbS" + str(settings["probS"]) + \
                "N" + str(settings["N"])
                    
    folder_path = "Figures/GetPenaltyFigures/rho" + objective + "/" + file
    
    file_short = file.split("Crop")[0]
    
    with open(folder_path + "/DictData_" + file_short + ".txt", "rb") as fp:    
        dictData = pickle.load(fp)
        
    final_rho = dictData["final_rho"]
    rhos_tried_order = dictData["rhos_tried_order"]
    rhos_tried = dictData["rhos_tried"]
    crop_allocs = dictData["crop_allocs"]
    probabilities = dictData["probabilities"]
    necessary_help = dictData["necessary_help"]
    
    return(final_rho, rhos_tried_order, rhos_tried, crop_allocs,
           probabilities, necessary_help, file, objective)