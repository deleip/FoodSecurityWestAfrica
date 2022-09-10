#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:05:11 2021

@author: Debbora Leip
"""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from ModelCode.Auxiliary import _printing
from ModelCode.ModelCore import SolveReducedLinearProblemGurobiPy
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.MetaInformation import GetMetaInformation

# %% ########################## WRAPPING FUNCTIONS ############################
    

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
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.
        
    Returns
    -------
    rhoF : float
        The correct penalty rhoF to reach the probability probF (or the highest
        possible probF).
    rhoS : float
        The correct penalty rhoF to reach the probability probS (or the highest
        possible probS).
    meta_solF: dict
        Additional information on the model results when running only for rhoF
        (with rhoS = 0).
    meta_solS: dict
        Additional information on the model results when running only for rhoS 
        (with rhoF = 0).
    crop_allocF: np.array
        Optimal crop allocation when only using rhoF (with rhoS = 0).
    crop_allocS: np.array
        Optimal crop allocation when only using rhoS (with rhoF = 0).
    """
    # extract some settings (that were originally passed on directly...)
    probF = args["probF"]
    probS = args["probS"]
    rhoFini = settings["rhoF"]
    rhoSini = settings["rhoS"]
    
    if probF == 0:
        rhoF = 0
        meta_solF = None
        crop_allocF = None
        methodF = "notApplicable"
    else:   
        
        accuracy_probF = settings["accuracyF_demandedProb"]
        accuracy_maxProbF = settings["accuracyF_maxProb"]
        accuracy_rhoF = settings["accuracyF_rho"]
        accuracy_helpF = settings["accuracy_help"]
        
        # all settings that affect the calculation of rhoF
        SettingsBasics = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Yield" + str(settings["yield_projection"]).capitalize() + \
                "Start" + str(settings["sim_start"]) + \
                "Pop" + str(settings["pop_scenario"]).capitalize() + \
                "T" + str(settings["T"]) + \
                "Import" + str(settings["import"]) + \
                "Seed" + str(settings["seed"])
        SettingsFirstGuess =  SettingsBasics + "ProbF" + str(probF)
        SettingsProbF = SettingsBasics + "N" + str(settings["N"])
        SettingsFinalRhoF = SettingsFirstGuess + \
                    "N" + str(settings["N"]) + \
                    "AccFtarg" + str(accuracy_probF * 1000) + \
                    "AccFmax" + str(accuracy_maxProbF * 1000) + \
                    "AccFrho" + str(accuracy_rhoF * 1000) + \
                    "AccFhelp" + str(accuracy_helpF * 1000)
                        
        # get dictionary of settings for which rhoF has been calculated already
        with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
            dict_rhoFs = pickle.load(fp)
        with open("PenaltiesAndIncome/crop_allocF.txt", "rb") as fp:    
            dict_crop_allocF = pickle.load(fp)
        with open("PenaltiesAndIncome/methodsF.txt", "rb") as fp:    
            dict_methodsF = pickle.load(fp)
            
        # if this setting was already calculated, fetch rhoF
        if SettingsFinalRhoF in dict_rhoFs.keys():
            rhoF = dict_rhoFs[SettingsFinalRhoF]
            _printing("\nFetching rhoF: " + str(rhoF), console_output = console_output, logs_on = logs_on)
            crop_allocF = dict_crop_allocF[SettingsFinalRhoF]
            methodF = dict_methodsF[SettingsFinalRhoF]
            meta_solF = GetMetaInformation(crop_allocF, args, rhoF, 0)
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoF calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 1)
            if rhoFini is None:
                rhoFini, checkedGuess = _GetInitialGuess(dict_rhoFs, SettingsFirstGuess, settings["N"])
                
            # calculating rhoF
            _printing("Calculating rhoF and import", console_output = console_output, logs_on = logs_on)
            rhoF, meta_solF, crop_allocF, methodF = \
                _GetRhoWrapper(args, probF, rhoFini, checkedGuess, "F", SettingsProbF,
                  SettingsFinalRhoF, accuracy_probF, accuracy_maxProbF, accuracy_rhoF,
                  accuracy_helpF, console_output = None, logs_on = None)
            
            # get dictionaries again in case they changed in the meantime (from other model run)
            with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
                dict_rhoFs = pickle.load(fp)
            with open("PenaltiesAndIncome/crop_allocF.txt", "rb") as fp:    
                dict_crop_allocF = pickle.load(fp)
            
            dict_rhoFs[SettingsFinalRhoF] = rhoF
            dict_crop_allocF[SettingsFinalRhoF] = crop_allocF
            dict_methodsF[SettingsFinalRhoF] = methodF
        
        # saving updated dicts
        with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:    
             pickle.dump(dict_rhoFs, fp)
        with open("PenaltiesAndIncome/crop_allocF.txt", "wb") as fp:     
             pickle.dump(dict_crop_allocF, fp)
        with open("PenaltiesAndIncome/methodsF.txt", "wb") as fp:     
             pickle.dump(dict_methodsF, fp)
            
            
            
    if (settings["solv_const"] == "off" or probS == 0):
        rhoS = 0
        meta_solS = None
        crop_allocS = None
        methodS = "notApplicable"
    else:         
        
        accuracy_probS = settings["accuracyS_demandedProb"]
        accuracy_maxProbS = settings["accuracyS_maxProb"]
        accuracy_rhoS = settings["accuracyS_rho"]
        accuracy_helpS = settings["accuracy_help"]
        
        # all settings that affect the calculation of rhoS
        SettingsBasics = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Yield" + str(settings["yield_projection"]).capitalize() + \
                "Start" + str(settings["sim_start"]) + \
                "Pop" + str(settings["pop_scenario"]).capitalize() +  \
                "T" + str(settings["T"]) + \
                "IniFund" + str(settings["import"]) + \
                "Seed" + str(settings["seed"])
        SettingsBasics = SettingsBasics + \
                "Risk" + str(settings["risk"]) + \
                "Tax" + str(settings["tax"]) + \
                "PercGuar" + str(settings["perc_guaranteed"])
        SettingsFirstGuess = SettingsBasics + "ProbS" + str(probS)
        SettingsProbS = SettingsBasics + "N" + str(settings["N"])
        SettingsFinalRhoS = SettingsFirstGuess + \
                    "N" + str(settings["N"]) + \
                    "AccStarg" + str(accuracy_probS * 1000) + \
                    "AccSmax" + str(accuracy_maxProbS * 1000) + \
                    "AccSrho" + str(accuracy_rhoS * 1000) + \
                    "AccShelp" + str(accuracy_helpS * 1000)
                     
        # get dictionary of settings for which rhoS has been calculated already
        with open("PenaltiesAndIncome/RhoSs.txt", "rb") as fp:    
            dict_rhoSs = pickle.load(fp)
        with open("PenaltiesAndIncome/crop_allocS.txt", "rb") as fp:    
            dict_crop_allocS = pickle.load(fp)
        with open("PenaltiesAndIncome/methodsS.txt", "rb") as fp:    
            dict_methodsS = pickle.load(fp)
           
        # if this setting was already calculated, fetch rhoS
        if SettingsFinalRhoS in dict_rhoSs.keys():
            rhoS = dict_rhoSs[SettingsFinalRhoS]
            _printing("\nFetching rhoS: " + str(rhoS), console_output = console_output, logs_on = logs_on)
            crop_allocS = dict_crop_allocS[SettingsFinalRhoS]    
            methodS = dict_methodsS[SettingsFinalRhoS]
            meta_solS = GetMetaInformation(crop_allocS, args, 0, rhoS)
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoS calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 100)
            if rhoSini is None:
                rhoSini, checkedGuess = _GetInitialGuess(dict_rhoSs, SettingsFirstGuess, settings["N"])
                
            # calculating rhoS
            _printing("\nCalculating rhoS", console_output = console_output, logs_on = logs_on)
            rhoS, meta_solS, crop_allocS, methodS = \
                _GetRhoWrapper(args, probS, rhoSini, checkedGuess, "S", SettingsProbS,
                  SettingsFinalRhoS, accuracy_probS, accuracy_maxProbS, accuracy_rhoS,
                  accuracy_helpS, console_output = None, logs_on = None)
   
            # get dictionaries again in case they changed in the meantime (from other model run)
            with open("PenaltiesAndIncome/RhoSs.txt", "rb") as fp:    
                dict_rhoSs = pickle.load(fp)
            with open("PenaltiesAndIncome/crop_allocS.txt", "rb") as fp:    
                dict_crop_allocS = pickle.load(fp)
                
            dict_rhoSs[SettingsFinalRhoS] = rhoS
            dict_crop_allocS[SettingsFinalRhoS] = crop_allocS
            dict_methodsS[SettingsFinalRhoS] = methodS
        
        # saving updated dict
        with open("PenaltiesAndIncome/RhoSs.txt", "wb") as fp:    
             pickle.dump(dict_rhoSs, fp)
        with open("PenaltiesAndIncome/crop_allocS.txt", "wb") as fp:     
             pickle.dump(dict_crop_allocS, fp)
        with open("PenaltiesAndIncome/methodsS.txt", "wb") as fp:     
             pickle.dump(dict_methodsS, fp)
             
    return(rhoF, rhoS, meta_solF, meta_solS, crop_allocF, crop_allocS, methodF, methodS)

def _GetRhoWrapper(args, prob, rhoIni, checkedGuess, objective, fileProb,
                  fileFinalRho, accuracy_prob, accuracy_maxProb, accuracy_rho,
                  accuracy_help, console_output = None, logs_on = None):
    """
    Finding the correct rhoS given the probability probS, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    prob : float
        demanded probability of keeping the food security constraint or the
        solvency constraint (depending on objective).
    rhoIni : float or None 
        Initial guess for rhoF or rhoS (depending on objective).
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed.
    objective : "F" or "S"
        Specifying whether we are looking for rhoF or rhoS
    fileProb : str
        String combining all settings affecting the resulting probability based
        on a given penalty.
    fileFinalRho : str
        String combining all settings affecting the final rho, used to save plots 
        generated during the calculation of the penalties. 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.
    accuracy_prob : float
        Accuracy demanded from the food demand probability as share of demanded
        probability (for probability method).
    accuracy_maxProb : float
        Accuracy demanded from the food demand probability as share of maximum
        probability (for maxProb method).
    accuracy_rho : float
        Accuracy of the food security penalty given thorugh size of the accuracy
        interval: the size needs to be smaller than final rhoF * accuracyF_rho.
    accuracy_help : float, optional
        If method "MinHelp" is used to find the correct penalty, this defines the 
        accuracy demanded from the resulting necessary help in terms distance
        to the minimal necessary help, given this should be the accuracy demanded from the 
        final average necessary help (given as share of the difference between 
        final necessary help and the minimum nevessary help).

    Returns
    -------
    rho : float
        The final penalty
    meta_sol : dict
        Dictionary on model outputs for the resulting penalty.
    crop_alloc : np.array
        Optimal crop allocation for the resulting penalty.
    """
    
    def _get_necessary_help(meta_sol, objective = objective):
        if objective == "F":
            return(meta_sol["avg_nec_import"])
        elif objective == "S":
            return(meta_sol["avg_nec_debt"])
    
    # check model output for a very high penalty (as proxy for infinity)
    maxProb, meta_max, meta_zero, max_crop_alloc = _CheckOptimalProb(args, prob, objective, fileProb,
                      console_output = console_output, logs_on = logs_on)
    nec_help = _get_necessary_help(meta_max)    
    
    # if probF can be reached find lowest penalty that gives probF
    if maxProb >= prob:
        _printing("     Finding corresponding penalty\n", console_output, logs_on = logs_on)
        rho, meta_sol, crop_alloc = _RhoProbability(args, prob, rhoIni, checkedGuess, \
               objective, fileProb, fileFinalRho, accuracy_rho, accuracy_prob, accuracy_help, 
               nec_help, "GivenProb", meta_zero, meta_max, max_crop_alloc, console_output, logs_on)
        method = "GivenProb"
    # if probF cannot be reached but the maximum probability (or what is 
    # assumed to be the maximum probability) is higher than for rho -> 0,
    # find the lowest penalty that gives the highest probability
     # rounding to avoid using this method for max probab super close to zero
     # (which would be only due to outlier high yields)
    elif round(maxProb, 4) > meta_zero["prob" + objective]:
        _printing("     Finding penalty that leads to max. probability\n", console_output, logs_on = logs_on)
        rho, meta_sol, crop_alloc = _RhoProbability(args, maxProb, rhoIni, checkedGuess, \
               objective, fileProb, fileFinalRho, accuracy_rho, accuracy_maxProb, accuracy_help,
               nec_help, "MaxProb", meta_zero, meta_max, max_crop_alloc, console_output, logs_on)
        method = "MaxProb"
    # if the max. probability is zero, find the lowest penalty that minimizes the
    # average import/debt that is needed to cover food demand/government payouts
    else:
        if objective == "F": 
            _printing(f"     Finding lowest penalty minimizing average total import ({nec_help:.2e} 10^12 kcal)\n", 
                     console_output, logs_on = logs_on)
        elif objective == "S":
            _printing(f"     Finding lowest penalty minimizing average total debt ({nec_help:.2e} 10^9 $)\n", 
                     console_output, logs_on = logs_on)
        rho, meta_sol, crop_alloc = _RhoMinHelp(args, prob, rhoIni,
                checkedGuess, objective, nec_help, accuracy_rho, accuracy_help,
                meta_zero, meta_max, max_crop_alloc, fileProb, fileFinalRho, \
                console_output = console_output, logs_on = logs_on)
        method = "MinHelp"
            
    _printing("\n     Final rho" + objective + ": " + str(rho), console_output = console_output, logs_on = logs_on)
    
    return(rho, meta_sol, crop_alloc, method)

    
# %% ##################### TWO PENALTY CALCULATION ALGORITHMS ######################
 
def _RhoProbability(args, prob, rhoIni, checkedGuess, objective, fileProb,
                    fileFinalRho, accuracy_rho, accuracy, accuracy_help_perc, 
                    nec_help, method, meta_zero, meta_max, max_crop_alloc, 
                    console_output = None, logs_on = None):
    """
    Finding the correct rho given the probability prob, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    prob : float
        The target probability for food security/solvency.
    rhoIni : float or None 
        Initial guess for the penalty.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed
    objective : "F" or "S"
        Specifying whether we are looking for rhoF or rhoS
    fileProb : str
        String combining all settings affecting the resulting probability based
        on a given penalty.
    fileFinalRho : str
        String combining all settings affecting the final rho, used to save plots 
        generated during the calculation of the penalties. 
    accuracy_rho : float
        The share of the final rho that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < accuracy_rho * rho, for rho
        the current best guess for the correct penalty, then we use rho).
    accuracy : float
        Accuracy of final probability as share of target - min probability.
    accuracy_help_perc : float
        Accuracy demanded from the resulting necessary help in terms distance
        to the minimal necessary help.
    nec_help: float
        The minimum average necessary import/debt (as calculated for the case 
        with a very high penalty).   
    method : str
        Whether the algorithm is called for them externally demanded probability
        ("GivenProb"), or for the maximum possible probability ("MaxProb") in 
        cases where the externally given probabiltiy cannot be reached
    meta_zero : float
        Meta information from running the model for rhoF = rhoS = 0.
    meta_max : float
        Meta information for running model with very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
    max_crop_alloc : np.array
        Crop area alocation resulting from very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    rho : float
        The final penalty
    meta_sol_out : dict
        Dictionary on model outputs for the resulting penalty.
    crop_alloc_out : np.array
        Optimal crop allocation for the resulting penalty.
    """
    
    min_prob = meta_zero["prob"+objective]
    tol_prob = accuracy * (prob - min_prob)
    
    def _setGuesses(nextGuess, objective = objective):
        if objective == "F":
            rhoF = nextGuess
            rhoS = 0
        elif objective == "S":
            rhoS = nextGuess
            rhoF = 0
        return(rhoF, rhoS)
    
    # def _get_necessary_help(meta_sol, objective = objective):
    #     if objective == "F":
    #         return(meta_sol["avg_nec_import"])
    #     elif objective == "S":
    #         return(meta_sol["avg_nec_debt"])
   
    def _testProbAccuracy(currentProb, prob = prob, tol_prob = tol_prob):
        testPassed = (abs(currentProb - prob) < tol_prob)
        return(testPassed)
   
    # accuracy information
    _printing("     accuracy that we demand for prob" + objective + ": " + \
             str(round(accuracy*100,1)) + "% of target - min probability", \
             console_output = console_output, logs_on = logs_on)
    _printing("     accuracy that we demand for rho" + objective + ": " + \
             str(round(accuracy_rho*100,1)) + "% of final rho" + objective + "\n", \
             console_output = console_output, logs_on = logs_on)
    
    # check if rho from run with smaller N works here as well:
    # if we get the right prob for our guess, and a lower prob for rhoCheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rho is in that interval and can return our guess
    rho, meta_sol, crop_alloc = _checkIniGuess(rhoIni, 
                                  args,
                                  checkedGuess,
                                  fileProb,
                                  prob = prob,
                                  min_prob = min_prob,
                                  objective = objective,
                                  accuracy_rho = accuracy_rho,
                                  accuracy = accuracy,
                                  console_output = console_output,
                                  logs_on = logs_on)
    if rho is not None:
        return(rho, meta_sol, crop_alloc)
    
    # else we start from scratch
    if objective == "F":
        rhoIni = 1
    elif objective == "S":
        rhoIni = 100
    rhoFini, rhoSini = _setGuesses(rhoIni, objective)
    
    # initialize values for search algorithm and reporting
    rhoLastDown = np.inf
    rhoLastUp = 0
    lowestCorrect = np.inf
    meta_sol_lowestCorrect = np.inf
    crop_alloc_lowestCorrest = np.inf
    crop_allocs = []
    rhos_tried = []
    probabilities = []
    necessary_help = []
    
    # calculate initial guess
    crop_alloc, currentProb, currentNecHelp, durations, meta_sol = \
        _getResultsForRho(args, rhoFini, rhoSini, objective, fileProb, console_output = False, logs_on = False)
    crop_allocs.append(crop_alloc)
    rhos_tried.append(rhoIni)
    probabilities.append(currentProb)
    necessary_help.append(currentNecHelp)
    # status, crop_alloc, meta_sol, sto_prob, durations = \
    #             SolveReducedLinearProblemGurobiPy(args, rhoFini, rhoSini, console_output = False, logs_on = False)
    # crop_allocs.append(crop_alloc)
    # rhos_tried.append(rhoIni)
    # probabilities.append(meta_sol["prob"+objective])
    # necessary_help.append(_get_necessary_help(meta_sol))
    
    # update information
    if _testProbAccuracy(currentProb):
        lowestCorrect = rhoIni
        meta_sol_lowestCorrect = meta_sol
        crop_alloc_lowestCorrest = crop_alloc
                
    # remember guess
    rhoOld = rhoIni
    
    # report
    accuracy_int = lowestCorrect - rhoLastUp
    _ReportProgressFindingRho(rhoOld, currentProb, currentNecHelp, durations, \
                             objective, accuracy_int = accuracy_int,
                             console_output = console_output, logs_on = logs_on)

    while True:   
        # find next guess
        rhoNew, rhoLastDown, rhoLastUp = \
                    _UpdatedRhoGuess(rhoLastUp, rhoLastDown, rhoOld, 
                                     currentProb = currentProb, currentNecHelp = currentNecHelp,
                                     prob = prob, min_prob = min_prob, objective = objective,
                                     accuracy = accuracy)
        rhoFnew, rhoSnew = _setGuesses(rhoNew, objective)
       
        # solve model for guess
        crop_alloc, currentProb, currentNecHelp, durations, meta_sol = \
            _getResultsForRho(args, rhoFnew, rhoSnew, objective, fileProb, console_output = False, logs_on = False)
        crop_allocs.append(crop_alloc)
        rhos_tried.append(rhoNew)
        probabilities.append(currentProb)
        necessary_help.append(currentNecHelp)
        # status, crop_alloc, meta_sol, sto_prob, durations = \
        #         SolveReducedLinearProblemGurobiPy(args, rhoFnew, rhoSnew, console_output = False, logs_on = False)
        # crop_allocs.append(crop_alloc)
        # rhos_tried.append(rhoNew)
        # probabilities.append(meta_sol["prob"+objective])
        # necessary_help.append(_get_necessary_help(meta_sol))
        
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalty we have
        # reached the necessary accuracy.
        if _testProbAccuracy(currentProb):
            accuracy_int = rhoNew - rhoLastUp
            if accuracy_int < rhoNew * accuracy_rho:    
                rho = rhoNew
                meta_sol_out = meta_sol
                crop_alloc_out = crop_alloc
                break
        elif currentProb < prob:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoNew
                if accuracy_int < lowestCorrect * accuracy_rho:
                    rho = lowestCorrect
                    meta_sol_out = meta_sol_lowestCorrect
                    crop_alloc_out = crop_alloc_lowestCorrest
                    break
            else:
                accuracy_int = rhoLastDown - rhoNew
        elif currentProb > prob:
            accuracy_int = rhoNew - rhoLastUp
            
        # report
        _ReportProgressFindingRho(rhoNew, currentProb, currentNecHelp, durations, \
                                 objective, accuracy_int = accuracy_int, 
                                 console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoOld = rhoNew
        if _testProbAccuracy(currentProb) and lowestCorrect > rhoNew:
            lowestCorrect = rhoNew
            meta_sol_lowestCorrect = meta_sol
            crop_alloc_lowestCorrest = crop_alloc
    
    # last report
    _ReportProgressFindingRho(rhoNew, currentProb, currentNecHelp, durations, \
               objective, accuracy_int = accuracy_int, console_output = console_output, 
               logs_on = logs_on)    
   
    # ploting of information
    if args["T"] > 1:
        if objective == "F":
            nec_help_zero = meta_zero["avg_nec_import"]
            prob_zero = meta_zero["probF"]
        elif objective == "S":
            nec_help_zero = meta_zero["avg_nec_debt"]
            prob_zero = meta_zero["probS"]
        accuracy_help = accuracy_help_perc * (nec_help_zero - nec_help)
        
        _PlotPenatlyStuff(rho, rhos_tried, crop_allocs, probabilities, \
                         necessary_help, objective, method, fileFinalRho,
                         nec_help, nec_help_zero, accuracy_help, prob_zero,
                         meta_max, max_crop_alloc, prob, tol_prob)
    
    return(rho, meta_sol_out, crop_alloc_out)

         
def _RhoMinHelp(args, prob, rhoIni, checkedGuess, objective, \
                nec_help, accuracy_rho, accuracy_help_perc, meta_zero, meta_max, \
                max_crop_alloc, fileProb, fileFinalRho, console_output = None, \
                logs_on = None):
    """
    In cases where the given probability cannot be reached, we instead look
    for the lowest penalty that minimizes the average necessary import/debt 
    to cover the food demand/the government payouts.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    prob : float
        The desired probability for food security/solvency.
    rhoIni : float or None 
        Initial guess for the penalty.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed
    objective : "F" or "S"
        Specifying whether we are looking for rhoF or rhoS
    nec_help: float
        The minimum average necessary import/debt (as calculated for the case 
        with a very high penalty).    
    accuracy_rho : float
        The share of the final rho that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < accuracy_rho * rho, for rho
        the current best guess for the correct penalty, then we use rho).
    accuracy_help_perc : float
        Accuracy demanded from the resulting necessary help in terms distance
        to the minimal necessary help.
    meta_zero : float
        Meta information from running the model for rhoF = rhoS = 0.
    meta_max : float
        Meta information for running model with very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
    max_crop_alloc : np.array
        Crop area alocation resulting from very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
    fileProb : str
        String combining all settings affecting the resulting probability based
        on a given penalty.
    fileFinalRho : str
        String combining all settings affecting the final rho, used to save plots 
        generated during the calculation of the penalties. 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    rho : float
        The final penalty
    meta_sol_out : dict
        Dictionary on model outputs for the resulting penalty.
    crop_alloc_out : np.array
        Optimal crop allocation for the resulting penalty.
    """   
    
    # accuracy information    
    if objective == "F":
        help_type = "import"
        nec_help_zero = meta_zero["avg_nec_import"]
        prob_zero = meta_zero["probF"]
    elif objective == "S":
        help_type = "debt"
        nec_help_zero = meta_zero["avg_nec_debt"]
        prob_zero = meta_zero["probS"]
        
    accuracy_help = accuracy_help_perc * (nec_help_zero - nec_help)
            
    _printing("     accuracy that we demand for necessary " + help_type + ": " + \
             str(round(accuracy_help_perc * 100, 1)) + \
                 "% of diff between maximal and minimal necessary " + help_type, \
             console_output = console_output, logs_on = logs_on)
    _printing("     accuracy that we demand for rho" + objective + ": " + 
              str(round(accuracy_rho * 100, 1)) + \
             "% of final rho" + objective + "\n", \
             console_output = console_output, logs_on = logs_on)
        
    # internal functions
    def _setGuesses(nextGuess, objective = objective):
        if objective == "F":
            rhoF = nextGuess
            rhoS = 0
        elif objective == "S":
            rhoS = nextGuess
            rhoF = 0
        return(rhoF, rhoS)
    
    def _rightHelp(currentNecHelp, nec_help = nec_help):
        testPassed = (np.abs(currentNecHelp - nec_help) < accuracy_help)
        return(testPassed)
    
    # check if rho from run with smaller N works here as well:
    # rho, meta_sol, crop_alloc = _checkIniGuess(rhoIni, 
    #                                         args,
    #                                         checkedGuess,
    #                                         fileProb,  
    #                                         nec_help = nec_help,
    #                                         objective = objective,
    #                                         accuracy_rho = accuracy_rho,
    #                                         accuracy = None,
    #                                         accuracy_help = accuracy_help,
    #                                         console_output = console_output,
    #                                         logs_on = logs_on)
    # if rho is not None:
    #     return(rho, meta_sol, crop_alloc)
    ## WE DON'T USE RHOS FROM LOWER N FOR THE NECESSARY HELP METHOD, AS THIS 
    ## APPARENTLY IS TOO SENSIBLE (??)
   
    # else we start from scratch
    if objective == "F":
        rhoIni = 1
    elif objective == "S":
        rhoIni = 100
    rhoFini, rhoSini = _setGuesses(rhoIni, objective)
    
    # initialize values for search algorithm and reporting
    rhoLastDown = np.inf
    rhoLastUp = 0
    lowestCorrect = np.inf
    meta_sol_lowestCorrect = np.inf
    crop_alloc_lowestCorrest = np.inf
    crop_allocs = []
    rhos_tried = []
    probabilities = []
    necessary_help = []
    
    # calculate initial guess
    crop_alloc, currentProb, currentNecHelp, durations, meta_sol = \
        _getResultsForRho(args, rhoFini, rhoSini, objective, fileProb, console_output = False, logs_on = False)
    # status, crop_alloc, meta_sol, sto_prob, durations = \
    #             SolveReducedLinearProblemGurobiPy(args, rhoFini, rhoSini, objective, \
    #                                               console_output = False, logs_on = False)
    crop_allocs.append(crop_alloc)
    rhos_tried.append(rhoIni)
    probabilities.append(currentProb)
    necessary_help.append(currentNecHelp)
    
    # update information
    testPassed = _rightHelp(currentNecHelp)
    if testPassed:
        lowestCorrect = rhoIni
        meta_sol_lowestCorrect = meta_sol
        crop_alloc_lowestCorrest = crop_alloc
                
    # remember guess
    rhoOld = rhoIni
    
    # report
    accuracy_int = lowestCorrect - rhoLastUp
    _ReportProgressFindingRho(rhoOld, currentProb, currentNecHelp, durations, \
                            objective, method = "nec_help", testPassed = testPassed, 
                            accuracy_int = accuracy_int, 
                            console_output = console_output, logs_on = logs_on)

    while True:   
        # find next guess
        rhoNew, rhoLastDown, rhoLastUp = _UpdatedRhoGuess(rhoLastUp, 
                     rhoLastDown, rhoOld, currentProb, currentNecHelp,
                     objective = objective, nec_help = nec_help,
                     accuracy_help = accuracy_help)
        rhoFnew, rhoSnew = _setGuesses(rhoNew, objective)
        
        # solve model for guess
        crop_alloc, currentProb, currentNecHelp, durations, meta_sol = \
            _getResultsForRho(args, rhoFnew, rhoSnew, objective, fileProb, console_output = False, logs_on = False)
        # status, crop_alloc, meta_sol, sto_prob, durations = \
        #         SolveReducedLinearProblemGurobiPy(args, rhoFnew, rhoSnew,
        #                              console_output = False, logs_on = False)
        crop_allocs.append(crop_alloc)
        rhos_tried.append(rhoNew)
        probabilities.append(currentProb)
        necessary_help.append(currentNecHelp)
        
        testPassed = _rightHelp(currentNecHelp)
        
       # Check if termination criteria are fulfilled
        if testPassed:
            accuracy_int = rhoNew - rhoLastUp
            if accuracy_int < rhoNew * accuracy_rho:
                rho = rhoNew
                meta_sol_out = meta_sol
                crop_alloc_out = crop_alloc
                break
        else:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoNew
                if accuracy_int < lowestCorrect * accuracy_rho:
                    rho = lowestCorrect
                    meta_sol_out = meta_sol_lowestCorrect
                    crop_alloc_out = crop_alloc_lowestCorrest
                    break
            else:
                accuracy_int = rhoLastDown - rhoNew
            
        # report
        _ReportProgressFindingRho(rhoNew, currentProb, currentNecHelp, durations, \
                             objective, method = "nec_help", testPassed = testPassed, 
                             accuracy_int = accuracy_int, console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoOld = rhoNew
        if  testPassed and lowestCorrect > rhoNew:
            lowestCorrect = rhoNew
            meta_sol_lowestCorrect = meta_sol
            crop_alloc_lowestCorrest = crop_alloc
    
    # last report
    _ReportProgressFindingRho(rhoNew, currentProb, currentNecHelp, durations, \
                         objective, method = "nec_help", testPassed = testPassed, 
                         accuracy_int = accuracy_int, console_output = console_output, logs_on = logs_on)
        
    # ploting of information
    if args["T"] > 1:
        _PlotPenatlyStuff(rho, rhos_tried, crop_allocs, probabilities, \
                         necessary_help, objective, "MinHelp", fileFinalRho,
                         nec_help, nec_help_zero, accuracy_help, prob_zero,
                         meta_max, max_crop_alloc)
                 
    return(rho, meta_sol_out, crop_alloc_out)


# %% ######################### AUXILIARY FUNCTIONS ############################

def _getResultsForRho(args, rhoF, rhoS, objective, fileProb,
                      console_output = False, logs_on = False):
    """
    Function to find the resulting probability and crop areas for the current
    guess.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    rhoF : float
        The current guess for rhoF.
    rhoS : float
        The current guess for rhoS.
    objective : "F" or "S"
        Specifying whether we are looking for rhoF or rhoS
    fileProb : str
        String combining all settings affecting the resulting probability based
        on a given penalty.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    crop_alloc : np.array
        Crop area allocation resulting from very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
    currentProb : float
        Resulting probability for food security or fund solvency (depending on
        objective).
    currentNecHelp : float
        Average necessary import or debt (depending on objective).
    durations : list
        Durations of solving the model (if it had to be run), or information
        that results were already available.
    """    
    
    def _get_necessary_help(meta_sol, objective = objective):
        if objective == "F":
            return(meta_sol["avg_nec_import"])
        elif objective == "S":
            return(meta_sol["avg_nec_debt"])
    
    if objective == "F":
        rho  = rhoF
    else:
        rho = rhoS
        
    fullFile = "PenaltiesAndIncome/Prob" + objective + "/" + fileProb + ".txt"
    
    if os.path.exists(fullFile):
        with open(fullFile, "rb") as fp:    
            dict_prob = pickle.load(fp)
    else:
        dict_prob = {}
            
    if rho in dict_prob.keys():
        crop_alloc  = dict_prob[rho]["crop_alloc"]
        currentProb = dict_prob[rho]["prob"]
        currentNecHelp = dict_prob[rho]["nec_help"]
        durations  = "pre-calculated"
        meta_sol   = GetMetaInformation(crop_alloc, args, rhoF, rhoS)
    else:
        status, crop_alloc, meta_sol, sto_prob, durations = \
            SolveReducedLinearProblemGurobiPy(args, rhoF, rhoS, console_output = False, logs_on = False)
        currentProb = meta_sol["prob"+objective]
        currentNecHelp = _get_necessary_help(meta_sol)
        
        dict_prob[rho] = {"crop_alloc": crop_alloc,
                          "prob"      : currentProb,
                          "nec_help"  : currentNecHelp}
        
        with open(fullFile, "wb") as fp:    
             pickle.dump(dict_prob, fp)
    
    return(crop_alloc, currentProb, currentNecHelp, durations, meta_sol)
    
    
def _CheckOptimalProb(args, prob, objective, fileProb,
                      console_output = None, logs_on = None):
    """
    Function to find the highest probF possible under the given settings, and
    calculating the amount of import needed to increase this probabtility to 
    the probF desired.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    prob : float
        The desired probability for food security/solvency.
    objective : "F" or "S"
        Specifying whether we are looking for rhoF or rhoS
    fileProb : str
        String combining all settings affecting the resulting probability based
        on a given penalty.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    maxProb : float
        Probability for food security/solvency resulting from running the model
        with a very high penalty rhoF/rhoS and setting the other penalty to zero.
    meta_max : float
        Meta information for running model with very high penalty rho (with other
        penalty kept zero)
    meta_zero : float
        Meta information from running the model for rhoF = rhoS = 0.
    crop_alloc : np.array
        Crop area allocation resulting from very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
        
    """
    
    if objective == "F":
        rhoF = 1e3
        rhoS = 0
    elif objective == "S":
        rhoS = 1e6
        rhoF = 0
        
    # try for rho = 1e12 (as a proxy for rho -> inf)
    crop_alloc, maxProb, minNecHelp, durations, meta_sol = \
        _getResultsForRho(args, rhoF, rhoS, objective, fileProb, console_output = False, logs_on = False)
    # status, crop_alloc, meta_max, sto_prob, durations = \
    #      SolveReducedLinearProblemGurobiPy(args, rhoF, rhoS, console_output = False, logs_on = False)  
     
    # try for rho = 0
    meta_max = GetMetaInformation(crop_alloc, args, rhoF, rhoS)
    meta_zero = GetMetaInformation(np.zeros((args["T"], args["num_crops"], len(args["k_using"]))), args, 0, 0)

    # get resulting probabilities
    minProb = meta_zero["prob" + objective]
    _printing("     maxProb" + objective + ": " + str(np.round(maxProb * 100, 2)) + "%, " + \
              "minProb" + objective + ": " + str(np.round(minProb * 100, 2)) + "%", \
              console_output = console_output, logs_on = logs_on)
    
    # check if probability is high enough 
    if maxProb >= prob:
        _printing("     Desired pro" + objective + " (" + str(np.round(prob * 100, 2)) \
                             + "%) can be reached\n", console_output = console_output, logs_on = logs_on)
    else:
        _printing("     Desired pro" + objective + " (" + str(np.round(prob * 100, 2)) \
                             + "%) cannot be reached\n", console_output = console_output, logs_on = logs_on)
            
    return(maxProb, meta_max, meta_zero, crop_alloc)

def _PlotPenatlyStuff(rho, rhos_tried, crop_allocs, probabilities, necessary_help, \
                     objective, method, file, min_nec_help, nec_help_zero,
                     accuracy_help, minProb, meta_max, max_crop_alloc,
                     target_prob = None, tol_prob = None):
    """
    The functions document the process of finding the right penalty (i.e. the
    necessary import/debt in each step, the probability in each step, ...). 
    This function creates plots to visualize this process.

    Parameters
    ----------
    rho : float
        Final penalty.
    rhos_tried : list of floats
        List of all penalties that were tried.
    crop_allocs : list of np.arrays
        List of resulting crop areas for each penalty that was tried.
    probabilities : list of floats
        List of resulting probabilities for each penalty that was tried.
    necessary_help : list of floats
        List of resulting necessary import/debt for each penalty that was tried.
    objective : "F" or "S"
        Specifying whether we are looking for rhoF or rhoS
    method : "GivenProb", "MaxProb", or "MinHelp"
        Which method was used to find the right penalty (depending on whether 
        the given probability could be reached or not).
    file : str
        String combining all settings affecting rho, used to save plots.
    min_nec_help : float
        The minimal possible average necessary help (as calculated for the case 
        with a very high penalty). 
    nec_help_zero : float
        The average necessary help needed if the penalty is set to zero.
    accuracy_help : float
        The accuracy demanded from the final average necessary help, given as 
        maximum absolute value of the difference between final necessary help 
        and the minimum nevessary help (only used as constraint if method is
        "MinHelp").
    minProb : float
        The resulting probability for rho = 0
    meta_max : float
        Meta information for running model with very high penalty rho (with other
        penalty kept zero)
    max_crop_alloc : np.array
        Crop area alocation resulting from very high penalty rho (with other
        penalty kept zero). Only needed to pass along to plotting.
    target_prob : float
        The target probability for food security/solvency if the _RhoProbability
        function is used. Default is None.
    tol_prob : float
        The tolerance for the final probability (as distance to taget 
        probability) if the _RhoProbability function is used. Default is None.

    Returns
    -------
    None.

    """
        
    from ModelCode.GeneralSettings import figsize
    
    if objective == "F":
        helptype = "import"
    elif objective == "S":
        helptype = "debt"   
        
    def _get_necessary_help(meta_sol, objective = objective):
        if objective == "F":
            return(meta_sol["avg_nec_import"])
        elif objective == "S":
            return(meta_sol["avg_nec_debt"])
        
    T = crop_allocs[0].shape[0]
        
    # setting up folder and file name
    folder_path = "Figures/GetPenaltyFigures/rho" + objective + "/" + method + "_" + file
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    file = file.split("Crop")[0]
    
    # sorting lists according to penalties (from low to high)
    s = sorted(zip(rhos_tried, probabilities, crop_allocs, necessary_help))
    rhos_tried_order = rhos_tried.copy()
    rhos_tried = [x for x,_,_,_ in s]
    probabilities = [x for _,x,_,_ in s]
    crop_allocs = [x for _,_,x,_ in s]
    necessary_help = [x for _,_,_,x in s]
    
    # calculate best grid size for subplots
    num_subplots = len(crop_allocs) + 1
    nrows = int(np.floor(np.sqrt(num_subplots)))
    if nrows * nrows >= num_subplots:
        ncols = nrows
    elif nrows * (nrows + 1) >= num_subplots:
        ncols = nrows + 1
    else:
        nrows = nrows + 1
        ncols = nrows
    
    # initialize colors (maxium needed are 9 as they correspond to the clusters)
    cols = ["royalblue", "darkred", "grey", "gold", \
                "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                "indigo"]
        
    # plot crop areas
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

    ax = fig.add_subplot(nrows, ncols, num_subplots)
    for cl in range(0, num_cluster):
        if cl == num_cluster -1:
            ax.plot(years, max_crop_alloc[:,0,cl], ls = "-", color = cols[cl], label = "Rice")
            ax.plot(years, max_crop_alloc[:,1,cl], ls = "--", color = cols[cl], label = "Maize")
        else:
            ax.plot(years, max_crop_alloc[:,0,cl], ls = "-", color = cols[cl])
            ax.plot(years, max_crop_alloc[:,1,cl], ls = "--", color = cols[cl])
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_title("rho" + objective + ": very high" +  
                 ", prob" + objective + ": " + str(round(meta_max["prob"+objective], 2)) + 
                 ", " + helptype + ": " + str(round(_get_necessary_help(meta_max), 2)), fontsize = 14)
    ax.set_xlabel("Years", fontsize = 14)
    if ((num_subplots - 1) % ncols) == 0:
        ax.set_ylabel("Crop areas", fontsize = 14)
    
    plt.legend()
    plt.suptitle("Crop areas for different penalties rho" + objective, fontsize = 22)
    fig.savefig(folder_path + "/CropAreas_" + file + ".jpg", \
                bbox_inches = "tight", pad_inches = 1)
    plt.close() 
    
    # add case with rho = 0 to lists 
    if nec_help_zero is not None:
        rhos_tried = [0] + rhos_tried
        probabilities = [minProb] + probabilities
        necessary_help = [nec_help_zero] + necessary_help
        idx_rho = idx_rho + 1
    
    # plot penalties vs probabilities
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,2,1)
    plt.scatter(rhos_tried, probabilities, color = "royalblue")
    if target_prob is not None:
        plt.fill_between([min(rhos_tried), max(rhos_tried)], 
                         [target_prob - tol_prob, target_prob - tol_prob], 
                         [target_prob + tol_prob, target_prob + tol_prob],
                         color = "green", alpha = 0.2)
    plt.scatter(rhos_tried[np.argmax(probabilities)], max(probabilities), color = "green")
    plt.scatter(rhos_tried[idx_rho], probabilities[idx_rho], color = "red")
    if target_prob is not None:
        plt.plot([min(rhos_tried), max(rhos_tried)], [target_prob, target_prob], color = "green")
    plt.xlabel("rho" + objective, fontsize = 16)
    plt.ylabel("prob" + objective, fontsize = 16)
    plt.title("Probability of meeting objective", fontsize = 24, pad = 10)
    # fig.savefig(folder_path + "/Prob_" + file + ".jpg", \
    #             bbox_inches = "tight", pad_inches = 1)
    # plt.close() 
        
    # plot penalties vs necessary help
    ax = fig.add_subplot(1,2,2)
    plt.scatter(rhos_tried, necessary_help, color = "royalblue")
    if nec_help_zero is not None:
        plt.fill_between([min(rhos_tried), max(rhos_tried)], 
                         [min_nec_help - accuracy_help, min_nec_help - accuracy_help], 
                         [min_nec_help + accuracy_help, min_nec_help + accuracy_help],
                         color = "green", alpha = 0.2)
    plt.scatter(rhos_tried[np.argmin(necessary_help)], min(necessary_help), color = "green")
    plt.scatter(rhos_tried[idx_rho], necessary_help[idx_rho], color = "red")
    plt.plot([min(rhos_tried), max(rhos_tried)], [min_nec_help, min_nec_help], color = "green")
    plt.xlabel("rho" + objective, fontsize = 16)
    plt.ylabel("necessary " + helptype, fontsize = 16)
    plt.title("Necessary " + helptype + " to reach objective", fontsize = 24, pad = 10)
    fig.savefig(folder_path + "/ProbAndNec" + helptype.capitalize() + 
                "_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    plt.close() 

    # save information on finding penalty as dict    
    dictData = {"final_rho": rho,
                "rhos_tried_order": rhos_tried_order,
                "rhos_tried": rhos_tried,
                "crop_allocs": crop_allocs,
                "probabilities": probabilities,
                "necessary_help": necessary_help,
                "min_nec_help": min_nec_help,
                "accuracy_help": accuracy_help,
                "meta_max": meta_max,
                "max_crop_alloc": max_crop_alloc,
                "target_prob": target_prob,
                "tol_prob": tol_prob}
    
    with open(folder_path + "/DictData_" + file + ".txt", "wb") as fp:    
         pickle.dump(dictData, fp)
    
    return(None)

def _GetInitialGuess(dictGuesses, name, N):
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
        Combining all settings that influence the penalty to a single string, 
        used as name in the dictionary.
    N : int
        current sample size.

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
        filestart = file.split("Acc")[0]
        if filestart.startswith(name + "N"):
            Ns.append(int(filestart[len(name)+1:]))
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

def _UpdatedRhoGuess(rhoLastUp, 
                    rhoLastDown, 
                    rhoOld, 
                    currentProb,
                    currentNecHelp,
                    prob = None, 
                    min_prob = None,
                    objective = None,
                    accuracy = None,
                    nec_help = None,
                    accuracy_help = None):
    """
    Provides the next guess in the search algorithms to find the penalty.

    Parameters
    ----------
    rhoLastUp : float
        The last (and thereby highest) penalty guess for which did not fulfill
        the criteria.
    rhoLastDown : float
        The last (and thereby lowest) penalty guess for which the the criteria
        was fulfilled (or exceeded in the case of a too high probability)
    rhoOld : float
        The last penalty that we tried.
    currentProb : float
        Probability resulting for current rho (either for food security or
        solvency, depending on objective).
    currentNecHelp : float
        Average necessary import or debt for current rho (depending on objective).
    prob : float, optional
        The probability for which we aim. Not needed if the MinHelp search
        algorithm is used. Default is None.
    min_prob : float, optional
        The probability that is reached without using any area. Not needed if
        the MinHelp search algorithm is used. Default is None.
    objective : string "F" or "S"
        Specifies whether the function is called to find the next guess for 
        rhoS or for rhoF.
    accuracy : float
        Accuracy of final probability as share of target - min probability. 
        Not needed if the MinHelp search algorithm is used. Default is None.
    nec_help : float, optional
        Average necessary import/debt of the last penalty. Not needed if the
        Prob seacrh algorithm is used. Default is None.
    accuracy_help : float, optional
        The accuracy demanded from the final average necessary help (given as
        maximum absolute value of the difference between final necessary help
        and the minimum nevessary help). Not needed if the Prob search 
        algorithm is used. Default is None.
    

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
        # find next guess
        if prob - currentProb > (prob - min_prob) * accuracy:
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
        
    elif nec_help is not None:
        # find next guess
        diff = np.abs(currentNecHelp - nec_help)
        if diff > accuracy_help:
            rhoLastUp = rhoOld
            if rhoLastDown == np.inf:
                rhoNew = rhoOld * 4
            else:
                rhoNew = (rhoOld + rhoLastDown) / 2 
        elif diff <= accuracy_help:
            rhoLastDown = rhoOld
            if rhoLastUp == 0:
                rhoNew = rhoOld / 4
            else:
                rhoNew = (rhoOld + rhoLastUp) / 2
    
    return(rhoNew, rhoLastDown, rhoLastUp)

def _checkIniGuess(rhoIni, 
                  args,
                  checkedGuess,
                  fileProb,
                  prob = None,
                  min_prob = None,
                  nec_help = None,
                  objective = "F",
                  accuracy_rho = None,
                  accuracy = None,
                  accuracy_help = None,
                  console_output = None,
                  logs_on = None):
    """
    Checks whether an initial guess from a run with a lower sample size is
    still a valid solution for correct the penalty. If so, it also returns the
    corresponding meta_sol, else it returns None.

    Parameters
    ----------
    rhoIni : float
        Guess for the penalty or None if there is no initial guess.
    args : dict
        Dictionary of arguments needed as model input.  
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed
    fileProb : str
        String combining all settings affecting the resulting probability based
        on a given penalty.
    prob : float, optional
        The desired probability for food security/solvency. Not needed if the
        MinHelp search  algorithm is used. Default is None.
    min_prob : float, optional
        The probability that is reached without using any area. Not needed if
        the MinHelp search algorithm is used. Default is None.
    nec_help : float, optional
        Average necessary import/debt of the last penalty. Not needed if the
        Prob seacrh algorithm is used. Default is None.
    objective : string "F" or "S"
        Specifies whether the function is called to check a guess for rhoS
        or for rhoF.
    accuracy_rho : float
        The share of the final rho that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < accuracy_rho * rho, for rho
        the current best guess for the correct penalty, then we use rho).
    accuracy : float
        Accuracy of final probability as share of target - min probability. 
        Not needed if the MinHelp search algorithm is used. Default is None.
    accuracy_help : float, optional
        The accuracy demanded from the final average necessary help (given as
        maximum absolute value of the difference between final necessary help
        and the minimum nevessary help). Not needed if the Prob search 
        algorithm is used. Default is None.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.


    Returns
    -------
    rho : float
        The final penalty (if guess worked) or None.
    meta_sol : dict
        The corresponding meta_sol or None.
    crop_alloc : np.array
        The corresponding optimal crop allocation or None.

    """
    
    if rhoIni is not None:
        
        if objective  == "F":
            rhoFguess = rhoIni
            rhoFcheck =  rhoIni - rhoIni * accuracy_rho
            rhoSguess = 0
            rhoScheck = 0
            rhoCheck = rhoFcheck
        elif objective  == "S":
            rhoSguess = rhoIni
            rhoScheck =  rhoIni - rhoIni * accuracy_rho
            rhoFguess = 0
            rhoFcheck = 0
            rhoCheck = rhoScheck
            
        if prob is not None:
            method = "prob"
        elif nec_help is not None:
            method = "nec_help"
        
        def _test(crop_alloc, currentProb, currentNecHelp, method = method, nec_help = nec_help, prob = prob, min_prob = min_prob, objective = objective):
            if method == "prob":
                tol = (prob - min_prob) * accuracy
                testPassed = (abs(currentProb - prob) < tol)
            elif method == "nec_help":
                testPassed = (np.abs(currentNecHelp - nec_help) < accuracy_help)
            return(testPassed)
  
        # check if rhoF from run with smaller N works here as well:
        _printing("     Checking guess from run with other N", console_output = console_output, logs_on = logs_on)
        crop_alloc, currentProb, currentNecHelp, durations, meta_sol = \
            _getResultsForRho(args, rhoFguess, rhoSguess, objective, fileProb, console_output = False, logs_on = False)
        # status, crop_alloc, meta_sol, sto_prob, durations = \
        #         SolveReducedLinearProblemGurobiPy(args, rhoFguess, rhoSguess, console_output = False, logs_on = False) 
        testPassed = _test(crop_alloc, currentProb, currentNecHelp)
        _ReportProgressFindingRho(rhoIni, currentProb, currentNecHelp, durations, \
                                 objective, method, testPassed, prefix = "Guess: ", console_output = console_output, \
                                 logs_on = logs_on)
        if checkedGuess:
            _printing("     We have a rho from a different N that was already double-checked!", console_output = console_output, logs_on = logs_on)
            return(rhoIni, meta_sol, crop_alloc)
        elif testPassed:    
            crop_alloc_check, currentProb_check, currentNecHelp_check, durations, meta_sol_check = \
                _getResultsForRho(args, rhoFcheck, rhoScheck, objective, fileProb, console_output = False, logs_on = False)
            # status, crop_alloc_check, meta_sol_check, sto_prob, durations = \
            #         SolveReducedLinearProblemGurobiPy(args, rhoFcheck, rhoScheck,
            #                                            console_output = False, logs_on = False) 
            testPassed = _test(crop_alloc_check, currentProb_check, currentNecHelp_check)
            _ReportProgressFindingRho(rhoCheck, currentProb_check, currentNecHelp_check, durations, \
                                objective, method, testPassed, prefix = "Check: ", console_output = console_output, \
                                 logs_on = logs_on)
            if not testPassed:
                _printing("     Cool, that worked!", console_output = console_output, logs_on = logs_on)
                return(rhoIni, meta_sol, crop_alloc)
            
        _printing("     Oops, that guess didn't work - starting from scratch\n", \
                 console_output = console_output, logs_on = logs_on)
            
    return(None, None, None)

    
def _ReportProgressFindingRho(rho,
                             currentProb,
                             currentNecHelp,
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
    rho : float
        Last penalty that was tried.
    currentProb : float
        Probability resulting for current rho (either for food security or
        solvency, depending on objective).
    nec_help : float
        Average necessary import or debt (depending on objective).
    durations : list
        Time that was needed for setting up the model, for solving the model,
        and total time used (in sec.)
    objective : string "F" or "S"
        Specifies whether the function is calles while searching for rhoF or rhoS.
    method : str "prob" or "nec_help"
        Specifying which search method is used to find the correct penalty .
        Default id "prob".
    testPassed : boolean
        Whether the last penalty had a corresponding probability close 
        enough to the given probability if method == "prob", or whether the
        last penalty had corresponding necessary import/debt close enough
        to the minimal necessary import/debt if method == "nec_help".    
   accuracy_int : float or False, optional
        Size of the current interval for which we know that the correct 
        penalty has to be within it. If false, the interval size will not be
        reported to the console. The default is False.
    prefix : float or False
        Used for additional information before the rest of the text. Used e.g. 
        when there are two next guesses when searching for the correct rhoS 
        within MinimizeNecessaryDebt. The default is "".
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    None.

    """
    # get correct probability and unit
    if objective == "F":
        unit = " $/10^3kcal"
        help_type = "import"
        unit_help = "10^12kcal"
    elif objective == "S":
        unit = " $/$"
        help_type = "debt"
        unit_help = "10^9$"
        
    # if length of accuracy interval is given create corresponding text piece
    if accuracy_int:
        accuracy_text = " (current accuracy interval: " + str(np.round(accuracy_int, 3)) + ")"
    else:
        accuracy_text = ""
        
    # if we are trying to find the lowest penatly that gives the right crop areas
    if method == "nec_help":
        if testPassed:
            help_text = ", min. " + help_type
        else:
            help_text = f", {currentNecHelp:.2e} " + unit_help 
    else:
        help_text = ""
    
    # either duration of solving model for this rho, or statment that result was already available
    if durations != "pre-calculated":
        durations = str(np.round(durations[2], 2)) + "s"
    
    # print information (if console_output = True)
    _printing("     " + prefix + "rho" + objective + ": " + str(rho) + unit + \
          ", prob" + objective + ": " + str(np.round(currentProb * 100, 2)) + \
          "%" + help_text + ", time: " + durations + accuracy_text, \
              console_output = console_output, logs_on = logs_on)
        
    return(None)

def LoadPenaltyStuff(objective, **kwargs):
    """
    Loads the information that were collected during the search for the 
    correct penalty for a given model run.

    Parameters
    ----------
    objective : str "F" or "S"
        Specifies whether information on finding rhoS or rhoF should be returned.
    **kwargs : 
        Specification of all model settings that differ from their deaults.

    Returns
    -------
    final_rho  : float
        The final penalty.
    rhos_tried_order : list of floats
        List of all penalties that were tried in the order of them being tried 
        in the search.
    rhos_tried : list of floats
        List of all penalties that were tried, ordered from low to high.
    crop_allocs : list of np.arrays
        List of crop allocations corresponding to all penalties that were tried
        (ordered corresponding rhos_tried).
    probabilities : list of floats
        List of probabilities corresponding to all penalties that were tried
        (ordered corresponding rhos_tried).
    necessary_help : list of np.arrays
        List of necessary import/debt corresponding to all penalties that were 
        tried (ordered corresponding rhos_tried).
    file : str
        filename of results that were loaded
    objective : str "F" or "S"
        Specifies whether information on finding rhoS or rhoF are returned.

    """
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