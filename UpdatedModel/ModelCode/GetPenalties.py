#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:05:11 2021

@author: Debbora Leip
"""
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

from ModelCode.MetaInformation import GetMetaInformation
from ModelCode.Auxiliary import printing
from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy
from ModelCode.GeneralSettings import figsize
from ModelCode.GeneralSettings import accuracyF
from ModelCode.GeneralSettings import accuracyS
from ModelCode.GeneralSettings import shareDiffF
from ModelCode.GeneralSettings import shareDiffS


# %% ########################## WRAPPING FUNCTION #############################

def GetPenalties(settings, args, yield_information, probF, probS, \
                 rhoFini = None, rhoSini = None, prints = True):
    """
    Given the probabilities probF and probS this either loads or calculates
    the corresponding penalties. Penalties are calculated with the respective
    other penalty set to zero, such that the probabilities resulting in the
    run using both penalties will always be at least as high as demanded.

    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Information on theon the yield distributions.
    probF : float
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). 
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). 
    rhoFini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoFini is None, a initial guess for rhoF will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for reaching 
        food demand. The default is None.
    rhoSini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoSini is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
        The default is None.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    rhoF : float
        The correct penalty rhoF to reach the probability probF
    rhoS : float
        The correct penalty rhoF to reach the probability probS
    necessary_debt : float
        The debt that needs to be taken to provide payments in case of 
        caastrophe with probS (when only using rhoS and setting rhoF = 0)
    needed_import : float
        Amount of food that needs to imported to reach the probability for
        food seecurity probF (when using only rhoF and setting rhoS = 0)
    """

    if probF == 0:
        rhoF = 0
    else:   
        # all settings that affect the calculation of rhoF
        SettingsBasics = "k" + str(settings["k"]) + \
                "using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "num_crops" + str(settings["num_crops"]) + \
                "yield_projection" + str(settings["yield_projection"]) + \
                "sim_start" + str(settings["sim_start"]) + \
                "pop_scenario" + str(settings["pop_scenario"]) + \
                "T" + str(settings["T"])
        SettingsMaxProbS = SettingsBasics + \
                "risk" + str(settings["risk"]) + \
                "tax" + str(settings["tax"]) + \
                "perc_guaranteed" + str(settings["perc_guaranteed"]) + \
                "N" + str(settings["N"])
        SettingsMaxProbF = SettingsBasics + "N" + str(settings["N"])
        SettingsFirstGuess =  SettingsBasics + "probF" + str(probF)
        SettingsAffectingRhoF = SettingsFirstGuess + "N" + str(settings["N"])
                        
        # get dictionary of settings for which rhoF has been calculated already
        with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
            dict_rhoFs = pickle.load(fp)
        with open("PenaltiesAndIncome/Imports.txt", "rb") as fp:    
            dict_imports = pickle.load(fp)
        with open("PenaltiesAndIncome/MaxProbFforAreaF.txt", "rb") as fp:    
            dict_maxProbF = pickle.load(fp)
        with open("PenaltiesAndIncome/MaxProbSforAreaF.txt", "rb") as fp:    
            dict_maxProbS = pickle.load(fp)
            
        # if this setting was already calculated, fetch rhoF
        if SettingsAffectingRhoF in dict_rhoFs.keys():
            printing("Fetching rhoF", prints = prints)
            rhoF = dict_rhoFs[SettingsAffectingRhoF]
            needed_import = dict_imports[SettingsAffectingRhoF]
            printing("     rhoF: " + str(rhoF) + ", needed import: " + \
                     str(np.round(needed_import, 2)) + " 10^12 kcal", \
                     prints = prints)
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoF calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 1)
            if rhoFini is None:
                rhoFini = GetInitialGuess(dict_rhoFs, SettingsFirstGuess)
            # calculating rhoF
            printing("Calculating rhoF and import", prints = prints)
            
            rhoF, maxProbF, maxProbS, needed_import, crop_alloc, meta_sol = \
                    GetRhoF(args, yield_information, probF, rhoFini, prints = prints)
                    
            dict_rhoFs[SettingsAffectingRhoF] = rhoF
            dict_imports[SettingsAffectingRhoF] = needed_import
            dict_maxProbF[SettingsMaxProbF] = maxProbF
            dict_maxProbS[SettingsMaxProbS] = maxProbS
        
        # saving updated dicts
        with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:    
             pickle.dump(dict_rhoFs, fp)
        with open("PenaltiesAndIncome/Imports.txt", "wb") as fp:    
             pickle.dump(dict_imports, fp)
        with open("PenaltiesAndIncome/MaxProbFforAreaF.txt", "wb") as fp:    
             pickle.dump(dict_maxProbF, fp)
        with open("PenaltiesAndIncome/MaxProbSforAreaF.txt", "wb") as fp:    
             pickle.dump(dict_maxProbS, fp)
            
  
    if probS == 0:
        rhoS = 0
    else:         
        # all settings that affect the calculation of rhoS
        SettingsBasics = "k" + str(settings["k"]) + \
                "using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "num_crops" + str(settings["num_crops"]) + \
                "yield_projection" + str(settings["yield_projection"]) + \
                "sim_start" + str(settings["sim_start"]) + \
                "pop_scenario" + str(settings["pop_scenario"]) +  \
                "T" + str(settings["T"])
        SettingsMaxProbF = SettingsBasics + "N" + str(settings["N"])
        SettingsBasics = SettingsBasics + \
                "risk" + str(settings["risk"]) + \
                "tax" + str(settings["tax"]) + \
                "perc_guaranteed" + str(settings["perc_guaranteed"])
        SettingsMaxProbS = SettingsBasics + "N" + str(settings["N"])
        SettingsFirstGuess = SettingsBasics + "probS" + str(probS)
        SettingsAffectingRhoS = SettingsFirstGuess + \
                "N" + str(settings["N"])
                     
        # get dictionary of settings for which rhoS has been calculated already
        with open("PenaltiesAndIncome/RhoSs.txt", "rb") as fp:    
            dict_rhoSs = pickle.load(fp)
        with open("PenaltiesAndIncome/MinimizedNecessaryDebt.txt", "rb") as fp:    
            dict_necDebt = pickle.load(fp)
        with open("PenaltiesAndIncome/MaxProbSforAreaS.txt", "rb") as fp:    
            dict_maxProbS = pickle.load(fp)
        with open("PenaltiesAndIncome/MaxProbFforAreaS.txt", "rb") as fp:    
            dict_maxProbF = pickle.load(fp)
           
        # if this setting was already calculated, fetch rhoS
        if SettingsAffectingRhoS in dict_rhoSs.keys():
            printing("\nFetching rhoS", prints = prints)
            rhoS = dict_rhoSs[SettingsAffectingRhoS]
            necessary_debt = dict_necDebt[SettingsAffectingRhoS]
            printing("     rhoS: " + str(rhoS) + ", necessary debt: " + \
                     str(np.round(necessary_debt, 3)) + " 10^9$", \
                     prints = prints)
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoS calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 100)
            if rhoSini is None:
                rhoSini = GetInitialGuess(dict_rhoSs, SettingsFirstGuess)
            # calculating rhoS
            printing("\nCalculating rhoS", prints = prints)
            rhoS, necessary_debt, maxProbS, maxProbF = GetRhoS_Wrapper(args, yield_information, probS, rhoSini, SettingsAffectingRhoS, prints = prints)
            dict_rhoSs[SettingsAffectingRhoS] = rhoS
            dict_necDebt[SettingsAffectingRhoS] = necessary_debt
            dict_maxProbS[SettingsMaxProbS] = maxProbS
            dict_maxProbF[SettingsMaxProbF] = maxProbF
        
        # saving updated dict
        with open("PenaltiesAndIncome/RhoSs.txt", "wb") as fp:    
             pickle.dump(dict_rhoSs, fp)
        with open("PenaltiesAndIncome/MinimizedNecessaryDebt.txt", "wb") as fp:    
             pickle.dump(dict_necDebt, fp)
        with open("PenaltiesAndIncome/MaxProbSforAreaS.txt", "wb") as fp:    
             pickle.dump(dict_maxProbS, fp)
        with open("PenaltiesAndIncome/MaxProbFforAreaS.txt", "wb") as fp:    
             pickle.dump(dict_maxProbF, fp)
             
    return(rhoF, rhoS, necessary_debt, needed_import)

# %% #################### FUNCTIONS TO CHECK POTENTIAL ########################

def CheckPotential(args, yield_information, probF = None, probS = None, \
                   accuracyF = accuracyF, accuracyS = accuracyS, prints = True):
    """
    Wrapper function for finding potential of area either for food security 
    or for solvency. Only one probF and probS should differ from None, thus
    deciding which potential we want to analyze.
    
    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Other information on the model setup (on the yield distributions).
    probF : float or None, optional
        The desired probability for food security. The default is None.
    probS : float or None, optional
        The desired probability for solvency. The default is None.
    accuracyF : int, optional
        Desired decimal places of accuracy of the obtained probF. 
        The default is defined in ModelCode/GeneralSettings.
    accuracyS : int, optional
        Desired decimal places of accuracy of the obtained probS. 
        The default is defined in ModelCode/GeneralSettings.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    Depending on settings, either results from CheckForFullAreaProbF() or 
    from CheckForFullAreaProbS() are returned.

    """
    if probS is not None and probF is not None:
        sys.exit("You need to choose between probF and probS to see potential of full area.")
    elif probF is None and probS is None:
        sys.exit("Either the desired probF or the desired probS needs to be given.")
    elif probF is not None and probS is None:
        return(CheckOptimalProbF(args, yield_information, probF, accuracyF, prints))
    elif probS is not None and probF is None:
        return(CheckOptimalProbS(args, yield_information, probS, accuracyS, prints))
    
def CheckOptimalProbF(args, other, probF, probS, accuracy, prints = True):
    """
    Function to find the highest probF possible under the given settings, and
    calculating the amount of import needed to increase this probabtility to 
    the probF desired.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    other : dict
        Other information on the model setup (on the yield distributions).
    probF : float
        The desired probability for food security.
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probF. 
        The default is defined in ModelCode/GeneralSettings.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

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
    
    # checking for import
    if args["import"] != 0:
        sys.exit("There is already a positive import value!")
    
    # find best crop per cluster (based on average yields)
    yld_means = other["yld_means"]  # t/ha
    yld_means = np.swapaxes(np.swapaxes(yld_means, 1, 2) * \
                            args["crop_cal"], 1, 2) # 10^6 kcal/ha
    which_crop = np.argmax(yld_means, axis = 1)

    # set area in the cluster to full area for the rigth crop
    x = np.zeros((args["T"], args["num_crops"], len(args["k_using"])))
    for t in range(0, args["T"]):
        for k in range(0, len(args["k_using"])):
            x[t, which_crop[t, k], k] = args["max_areas"][k]
    
    # run obective function for this area and the given settings
    meta_sol = GetMetaInformation(x, args, rhoF = 0, rhoS = 0) 
    
    # get resulting probabilities
    max_probF = meta_sol["prob_food_security"]
    max_probS = meta_sol["prob_staying_solvent"]
    printing("     maxProbF: " + str(np.round(max_probF * 100, accuracy - 1)) + "%" + \
          ", maxProbS: " + str(np.round(max_probS * 100, accuracy - 1)) + "%", prints)
    
    # check if it is high enough (shortcomings given as demand - production (- import))
    needed_import = np.quantile(meta_sol["shortcomings"]\
                 [~np.isnan(meta_sol["shortcomings"])].flatten(), probF)
    if max_probF >= probF:
        printing("     Desired probF (" + str(np.round(probF * 100, accuracy - 1)) \
                             + "%) can be reached\n", prints)
    else:
        printing("     Import of " + str(np.round(needed_import, 2)) + \
                 " 10^12 kcal is needed to reach probF = " + \
                 str(np.round(probF * 100, accuracy - 1)) + "%\n", prints = prints)
            
    return(max_probF, max_probS, needed_import)

def CheckOptimalProbS(args, other, probS, accuracy, prints = True):
    """
    Function to find the highest probS that is possible under given settings.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    other : dict
        Other information on the model setup (on the yield distributions).
    probS : float
        The desired probability for solvency.
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probS. 
        The default is defined in ModelCode/GeneralSettings.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    maxProbS : float
        Maximum probability for solvency that can be reached under these 
        settings.
    maxProbF : float
        Probability for food security for the settings that give the maxProbS.
    probSnew : float
        The new probability for solvency (different from the input probS in 
        case that probability can't be reached).

    """
    # try for rhoS = 1e9 (as a proxy for rhoS = inf)
    status, crop_alloc, meta_sol, prob, durations = \
         SolveReducedcLinearProblemGurobiPy(args, 0, 1e9, probS, prints = False)   
    
    # get resulting probabilities
    max_probS = meta_sol["prob_staying_solvent"]
    max_probF = meta_sol["prob_food_security"]
    printing("     maxProbS: " + str(np.round(max_probS * 100, accuracy - 1)) + "%" + \
          ", maxProbF: " + str(np.round(max_probF * 100, accuracy - 1)) + "%", prints)
        
    # check if it is high enough
    necessary_debt = meta_sol["necessary_debt"]
    if max_probS >= probS:
        printing("     Desired probS (" + str(np.round(probS * 100, accuracy - 1)) \
                             + "%) can be reached", prints)
    else:
        printing("     Desired probS (" + str(np.round(probS * 100, accuracy - 1)) \
                  + "%) cannot be reached (neccessary debt " + \
                  str(np.round(necessary_debt, 3)) + " 10^9$)", prints)
        
    return(max_probS, max_probF, necessary_debt)

# %% ############################## GET RHOF ##################################

def GetRhoF(args, yield_information, probF, rhoFini, shareDiff = shareDiffF, \
            accuracy = accuracyF, prints = True):
    """
    Finding the correct rhoF given the probability probF, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Other information on the model setup (on the yield distributions).
    probF : float
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob").
    rhoFini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoFini is None, a initial guess for rhoF will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for reaching 
        food demand.
    shareDiff : int
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoF, for rhoF
        the current best guess for the correct penalty, then we use rhoF).
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probF. 
        The default is defined in ModelCode/GeneralSettings.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    rhoF : float
        The correct penalty rhoF to reach the probability probF
    maxProbF : float
        Maximum probability for food security that can be reached under these 
        settings.
    maxProbS : float
        Probability for solvency for the settings that give the maxProbF.
    needed_import : float
        Amount of food that needs to imported to reach the probability for
        food seecurity probF.    
    crop_alloc : np.array
        The optimal crop allocations using the penalty rhoF and setting rhoS 
        to zero in the given settings.
    meta_sol : dict
        Dictionary of meta information to the optimal crop allocations
    """
    
    # needed import
    args_tmp = args.copy()
    maxProbF, maxProbS, needed_import = CheckPotential(args_tmp, yield_information, \
                                                       probF = probF, prints = prints)
    
    if needed_import > 0:
        args_tmp["import"] = needed_import
        
    # accuracy information
    printing("     accuracy we demand for probF: " + str(accuracy - 2) + " decimal places", prints = prints)
    printing("     accuracy we demand for rhoF: 1/" + str(shareDiff) + " of final rhoF\n", prints = prints)
    
    # check if rhoF from run with smaller N works here as well:
    # if we get the right probF for our guess, and a lower probF for rhoFcheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rhoF is in that interval and can return our guess
    # TODO export this in extra function
    if rhoFini is not None:
        printing("     Checking guess from run with lower N", prints = prints)
        status, crop_alloc, meta_sol, prob, durations = \
                        SolveReducedcLinearProblemGurobiPy(args_tmp, rhoFini, 0, prints = False) 
        ReportProgressFindingRho(rhoFini, meta_sol, accuracy, durations, \
                                 "F", prefix = "Guess: ", prints = prints) 
        if np.round(meta_sol["prob_food_security"], accuracy) == probF:
            rhoFcheck = rhoFini - rhoFini/shareDiff
            status, crop_alloc_check, meta_sol_check, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args_tmp, rhoFcheck, 0, prints = False)  
            ReportProgressFindingRho(rhoFcheck, meta_sol_check, accuracy, durations, \
                                     "F", prefix = "Check: ", prints = prints) 
            if np.round(meta_sol_check["prob_food_security"], accuracy) < probF:
                printing("     Cool, that worked!", prints = prints)
                printing("\n     Final rhoF: " + str(rhoFini), prints = prints)
                return(rhoFini, maxProbF, maxProbS, needed_import, crop_alloc, meta_sol)    
        printing("     Oops, that guess didn't work - starting from scratch\n", prints = prints)
    
    # else we start from scratch
    rhoFini = 1
    
    # initialize values for search algorithm
    rhoFLastDown = np.inf
    rhoFLastUp = 0
    lowestCorrect = np.inf
    
    # calculate initial guess
    status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args_tmp, rhoFini, 0, prints = False)
    
    # update information
    if np.round(meta_sol["prob_food_security"], accuracy) == probF:
        lowestCorrect = rhoFini
                
    # remember guess
    rhoFold = rhoFini
    
    # report
    accuracy_int = lowestCorrect - rhoFLastUp
    ReportProgressFindingRho(rhoFold, meta_sol, accuracy, durations, \
                             "F", accuracy_int, prints = prints)
        
    while True:   
        # find next guess
        rhoFnew, rhoFLastDown, rhoFLastUp = \
                    UpdatedRhoGuess(meta_sol, rhoFLastUp, rhoFLastDown, \
                                    rhoFold, probF, accuracy, probType = "F")
       
        # solve model for guess
        status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args_tmp, rhoFnew, 0, prints = False)
        
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if np.round(meta_sol["prob_food_security"], accuracy) == probF:
            accuracy_int = rhoFnew - rhoFLastUp
            if accuracy_int < rhoFnew/shareDiff:
                rhoF = rhoFnew
                break
        elif np.round(meta_sol["prob_food_security"], accuracy) < probF:
            accuracy_int = lowestCorrect - rhoFnew
            if accuracy_int < lowestCorrect/shareDiff:
                rhoF = lowestCorrect
                break
        else:
            accuracy_int = lowestCorrect - rhoFLastUp
            
        # report
        ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                                 "F", accuracy_int, prints = prints)
            
        # remember guess
        rhoFold = rhoFnew
        if np.round(meta_sol["prob_food_security"], accuracy) == probF \
            and lowestCorrect > rhoFnew:
            lowestCorrect = rhoFnew
    
    # last report
    ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                             "F", accuracy_int, prints = prints)    
        
    printing("\n     Final rhoF: " + str(rhoF), prints = prints)
    
    return(rhoF, maxProbF, maxProbS, needed_import, crop_alloc, meta_sol)

# %% ############################## GET RHOS ##################################

def GetRhoS_Wrapper(args, other, probS, rhoSini, file, shareDiff = shareDiffS, \
                    accuracy = accuracyS, prints = True):
    """
    Finding the correct rhoS given the probability probS, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    other : dict
        Other information on the model setup (on the yield distributions).
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob").
    rhoSini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoSini is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probS. 
        The default is defined in ModelCode/GeneralSettings.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    rhoS : float
        The correct penalty rhoF to reach the probability probS
    probSnew : float
        The new probability for solvency (different from the input probS in 
        case that probability can't be reached).
    maxProbS : float
        Maximum probability for solvency that can be reached under these 
        settings.
    maxProbF : float
        Probability for food security for the settings that give the maxProbS.
    """
    
    # find the highest possible probS (and probF when using area to get the max
    # probS), and choose probSnew to be either the wanted probS or probSmax if
    # the wanted one is not possible
    maxProbS, maxProbF, necessary_debt = CheckPotential(args, other, probS = probS, prints = prints)   
    
    # if probS can be reached find lowest rhoS that gives probS
    if maxProbS >= probS:
        printing("     Finding corresponding penalty\n", prints)
        rhoS = GetRhoS(args, probS, rhoSini, shareDiff, accuracy, prints)
        necessary_debt = 0
    # if probS cannot be reached find rhoS that minimizes the debt that is
    # necessary for the government to provide payouts in probS of the samples
    else:
        printing("     Finding lowest penalty minimizing necessary debt\n", prints)
        rhoS, necessary_debt = MinimizeNecessaryDebt(args, probS, rhoSini, \
                            necessary_debt,  shareDiff, accuracy, file, prints)
    
    printing("\n     Final rhoS: " + str(rhoS))
    
    return(rhoS, necessary_debt, maxProbS, maxProbF)

def GetRhoS(args, probS, rhoSini, shareDiff, accuracy, prints = True):
    """
    Finding the correct rhoS given the probability probS, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    other : dict
        Other information on the model setup (on the yield distributions).
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob").
    rhoSini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoSini is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
    accuracy : int
        Desired decimal places of accuracy of the obtained probS. 
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    rhoS : float
        The correct penalty rhoF to reach the probability probS
    probSnew : float
        The new probability for solvency (different from the input probS in 
        case that probability can't be reached).
    maxProbS : float
        Maximum probability for solvency that can be reached under these 
        settings.
    maxProbF : float
        Probability for food security for the settings that give the maxProbS.
    """
        
    # accuracy information
    printing("     accuracy we demand for probS: " + str(accuracy - 2) + " decimal places", prints = prints)
    printing("     accuracy we demand for rhoS: 1/" + str(shareDiff) + " of final rhoS\n", prints = prints)
    
    # check if rhoS from run with smaller N works here as well
    # if we get the right probS for our guess, and a lower probS for rhoScheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rhoS is in that interval and can return our guess
    # TODO export that to separate function
    if rhoSini is not None:
        printing("     Checking guess from run with lower N", prints = prints)
        status, crop_alloc, meta_sol, prob, durations = \
                        SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, probS, prints = False)  
        ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                 "S", prefix = "Guess: ", prints = prints)
        if np.round(meta_sol["prob_staying_solvent"], accuracy) == probS:
            rhoScheck = rhoSini - rhoSini/shareDiff
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck, probS, prints = False)  
            ReportProgressFindingRho(rhoScheck, meta_sol, accuracy, durations, \
                                     "S", prefix = "Check: ", prints = prints)
            if np.round(meta_sol["prob_staying_solvent"], accuracy) < probS:
                printing("     Cool, that worked!", prints = prints)
                return(rhoSini)    
        printing("     Oops, that guess didn't work - starting from scratch\n", prints = prints)
    
    # else we start from scratch
    rhoSini = 100

    # initialize values for search algorithm
    rhoSLastDown = np.inf
    rhoSLastUp = 0
    lowestCorrect = np.inf
    
    # calculate results for initial guess
    status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, probS, prints = False)    
                  
    # remember guess
    rhoSold = rhoSini
    if np.round(meta_sol["prob_staying_solvent"], accuracy) == probS:
        lowestCorrect = rhoSini

    # report
    accuracy_int = lowestCorrect - rhoSLastUp
    ReportProgressFindingRho(rhoSold, meta_sol, accuracy, durations, \
                              "S", accuracy_int, prints = prints)

    while True:   
        
        # find next guess
        rhoSnew, rhoSLastDown, rhoSLastUp = \
                    UpdatedRhoGuess(meta_sol, rhoSLastUp, rhoSLastDown, \
                                    rhoSold, probS, accuracy, probType = "S")    
        
        # solve model for guess
        status, crop_alloc, meta_sol, prob, durations = \
           SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, probS, prints = False)
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if np.round(meta_sol["prob_staying_solvent"], accuracy) == probS:
            accuracy_int = rhoSnew - rhoSLastUp
            if accuracy_int < rhoSnew/shareDiff:
                rhoS = rhoSnew
                break
        elif np.round(meta_sol["prob_staying_solvent"], accuracy) < probS:
            accuracy_int = lowestCorrect - rhoSnew
            if accuracy_int < lowestCorrect/shareDiff:
                rhoS = lowestCorrect
                break
        else:
            accuracy_int = lowestCorrect - rhoSLastUp
            
        # report
        ReportProgressFindingRho(rhoSold, meta_sol, accuracy, durations, \
                                 "S", accuracy_int, prints = prints)
            
        # remember guess
        rhoSold = rhoSnew
        if np.round(meta_sol["prob_staying_solvent"], accuracy) == probS \
            and lowestCorrect > rhoSnew:
            lowestCorrect = rhoSnew

    # last report
    ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, "S", \
                             accuracy_int, prints = prints)    
    
    return(rhoS)
 
def MinimizeNecessaryDebt(args, probS, rhoSini, debt_top, shareDiff, accuracy, file, prints):
    """
    If the demanded probS can't be reached, we instead find rhoS such that 
    the debt that would be necessary to provide payments in probS of the 
    samples is minimized.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob").
    rhoSini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoSini is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).
    shareDiff : int
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoS, for rhoS
        the current best guess for the correct penalty, then we use rhoS).
    accuracy : int
        Desired decimal places of accuracy of the obtained probS. (Here this
        is only used for rounding of output for the console as the correct
        probS can't be reached anyway.)
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.
    file : str
        String combining all settings affecting rhoS, used to save a plot 
        of rhoS vs. necessary debt. 
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    FinalRhoS : flost
        The rhoS for which the necessary debt such that the government is able to 
        provide payments in probS of the samples is minimized.
    FinalNecessaryDebt : float
        The (minimized) necessary debt such that the government is able to 
        provide payments in probS of the samples.

    """
    
    # accuracy information
    printing("     accuracy we demand for rhoS: 1/" + str(shareDiff) + " of final rhoS\n", prints = prints)
    
    # checking for rhoS = 0
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, 0, 0, probS, prints = False) 
    debt_bottom = meta_sol["necessary_debt"]
    
    # check if rhoS from run with smaller N works here as well
    if rhoSini is not None:
        rhoS, necessary_debt = CheckRhoSiniDebt(args, probS, rhoSini, \
                    debt_top, debt_bottom, shareDiff, accuracy, file, prints)
        if rhoS is not None:
            printing("     Cool, that worked!", prints = prints)
            return(rhoS, necessary_debt)
    
    # initializing values for search algorithm
    LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
        FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(0, \
                    debt_bottom, debt_top, debt_bottom, shareDiff)
    
    # initialize figure showing rhoS vs. necessary debt to reach probS
    fig = plt.figure(figsize = figsize)  
    
    # plot and report
    plt.scatter(0, debt_bottom, s = 14, color = "blue")
    ReportProgressFindingRho(0, meta_sol, accuracy, durations, \
                            "S", interval, debt = debt_bottom, prints = prints)
    
    # checking for high rhoS
    rhoSnew = 100
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, probS, prints = False) 
    necessary_debt = meta_sol["necessary_debt"]
 
    # update information
    LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
        FinalrhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew, \
                    necessary_debt, debt_top, debt_bottom, shareDiff, \
                    UpperBorder, LowerBorder, rhoSvalley, debtsValley)
 
    # plot and report
    plt.scatter(rhoSnew, necessary_debt, s = 10)
    ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                             "S", interval, debt = necessary_debt, prints = prints)
        
    while True:
        # get next guess
        rhoSnew = UpdateRhoDebtOutside(debt_top, debt_bottom, UpperBorder, \
                                    LowerBorder, rhoSvalley, debtsValley)
        
        # if we are in the "valley" we need to change searching technique
        if rhoSnew == "Found valley!":
            break
        
        # calculate for new guess
        status, crop_alloc, meta_sol, prob, durations = \
            SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, probS, prints = False) 
        necessary_debt = meta_sol["necessary_debt"]
        
        # update information
        LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
            FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew, \
                        necessary_debt, debt_top, debt_bottom, shareDiff, \
                        UpperBorder, LowerBorder, rhoSvalley, debtsValley)
        
        # report
        ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                                 "S", interval, debt = necessary_debt, prints = prints)
        
        # plot
        plt.scatter(rhoSnew, necessary_debt, s = 10, color = "blue")
    
        # are we accurate enough?
        if FinalRhoS is not None:
            break
        
    if rhoSnew == "Found valley!":        
        while True:
            # get new guesses
            rhoSnew1, rhoSnew2 = UpdateRhoDebtValley(rhoSvalley, debtsValley)
                    
            # calculating results for first point
            status, crop_alloc, meta_sol1, prob, durations1 = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew1, probS, prints = False) 
            necessary_debt1 = meta_sol1["necessary_debt"]
        
            # plot
            plt.scatter(rhoSnew1, necessary_debt1, s = 10, color = "blue")
            
            # update information
            LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
                FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew1, \
                            necessary_debt1, debt_top, debt_bottom, shareDiff, \
                            UpperBorder, LowerBorder, rhoSvalley, debtsValley)
             
            # report
            ReportProgressFindingRho(rhoSnew1, meta_sol1, accuracy, durations1, \
                            "S", interval, debt = necessary_debt1, \
                            prefix = "1. ", prints = prints)
                
            # are we accurate enough?    
            if FinalRhoS is not None:
                break
                
            # calculating results for second point
            status, crop_alloc, meta_sol2, prob, durations2 = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew2, probS, prints = False) 
            necessary_debt2 = meta_sol2["necessary_debt"] 
        
            # plot
            plt.scatter(rhoSnew2, necessary_debt2, s = 10, color = "blue")
            
            # update information
            LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
                FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew2, \
                            necessary_debt2, debt_top, debt_bottom, shareDiff, \
                            UpperBorder, LowerBorder, rhoSvalley, debtsValley)
                    
            # report
            ReportProgressFindingRho(rhoSnew2, meta_sol2, accuracy, durations2, \
                            "S", interval, debt = necessary_debt2, \
                            prefix = "2. ", prints = prints)
               
            # are we accurate enough?    
            if FinalRhoS is not None:
                break
            
    # finish and save plot
    plt.xlabel("rhoS [$/$]", fontsize = 24)
    plt.ylabel("Necessary debt to reach probS [10^9$]", fontsize = 24)
    plt.title("Necessary debt for different rhoS", fontsize = 30)
    fig.savefig("Figures/rhoSvsDebts/CropAlloc_" + file + ".jpg", \
                bbox_inches = "tight", pad_inches = 1)
            
    return(FinalRhoS, FinalNecessaryDebt)

    
# %% ####################### AUXILIARY FUNCTIONS RHOS #########################
    
def CheckRhoSiniDebt(args, probS, rhoSini, debt_top, debt_bottom, shareDiff, accuracy, prints):
    """
    For the case that probS cannot be reached and therefore the necessary debt 
    is minimized, this checks if the rhoS calculated for a lower sample size
    N or given manually as rhoSini also works for this sample size.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob").
    rhoSini : float or None 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoSini is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).
    debt_top : float
        The debt that would be necessary for rhoS = 0.
    shareDiff : int
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoS, for rhoS
        the current best guess for the correct penalty, then we use rhoS).
    accuracy : int
        Desired decimal places of accuracy of the obtained probS. (Here this
        is only used for rounding of output for the console as the correct
        probS can't be reached anyway.)
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    rhoS : float or None
        If the guess worked, this is the rhoS, else it is None.
    necessary_debt : float or None
        If the guess worked, this is the necessary debt, else it is None.

    """
    # TODO this needs to be updated
    
    
    # TODO I think this case doesn't happen - if it does I could still
    # optimize this to use the guess to improve computational time for 
    # rhoSini == 0
    if rhoSini != 0:
        printing("     Checking guess from run with lower N", prints = prints)
        status, crop_alloc, meta_sol, prob, durations = \
            SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, probS, prints = False) 
        necessary_debt = meta_sol["necessary_debt"]
        ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                 "S", debt = necessary_debt, prefix = "Guess: ", prints = prints)
            
        if necessary_debt == debt_top:
            rhoScheck = rhoSini - rhoSini/shareDiff
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck, probS, prints = False) 
            necessary_debt_check = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoScheck, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt_check, prefix = "Check: ", prints = prints)
            if necessary_debt_check > necessary_debt:
                return(rhoSini, necessary_debt)
        elif necessary_debt != debt_bottom:          
            rhoScheck1 = rhoSini - rhoSini/shareDiff
            rhoScheck2 = rhoSini + rhoSini/shareDiff
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck1, probS, prints = False) 
            necessary_debt_check1 = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoScheck1, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt_check1, prefix = "Check 1: ", prints = prints)
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck2, probS, prints = False) 
            necessary_debt_check2 = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoScheck2, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt_check2, prefix = "Check 2: ", prints = prints)
            if necessary_debt_check1 > necessary_debt and \
                necessary_debt_check2 > necessary_debt:
                return(rhoSini, necessary_debt)
    printing("     Oops, that guess didn't work - starting from scratch\n", prints = prints)
    return(None, None)
       

def UpdateDebtInformation(rhoSnew, necessary_debt, debt_top, debt_bottom, \
                          shareDiff, UpperBorder = np.inf, LowerBorder = 0, \
                          rhoSvalley =  [], debtsValley = []):
    """
    When searching for the rhoS by minimizing the necessary debt, this updates
    the information of all rhoS-guesses so far (called after each model run in 
    MinimizeNecessaryDebt). It then checks whether the current most accurate
    guess for rhoS is accurate enough.

    Parameters
    ----------
    rhoSnew : float
        The latest guess for rhoS.
    necessary_debt : TYPE
        The necessary debt when using rhoSnew.
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).
    debt_top : float
        The debt that would be necessary for rhoS = 0.
    shareDiff : int
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoS, for rhoS
        the current best guess for the correct penalty, then we use rhoS).
    UpperBorder : float, optional
        The lowest rhoS for which the necessary debt is equal to debt_top.
        The default is np.inf.
    LowerBorder : float, optional
        The highest rhoS for which the necessary debt is equal to debt_bottom.
        The default is 0.
    rhoSvalley : list, optional
        List (sorted) of all rhoS for which the necessary debt is neither debt_top nor
        debt_bottom. The default is [].
    debtsValley : list, optional
        List of necessary debts corresponding to the penalties in rhoSvalley.
        The default is [].

    Returns
    -------
    LowerBorder : float
        Updated value of LowerBorder
    UpperBorder : float
        Updated value of UpperBorder
    rhoSvalley : list
        Updated version of rhoSvalley
    debtsValley : list
        Updated version of debtsValley
    interval : float
        Size of the current interval for which we know that the correct rhoS
        has to be within.
    rhoS : float or None
        If the current best guess for the correct rhoS is accurate enough it
        is returned, else None.
    necessary_debt :
        If the current best guess for the correct rhoS is accurate enough the
        corresponding necessary debt is returned, else None.

    """

    # Update inforamtion on Borders 
    if necessary_debt == debt_bottom:
        LowerBorder = rhoSnew
        
    elif necessary_debt == debt_top:
        UpperBorder = rhoSnew
    
    # update information on "valley"
    else:
        rhoSvalley.append(rhoSnew)
        debtsValley.append(necessary_debt)
        s = sorted(zip(rhoSvalley,debtsValley))
        rhoSvalley = [x for x,_ in s]
        debtsValley = [x for _,x in s]
      
    # get current best guess for rhoS and the current accuracy interval
    if len(debtsValley) == 0:
        interval = UpperBorder - LowerBorder
        if debt_bottom < debt_top:
            rhoS = 0
            necessary_debt = debt_bottom
        elif debt_bottom > debt_top:
            rhoS = UpperBorder
            necessary_debt = debt_top
                 
    elif len(debtsValley) == 1:
        if debtsValley[0] < debt_bottom and debtsValley[0] < debt_top:
            rhoS = rhoSvalley[0]
            necessary_debt = debtsValley[0]
            interval = max(UpperBorder - rhoS, rhoS - LowerBorder)
        elif debt_bottom < debt_top:
            rhoS = 0
            necessary_debt = debt_bottom
            interval = rhoSvalley[0] - LowerBorder
        elif debt_bottom > debt_top:
            rhoS = UpperBorder
            necessary_debt = debt_top
            interval = UpperBorder - rhoSvalley[0]
            
    else:
        if debtsValley[0] == min(debtsValley):
            rhoS = 0
            necessary_debt = debt_bottom
            interval = rhoSvalley[0] - LowerBorder
        elif debtsValley[-1] == min(debtsValley):
            rhoS = UpperBorder
            necessary_debt = debt_top
            interval = UpperBorder - rhoSvalley[-1]
        else:
            i = debtsValley.index(min(debtsValley))
            rhoS = rhoSvalley[i]
            necessary_debt = min(debtsValley)  
            interval = rhoSvalley[i+1] - rhoSvalley[i-1]
    
    # check whether we are acurate enough
    if rhoS != 0:
        if interval < rhoS/shareDiff:
            return(LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
                   rhoS, necessary_debt)
    else:
        if interval < LowerBorder/shareDiff:
            return(LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
                   rhoS, necessary_debt)          
        
    return(LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, None, None)


def UpdateRhoDebtOutside(debt_top, debt_bottom, \
                  UpperBorder, LowerBorder, rhoSvalley, debtsValley):
    """
    While we have not found a valley in the function of necessary debt per rhoS,
    this gives the next guess for rhoS. (Not for all settings there is such 
    a valley).

    Parameters
    ----------
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).
    debt_top : float
        The debt that would be necessary for rhoS = 0.
    UpperBorder : float, optional
        The lowest rhoS for which the necessary debt is equal to debt_top.
        The default is np.inf.
    LowerBorder : float, optional
        The highest rhoS for which the necessary debt is equal to debt_bottom.
        The default is 0.
    rhoSvalley : list
        List (sorted) of all rhoS for which the necessary debt is neither 
        debt_top nor debt_bottom.
    debtsValley : list
        List of necessary debts corresponding to the penalties in rhoSvalley.
        

    Returns
    -------
    rhoSnew : float
        The next guess for rhoS

    """
    
    # as long as we don't have an UpperBorder we keep increasing
    if UpperBorder == np.inf:
        if len(rhoSvalley) == 0:
            rhoSnew = LowerBorder * 2
        else:
            rhoSnew = rhoSvalley[-1] * 2
    
    else:
        if len(rhoSvalley) == 0:
            rhoSnew = (LowerBorder + UpperBorder)/2
        elif debtsValley[-1] == min(debtsValley):
            rhoSnew = (rhoSvalley[-1] + UpperBorder)/2
        elif debtsValley[0] == min(debtsValley):
            rhoSnew = (rhoSvalley[0] + LowerBorder)/2
        else:
            rhoSnew = "Found valley!"
            
    return(rhoSnew)

def UpdateRhoDebtValley(rhoSvalley, debtsValley):
    """
    If we have found a valley in the function of necessary debt per rhoS,
    this gives the next two guess for rhoS. (In a valley, the correct rhoS
    could always be left or right of the current best guess for rhoS.)
    
    Parameters
    ----------
    rhoSvalley : list
        List (sorted) of all rhoS for which the necessary debt is neither 
        debt_top nor debt_bottom.
    debtsValley : list
        List of necessary debts corresponding to the penalties in rhoSvalley.

    Returns
    -------
    rhoSnew1 : float
        First next guess for rhoS (higher than the current best guess for rhoS).
    rhoSnew2 : float
        Second next guess for rhoS (lower than the current best guess for rhoS).
    

    """
    
    i = debtsValley.index(min(debtsValley))
    rhoSnew1 = (rhoSvalley[i] + rhoSvalley[i+1])/2
    rhoSnew2 = (rhoSvalley[i] + rhoSvalley[i-1])/2
    
    return(rhoSnew1, rhoSnew2)  

# %% ###################### JOINT AUXILIARY FUNCTIONS #########################

def GetInitialGuess(dictGuesses, name):
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

    Returns
    -------
    rho : float
        The value of the penalty for the same settings with a lower sample 
        size if existing, None else.

    """
    
    # initialize values
    bestN = 0
    bestFile = None
    rho = None
    
    # check for cases with same settings but different N
    for file in dictGuesses.keys():
        if file.startswith(name + "N"):
            N = int(file[len(name)+1:])
            if N > bestN:
                bestN = N
                bestFile = file
                
    # get rho from the case with the highest N
    if bestFile is not None: 
        rho = dictGuesses[bestFile]
        
    return(rho)

def UpdatedRhoGuess(meta_sol, rhoLastUp, rhoLastDown, rhoOld, prob, accuracy, probType):
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
    probType : string, "F" or "S"
        Specifies whether the function is called to find the next guess for 
        rhoS or for rhoF.

    Returns
    -------
    rhoNew : float
        The next penalty guess.
    rhoLastDown : float
        Updated version of rhoLastDown
    rhoLastUp : float
        Updated version of rhoFLastUp

    """
    # specifiy which probability to use
    if probType == "F":
        currentProb = meta_sol["prob_food_security"]
    elif probType == "S":
        currentProb = meta_sol["prob_staying_solvent"]
    
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
    
    return(rhoNew, rhoLastDown, rhoLastUp)

def ReportProgressFindingRho(rhoOld, meta_sol, accuracy, durations, \
                             ProbType, accuracy_int = False, debt = False, prefix = "", prints = True):
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
    accuracy_int : float or False, optional
        Size of the current interval for which we know that the correct 
        penalty has to be within it. If false, the interval size will not be
        reported to the console. The default is False.
    ProbType : string,F" or "S"
        Specifies whether the function while searching for rhoF or rhoS.
    debt : float or False
        Necessary debt for the government being able to provide payouts in 
        probS of the samples. Only relevant when called from 
        MinimizeNecessaryDebt. If False the debt is not reported. 
        The default is False.
    prefix : float or False
        Used for additional information before the rest of the text. Used e.g. 
        when there are two next guesses when searching for the correct rhoS 
        within MinimizeNecessaryDebt. The default is "".
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    None.

    """
    # get correct probability and unit
    if ProbType == "F":
        currentProb = meta_sol["prob_food_security"]
        unit = " $/10^3kcal"
    elif ProbType == "S":
        currentProb = meta_sol["prob_staying_solvent"]
        unit = " $/$"
    
    # if debt is given create corresponding text piece
    if debt:
        debt_text = ", nec. debt: " + str(np.round(debt, 3)) + " 10^9$"
    else:
        debt_text = ""
        
    # if length of accuracy interval is given create corresponding text piece
    if accuracy_int:
        accuracy_text = " (current accuracy interval: " + str(np.round(accuracy_int, 2)) + ")"
    else:
        accuracy_text = ""
        
    # print information (if prints = True)
    printing("     " + prefix + "rho" + ProbType + ": " + str(rhoOld) + unit + \
          ", prob" + ProbType + ": " + str(np.round(currentProb * 100, \
                                                    accuracy -1)) + \
          "%" + debt_text + ", time: " + str(np.round(durations[2], 2)) + "s" + accuracy_text, \
              prints = prints)
    
    return(None)

    