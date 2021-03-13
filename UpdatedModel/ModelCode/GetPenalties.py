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


# %% ########################## WRAPPING FUNCTION #############################

def GetPenalties(settings, args, yield_information, \
                 console_output = None,  logs_on = None):
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
    yield_information : dict
        Information on theon the yield distributions.
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
            printing("Fetching rhoF", console_output = console_output, logs_on = logs_on)
            rhoF = dict_rhoFs[SettingsAffectingRhoF]
            needed_import = dict_imports[SettingsAffectingRhoF]
            maxProbFareaF = dict_maxProbF[SettingsMaxProbF]
            maxProbSareaF = dict_maxProbS[SettingsMaxProbS]
            if needed_import <= 0:
                printing("     rhoF: " + str(rhoF) + ", no import needed", \
                         console_output = console_output, logs_on = logs_on)    
            else:
                printing("     rhoF: " + str(rhoF) + ", needed import: " + \
                         str(np.round(needed_import, 2)) + " 10^12 kcal", \
                         console_output = console_output, logs_on = logs_on)
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoF calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 1)
            if rhoFini is None:
                rhoFini, checkedGuess = GetInitialGuess(dict_rhoFs, SettingsFirstGuess, settings["N"])
            # calculating rhoF
            printing("Calculating rhoF and import", console_output = console_output, logs_on = logs_on)
            
            rhoF, maxProbFareaF, maxProbSareaF, needed_import, crop_alloc, meta_sol = \
                GetRhoF_Wrapper(args, yield_information, probF, rhoFini, checkedGuess, \
                                SettingsAffectingRhoF, console_output = console_output, logs_on = logs_on)    
                  
            dict_rhoFs[SettingsAffectingRhoF] = rhoF
            dict_imports[SettingsAffectingRhoF] = needed_import
            dict_maxProbF[SettingsMaxProbF] = maxProbFareaF
            dict_maxProbS[SettingsMaxProbS] = maxProbSareaF
        
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
            printing("\nFetching rhoS", console_output = console_output, logs_on = logs_on)
            rhoS = dict_rhoSs[SettingsAffectingRhoS]
            necessary_debt = dict_necDebt[SettingsAffectingRhoS]
            maxProbSareaS = dict_maxProbS[SettingsMaxProbS]
            maxProbFareaS = dict_maxProbF[SettingsMaxProbF]
            if necessary_debt <= 0:
                printing("     rhoS: " + str(rhoS) + ", no debt needed", \
                         console_output = console_output, logs_on = logs_on)  
            else:
                printing("     rhoS: " + str(rhoS) + ", necessary debt: " + \
                     str(np.round(necessary_debt, 4)) + " 10^9$", \
                     console_output = console_output, logs_on = logs_on)
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoS calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 100)
            if rhoSini is None:
                rhoSini, checkedGuess = GetInitialGuess(dict_rhoSs, SettingsFirstGuess, settings["N"])
            # calculating rhoS
            printing("\nCalculating rhoS", console_output = console_output, logs_on = logs_on)
            rhoS, necessary_debt, maxProbSareaS, maxProbFareaS = \
                GetRhoS_Wrapper(args, yield_information, probS, rhoSini, checkedGuess, \
                                SettingsAffectingRhoS, console_output = console_output, 
                                logs_on = logs_on)
            dict_rhoSs[SettingsAffectingRhoS] = rhoS
            dict_necDebt[SettingsAffectingRhoS] = necessary_debt
            dict_maxProbS[SettingsMaxProbS] = maxProbSareaS
            dict_maxProbF[SettingsMaxProbF] = maxProbFareaS
        
        # saving updated dict
        with open("PenaltiesAndIncome/RhoSs.txt", "wb") as fp:    
             pickle.dump(dict_rhoSs, fp)
        with open("PenaltiesAndIncome/MinimizedNecessaryDebt.txt", "wb") as fp:    
             pickle.dump(dict_necDebt, fp)
        with open("PenaltiesAndIncome/MaxProbSforAreaS.txt", "wb") as fp:    
             pickle.dump(dict_maxProbS, fp)
        with open("PenaltiesAndIncome/MaxProbFforAreaS.txt", "wb") as fp:    
             pickle.dump(dict_maxProbF, fp)
             
    return(rhoF, rhoS, necessary_debt, needed_import, \
           maxProbFareaF, maxProbSareaF, maxProbFareaS, maxProbSareaS)

# %% #################### FUNCTIONS TO CHECK POTENTIAL ########################

def CheckPotential(args, yield_information, probF = None, probS = None, \
                   console_output = None, logs_on = None):
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
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    Depending on settings, either results from CheckForFullAreaProbF() or 
    from CheckForFullAreaProbS() are returned.

    """
    
    from ModelCode.GeneralSettings import accuracyS
    from ModelCode.GeneralSettings import accuracyF
        
    if probS is not None and probF is not None:
        sys.exit("You need to choose between probF and probS to see potential of full area.")
    elif probF is None and probS is None:
        sys.exit("Either the desired probF or the desired probS needs to be given.")
    elif probF is not None and probS is None:
        return(CheckOptimalProbF(args, yield_information, probF, accuracyF, console_output, logs_on))
    elif probS is not None and probF is None:
        return(CheckOptimalProbS(args, yield_information, probS, accuracyS, console_output, logs_on))
    
def CheckOptimalProbF(args, yield_information, probF, accuracy,
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
              
    from ModelCode.GeneralSettings import accuracy_import as accuracy_import  
    
    # try for rhoF = 1e9 (as a proxy for rhoF = inf)
    status, crop_alloc, meta_sol, prob, durations = \
         SolveReducedcLinearProblemGurobiPy(args, 1e12, 0, console_output = False, logs_on = False)  
         
    plt.figure()
    plt.plot(range(0, args["T"]), crop_alloc[:,0,0])
    plt.plot(range(0, args["T"]), crop_alloc[:,1,0])
       
    
    # this should in thoery work (as yields cannot be negative) but if there
    # is not enough samples (especially with yield trends where the last years
    # will be the ones leading to the maximum import, where samples are the 
    # lowest) it could be off?
    
    # # find best crop per cluster (based on average yields)
    # yld_means = yield_information["yld_means"]  # t/ha
    # yld_means = np.swapaxes(np.swapaxes(yld_means, 1, 2) * \
    #                         args["crop_cal"], 1, 2) # 10^6 kcal/ha
    # which_crop = np.argmax(yld_means, axis = 1)

    # # set area in the cluster to full area for the rigth crop
    # x = np.zeros((args["T"], args["num_crops"], len(args["k_using"])))
    # for t in range(0, args["T"]):
    #     for k in range(0, len(args["k_using"])):
    #         x[t, which_crop[t, k], k] = args["max_areas"][k]
    
    # # run obective function for this area and the given settings
    # meta_sol = GetMetaInformation(x, args, rhoF = 0, rhoS = 0) 
    
    # get resulting probabilities
    max_probF = meta_sol["probF"]
    max_probS = meta_sol["probS"]
    printing("     maxProbF: " + str(np.round(max_probF * 100, accuracy - 1)) + "%" + \
          ", maxProbS: " + str(np.round(max_probS * 100, accuracy - 1)) + "%", \
              console_output = console_output, logs_on = logs_on)
    
    # check if it is high enough (shortcomings given as demand - production (- import))
    needed_import = meta_sol["necessary_import"]
    if max_probF >= probF:
        printing("     Desired probF (" + str(np.round(probF * 100, accuracy - 1)) \
                             + "%) can be reached\n", console_output = console_output, logs_on = logs_on)
    else:
        printing("     Import of " + str(np.round(needed_import, accuracy_import + 5)) + \
                 " 10^12 kcal is needed to reach probF = " + \
                 str(np.round(probF * 100, accuracy - 1)) + "%\n", \
                     console_output = console_output, logs_on = logs_on)
            
    return(max_probF, max_probS, needed_import)

def CheckOptimalProbS(args, yield_information, probS, accuracy,
                      console_output = None, logs_on = None):
    """
    Function to find the highest probS that is possible under given settings.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Information on the yield distributions.
    probS : float
        The desired probability for solvency.
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probS. 
        If None, the default as defined in ModelCode/GeneralSettings is used.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    maxProbS : float
        Maximum probability for solvency that can be reached under these 
        settings.
    maxProbF : float
        Probability for food security for the settings that give the maxProbS.
    necessary_debt : float
        The necessary debt to cover the payouts in probS of the cases (when 
        rhoF = 0).

    """
            
    # try for rhoS = 1e9 (as a proxy for rhoS = inf)
    status, crop_alloc, meta_sol, prob, durations = \
         SolveReducedcLinearProblemGurobiPy(args, 0, 1e12, console_output = False, logs_on = False)   
    
    # get resulting probabilities
    max_probS = meta_sol["probS"]
    max_probF = meta_sol["probF"]
    printing("     maxProbS: " + str(np.round(max_probS * 100, accuracy - 1)) + "%" + \
          ", maxProbF: " + str(np.round(max_probF * 100, accuracy - 1)) + "%", console_output, logs_on = logs_on)
        
    # check if it is high enough
    necessary_debt = meta_sol["necessary_debt"]
    if max_probS >= probS:
        printing("     Desired probS (" + str(np.round(probS * 100, accuracy - 1)) \
                             + "%) can be reached", console_output, logs_on = logs_on)
    else:
        printing("     Desired probS (" + str(np.round(probS * 100, accuracy - 1)) \
                  + "%) cannot be reached (neccessary debt " + \
                  str(np.round(necessary_debt, 4)) + " 10^9$)", console_output, logs_on = logs_on)
        
    return(max_probS, max_probF, necessary_debt)

# %% ############################## GET RHOF ##################################

def GetRhoF_Wrapper(args, yield_information, probF, rhoFini, checkedGuess, file, \
                    console_output = None, logs_on = None):
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
    
    from ModelCode.GeneralSettings import accuracyF as accuracy
    from ModelCode.GeneralSettings import shareDiffF as shareDiff
    
    # find the highest possible probF (and probS when using area to get the max
    # probF)
    maxProbF, maxProbS, necessary_import = CheckPotential(args, yield_information, \
                 probF = probF, console_output = console_output, logs_on = logs_on) 
    
        
    # if probF can be reached find lowest rhoF that gives probF
    if maxProbF >= probF:
        printing("     Finding corresponding penalty\n", console_output)
        rhoF, crop_alloc, meta_sol = GetRhoF(args, probF, rhoFini, checkedGuess, \
                                             shareDiff, accuracy, console_output, logs_on)
    # if probF cannot be reached find rhoF that minimizes the import that is
    # necessary for to provide the food demand in probF of the samples
    else:
        printing("     Finding lowest penalty minimizing necessary import\n", console_output)
        rhoF, crop_alloc, meta_sol = MinimizeNecessaryImport(args, probF, rhoFini, checkedGuess, \
                            necessary_import,  shareDiff, accuracy, file, console_output, logs_on)
        
    printing("\n     Final rhoF: " + str(rhoF), console_output = console_output, logs_on = logs_on)
    
    return(rhoF, maxProbF, maxProbS, necessary_import, crop_alloc, meta_sol)


def GetRhoF(args, probF, rhoFini, checkedGuess, shareDiff, accuracy, \
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
   
    # accuracy information
    printing("     accuracy that we demand for probF: " + str(accuracy - 2) + " decimal places",\
             console_output = console_output, logs_on = logs_on)
    printing("     accuracy that we demand for rhoF: 1/" + str(shareDiff) + " of final rhoF\n", \
             console_output = console_output, logs_on = logs_on)
    
    # check if rhoF from run with smaller N works here as well:
    # if we get the right probF for our guess, and a lower probF for rhoFcheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rhoF is in that interval and can return our guess
    # TODO export this in extra function
    if rhoFini is not None:
        if not checkedGuess:
            printing("     Checking guess from run with other N", console_output = console_output, logs_on = logs_on)
            status, crop_alloc, meta_sol, prob, durations = \
                            SolveReducedcLinearProblemGurobiPy(args, rhoFini, 0, console_output = False, logs_on = False) 
            ReportProgressFindingRho(rhoFini, meta_sol, accuracy, durations, \
                                     "F", prefix = "Guess - ", console_output = console_output, logs_on = logs_on) 
            if np.round(meta_sol["probF"], accuracy) == probF:
                rhoFcheck = rhoFini - rhoFini/shareDiff
                status, crop_alloc_check, meta_sol_check, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, rhoFcheck, 0, console_output = False, logs_on = False)  
                ReportProgressFindingRho(rhoFcheck, meta_sol_check, accuracy, durations, \
                                         "F", prefix = "Check - ", console_output = console_output, logs_on = logs_on) 
                if np.round(meta_sol_check["probF"], accuracy) < probF:
                    printing("     Cool, that worked!", console_output = console_output, logs_on = logs_on)
                    return(rhoFini, crop_alloc, meta_sol)    
            printing("     Oops, that guess didn't work - starting from scratch\n", console_output = console_output, logs_on = logs_on)
        else:
            printing("     We have a rhoF from a different N that was already double-checked!", console_output = console_output, logs_on = logs_on)
            status, crop_alloc, meta_sol, prob, durations = \
                            SolveReducedcLinearProblemGurobiPy(args, rhoFini, 0, console_output = False, logs_on = False) 
            ReportProgressFindingRho(rhoFini, meta_sol, accuracy, durations, \
                                     "F", prefix = "", console_output = console_output, logs_on = logs_on) 
            return(rhoFini, crop_alloc, meta_sol)    
            
    
    # else we start from scratch
    rhoFini = 1
    
    # initialize values for search algorithm
    rhoFLastDown = np.inf
    rhoFLastUp = 0
    lowestCorrect = np.inf
    meta_sol_lowestCorrect = []
    crop_alloc_lowestCorrect = []
    
    # calculate initial guess
    status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFini, 0, console_output = False, logs_on = False)
    
    # update information
    if np.round(meta_sol["probF"], accuracy) == probF:
        lowestCorrect = rhoFini
                
    # remember guess
    rhoFold = rhoFini
    
    # report
    accuracy_int = lowestCorrect - rhoFLastUp
    ReportProgressFindingRho(rhoFold, meta_sol, accuracy, durations, \
                             "F", accuracy_int, console_output = console_output, logs_on = logs_on)
        
    while True:   
        # find next guess
        rhoFnew, rhoFLastDown, rhoFLastUp = \
                    UpdatedRhoGuess(meta_sol, rhoFLastUp, rhoFLastDown, \
                                    rhoFold, probF, accuracy, probType = "F")
       
        # solve model for guess
        status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFnew, 0, console_output = False, logs_on = False)
        
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if np.round(meta_sol["probF"], accuracy) == probF:
            accuracy_int = rhoFnew - rhoFLastUp
            if accuracy_int < rhoFnew/shareDiff:
                rhoF = rhoFnew
                meta_sol_out = meta_sol
                crop_alloc_out = crop_alloc
                break
        elif np.round(meta_sol["probF"], accuracy) < probF:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoFnew
                if accuracy_int < lowestCorrect/shareDiff:
                    rhoF = lowestCorrect
                    meta_sol_out = meta_sol_lowestCorrect
                    crop_alloc_out = crop_alloc_lowestCorrect
                    break
            else:
                accuracy_int = rhoFLastDown - rhoFnew
        elif np.round(meta_sol["probF"], accuracy) > probF:
            accuracy_int = rhoFnew - rhoFLastUp
            
        # report
        ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                                 "F", accuracy_int, console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoFold = rhoFnew
        if np.round(meta_sol["probF"], accuracy) == probF \
            and lowestCorrect > rhoFnew:
            lowestCorrect = rhoFnew
            meta_sol_lowestCorrect = meta_sol
            crop_alloc_lowestCorrect = crop_alloc
    
    # last report
    ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
               "F", accuracy_int, console_output = console_output, logs_on = logs_on)    
            
    return(rhoF, crop_alloc_out, meta_sol_out)

def MinimizeNecessaryImport(args, probF, rhoFini, checkedGuess, \
                          necessary_import, shareDiff, accuracy, file, \
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
    
    # accuracy information
    printing("     accuracy that we demand for rhoF: 1/" + str(shareDiff) + " of final rhoF", console_output = console_output, logs_on = logs_on)
    
    # the demanded accuracy in the import is given as a share of the difference
    # between debt_top and debt_bottom
    from ModelCode.GeneralSettings import accuracy_import
    printing("     accuracy that we demand for the necessary import: " + str(accuracy_import) + " decimal places\n",\
             console_output = console_output, logs_on = logs_on)
        
    # minimized import
    min_import = np.round(necessary_import, accuracy_import)
    
    # check if rhoF from run with smaller N works here as well:
    if rhoFini is not None:
        printing("     Checking guess from run with other N", console_output = console_output, logs_on = logs_on)
        status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFini, 0, console_output = False, logs_on = False) 
        ReportProgressFindingRho(rhoFini, meta_sol, accuracy, durations, \
                                 "F", imports = min_import, 
                                 prefix = "Guess: ", console_output = console_output, \
                                 logs_on = logs_on)
        if checkedGuess:
            printing("     We have a rhoF from a different N that was already double-checked!", console_output = console_output, logs_on = logs_on)
            return(rhoFini, crop_alloc, meta_sol)
        elif np.round(meta_sol["necessary_import"], accuracy_import) == min_import:    
            rhoFcheck = rhoFini - rhoFini/shareDiff
            status, crop_alloc_check, meta_sol_check, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, rhoFcheck, 0, console_output = False, logs_on = False) 
            ReportProgressFindingRho(rhoFcheck, meta_sol_check, accuracy, durations, \
                                     "F", imports = min_import,
                                     prefix = "Check: ", console_output = console_output, \
                                     logs_on = logs_on)
            if np.round(meta_sol_check["necessary_import"], accuracy_import) > np.round(meta_sol["necessary_import"], accuracy_import):
                printing("     Cool, that worked!", console_output = console_output, logs_on = logs_on)
                return(rhoFini, crop_alloc, meta_sol)
        printing("     Oops, that guess didn't work - starting from scratch\n", \
                 console_output = console_output, logs_on = logs_on)
 
    
    # else we start from scratch
    rhoFini = 1

    # initialize figure showing rhoF vs. necessary import to reach probF
    from ModelCode.GeneralSettings import figsize
    fig = plt.figure(figsize = figsize) 
    
    # initialize values for search algorithm
    rhoFLastDown = np.inf
    rhoFLastUp = 0
    lowestCorrect = np.inf
    meta_sol_lowestCorrect = []
    crop_alloc_lowestCorrect = []
    
    # calculate initial guess
    status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFini, 0, console_output = False, logs_on = False)
    
    # update information
    if np.round(meta_sol["necessary_import"], accuracy_import) == min_import:
        lowestCorrect = rhoFini
                
    # remember guess
    rhoFold = rhoFini
    
    # plot and report
    plt.scatter(rhoFini, meta_sol["necessary_import"], s = 10, color = "blue")
    accuracy_int = lowestCorrect - rhoFLastUp
    ReportProgressFindingRho(rhoFold, meta_sol, accuracy, durations, \
                             "F", accuracy_int, imports = min_import, \
                             console_output = console_output, logs_on = logs_on)
        
    while True:   
        # find next guess
        rhoFnew, rhoFLastDown, rhoFLastUp = \
                    UpdatedRhoGuessImports(meta_sol, rhoFLastUp, rhoFLastDown, \
                                    rhoFold, min_import, accuracy_import)
       
        # solve model for guess
        status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, rhoFnew, 0, console_output = False, logs_on = False)
        
        plt.figure()
        plt.plot(range(0, args["T"]), crop_alloc[:,0,0])
        plt.plot(range(0, args["T"]), crop_alloc[:,1,0])
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if np.round(meta_sol["necessary_import"], accuracy_import) == min_import:
            accuracy_int = rhoFnew - rhoFLastUp
            if accuracy_int < rhoFnew/shareDiff:
                rhoF = rhoFnew
                meta_sol_out = meta_sol
                crop_alloc_out = crop_alloc
                break
        elif np.round(meta_sol["necessary_import"], accuracy_import) > min_import:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoFnew
                if accuracy_int < lowestCorrect/shareDiff:
                    rhoF = lowestCorrect
                    meta_sol_out = meta_sol_lowestCorrect
                    crop_alloc_out = crop_alloc_lowestCorrect
                    break
            else:
                accuracy_int = rhoFLastDown - rhoFnew
        elif np.round(meta_sol["necessary_import"], accuracy_import) < min_import:
            plt.scatter(rhoFnew, meta_sol["necessary_import"], s = 10, color = "blue")
            ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                                     "F", accuracy_int, imports = min_import, \
                                     console_output = console_output, logs_on = logs_on)
            
            sys.exit("Necessary import seems to be off, logic must be flawed.")
            
        # report
        plt.scatter(rhoFnew, meta_sol["necessary_import"], s = 10, color = "blue")
        ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                                 "F", accuracy_int, imports = min_import, \
                                 console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoFold = rhoFnew
        if np.round(meta_sol["necessary_import"], accuracy_import) == min_import \
            and lowestCorrect > rhoFnew:
            lowestCorrect = rhoFnew
            meta_sol_lowestCorrect = meta_sol
            crop_alloc_lowestCorrect = crop_alloc
    
    # last report
    plt.scatter(rhoFnew, meta_sol["necessary_import"], s = 10, color = "blue")
    ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                             "F", accuracy_int, imports = min_import, \
                             console_output = console_output, logs_on = logs_on)    

    # finish and save plot
    plt.xlabel(r"$\rho_\mathrm{F}$ [$\$/10^3\,kcal$]", fontsize = 24)
    plt.ylabel(r"Necessary import to reach $\alpha_\mathrm{F}$ [$10^{12}\,kcal$]", fontsize = 24)
    plt.title(r"Necessary import for different $\rho_\mathrm{F}$", fontsize = 30, pad = 20)
    fig.savefig("Figures/rhoFvsImports/" + file + ".jpg", \
                bbox_inches = "tight", pad_inches = 1)
            
    return(rhoF, crop_alloc_out, meta_sol_out)

# %% ############################## GET RHOS ##################################

def GetRhoS_Wrapper(args, yield_information, probS, rhoSini, checkedGuess, file, \
                    console_output = None, logs_on = None):
    """
    Finding the correct rhoS given the probability probS, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Information on the yield distributions.
    probS : float
        demanded probability of keeping the solvency constraint
    rhoSini : float or None 
        Initial guess for rhoS.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed.
    file : str
        String combining all settings affecting rhoS, used to save a plot 
        of rhoS vs. necessary debt in MinimizeNecessaryDebt. 
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
        The resulting penalty rhoS
    necessary_debt : float
        The necessary debt to cover the payouts in probS of the cases (when 
        rhoF = 0).
    maxProbS : float
        Maximum probability for solvency that can be reached under these 
        settings.
    maxProbF : float
        Probability for food security for the settings that give the maxProbS.
    """
    
    from ModelCode.GeneralSettings import accuracyS as accuracy
    from ModelCode.GeneralSettings import shareDiffS as shareDiff
    
    # find the highest possible probS (and probF when using area to get the max
    # probS), and choose probSnew to be either the wanted probS or probSmax if
    # the wanted one is not possible
    maxProbS, maxProbF, necessary_debt = CheckPotential(args, yield_information, probS = probS, console_output = console_output, logs_on = logs_on)   
    
    # if probS can be reached find lowest rhoS that gives probS
    if maxProbS >= probS:
        printing("     Finding corresponding penalty\n", console_output, logs_on = logs_on)
        rhoS = GetRhoS(args, probS, rhoSini, checkedGuess, shareDiff, accuracy, console_output, logs_on = logs_on)
    # if probS cannot be reached find rhoS that minimizes the debt that is
    # necessary for the government to provide payouts in probS of the samples
    else:
        printing("     Finding lowest penalty minimizing necessary debt\n", console_output, logs_on = logs_on)
        rhoS, necessary_debt = MinimizeNecessaryDebt(args, probS, rhoSini, checkedGuess, \
                            necessary_debt,  shareDiff, accuracy, file, console_output, logs_on = logs_on)
    
    printing("\n     Final rhoS: " + str(rhoS), console_output, logs_on = logs_on)
    
    return(rhoS, necessary_debt, maxProbS, maxProbF)

def GetRhoS(args, probS, rhoSini, checkedGuess, shareDiff, accuracy, 
             console_output = None, logs_on = None):
    """
    Finding the correct rhoS given the probability probS, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    yield_information : dict
        Information on the yield distributions.
    probS : float
        demanded probability of keeping the solvency constraint
    rhoSini : float or None 
        Initial guess for rhoS.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed.
    shareDiff : float
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoS, for rhoS
        the current best guess for the correct penalty, then we use rhoS).
    accuracy : int
        Desired decimal places of accuracy of the obtained probS. 
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
        The resulting penalty rhoS
    """
            
    # accuracy information
    printing("     accuracy that we demand for probS: " + str(accuracy - 2) +
             " decimal places", console_output = console_output, logs_on = logs_on)
    printing("     accuracy that we demand for rhoS: 1/" + str(shareDiff) + 
             " of final rhoS\n", console_output = console_output, logs_on = logs_on)
    
    # check if rhoS from run with smaller N works here as well
    # if we get the right probS for our guess, and a lower probS for rhoScheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rhoS is in that interval and can return our guess
    # TODO export that to separate function
    if rhoSini is not None:
        if not checkedGuess:
            printing("     Checking guess from run with other N", console_output = console_output, logs_on = logs_on)
            status, crop_alloc, meta_sol, prob, durations = \
                            SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, console_output = False, logs_on = False)  
            ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                     "S", prefix = "Guess - ", console_output = console_output, logs_on = logs_on)
            if np.round(meta_sol["probS"], accuracy) == probS:
                rhoScheck = rhoSini - rhoSini/shareDiff
                status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck, console_output = False, logs_on = False)  
                ReportProgressFindingRho(rhoScheck, meta_sol, accuracy, durations, \
                                         "S", prefix = "Check - ", console_output = console_output, logs_on = logs_on)
                if np.round(meta_sol["probS"], accuracy) < probS:
                    printing("     Cool, that worked!", console_output = console_output, logs_on = logs_on)
                    return(rhoSini)    
            printing("     Oops, that guess didn't work - starting from scratch\n", console_output = console_output, logs_on = logs_on)
        else:
            printing("     We have a rhoS from a different N that was already double-checked!", console_output = console_output, logs_on = logs_on)
            return(rhoSini)
        
    # else we start from scratch
    rhoSini = 100

    # initialize values for search algorithm
    rhoSLastDown = np.inf
    rhoSLastUp = 0
    lowestCorrect = np.inf
    
    # calculate results for initial guess
    status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, console_output = False, logs_on = False)    
                  
    # remember guess
    rhoSold = rhoSini
    if np.round(meta_sol["probS"], accuracy) == probS:
        lowestCorrect = rhoSini

    # report
    accuracy_int = lowestCorrect - rhoSLastUp
    ReportProgressFindingRho(rhoSold, meta_sol, accuracy, durations, \
                              "S", accuracy_int, console_output = console_output, logs_on = logs_on)

    while True:   
        
        # find next guess
        rhoSnew, rhoSLastDown, rhoSLastUp = \
                    UpdatedRhoGuess(meta_sol, rhoSLastUp, rhoSLastDown, \
                                    rhoSold, probS, accuracy, probType = "S")    
        
        # solve model for guess
        status, crop_alloc, meta_sol, prob, durations = \
           SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, console_output = False, logs_on = False)
        
        # We want to find the lowest penalty for which we get the right probability.
        # The accuracy interval is always the difference between the lowest 
        # penalty for which we get the right probability and the highest penalty
        # that gives a smaller probability (which is the rhoLastUp). If that is 
        # smaller than a certain share of the lowest correct penalte we have
        # reached the necessary accuracy.
        if np.round(meta_sol["probS"], accuracy) == probS:
            accuracy_int = rhoSnew - rhoSLastUp
            if accuracy_int < rhoSnew/shareDiff:
                rhoS = rhoSnew
                break
        elif np.round(meta_sol["probS"], accuracy) < probS:
            if lowestCorrect != np.inf:
                accuracy_int = lowestCorrect - rhoSnew
                if accuracy_int < lowestCorrect/shareDiff:
                    rhoS = lowestCorrect
                    break
            else:
                accuracy_int = rhoSLastDown - rhoSnew
        elif np.round(meta_sol["probS"], accuracy) > probS:
            accuracy_int = rhoSnew - rhoSLastUp
            
        # report
        ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                                 "S", accuracy_int, console_output = console_output, logs_on = logs_on)
            
        # remember guess
        rhoSold = rhoSnew
        if np.round(meta_sol["probS"], accuracy) == probS \
            and lowestCorrect > rhoSnew:
            lowestCorrect = rhoSnew

    # last report
    ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, "S", \
                             accuracy_int, console_output = console_output, logs_on = logs_on)    
    
    return(rhoS)
 
def MinimizeNecessaryDebt(args, probS, rhoSini, checkedGuess, \
                          debt_top, shareDiff, accuracy, file, \
                          console_output = None, logs_on = None):
    """
    If the demanded probS can't be reached, we instead find rhoS such that 
    the debt that would be necessary to provide payments in probS of the 
    samples is minimized.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    probS : float
        demanded probability of keeping the solvency constraint
    rhoSini : float or None 
        Initial guess for rhoS.
    checkedGuess : boolean
        True if there is an initial guess that we are already sure about, as 
        it was confirmed for two sample sizes N and N' with N >= 2N' (and the
        current N* > N'). False if there is no initial guess or the initial 
        guess was not yet confirmed.
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).
    shareDiff : float
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoS, for rhoS
        the current best guess for the correct penalty, then we use rhoS).
    accuracy : int
        Desired decimal places of accuracy of the obtained probS. (Here this
        is only used for rounding of output for the console as the correct
        probS can't be reached anyway.)
    file : str
        String combining all settings affecting rhoS, used to save a plot 
        of rhoS vs. necessary debt. 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

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
    printing("     accuracy that we demand for rhoS: 1/" + str(shareDiff) + " of final rhoS", console_output = console_output, logs_on = logs_on)
    
    # checking for rhoS = 0
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, 0, 0, console_output = False, logs_on = False) 
    debt_bottom = meta_sol["necessary_debt"]
    
    # the demanded accuracy in the debt is given as a share of the difference
    # between debt_top and debt_bottom
    from ModelCode.GeneralSettings import accuracy_debt
    accuracy_diff_debt = np.abs(debt_top - debt_bottom) * accuracy_debt
    printing("     accuracy that we demand for the necessary debt: " + \
             str(np.round(accuracy_diff_debt, 4)) + \
             " (depending on debt_top and debt_bottom)\n", 
             console_output = console_output, logs_on = logs_on)
    
    # check if rhoS from run with smaller N works here as well
    if rhoSini is not None:
        if checkedGuess:
            printing("     We have a rhoS from a different N that was already double-checked!", console_output = console_output, logs_on = logs_on)
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, console_output = False, logs_on = False) 
            necessary_debt = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt, prefix = "", 
                                     console_output = console_output, logs_on = logs_on)
            return(rhoSini, necessary_debt)
        else:    
            rhoS, necessary_debt = CheckRhoSiniDebt(args, probS, rhoSini, \
                        debt_top, debt_bottom, shareDiff, accuracy, console_output, logs_on)
            if rhoS is not None:
                printing("     Cool, that worked!", console_output = console_output, logs_on = logs_on)
                return(rhoS, necessary_debt)
    
    # initializing values for search algorithm and updating
    LowerBorder = 0
    UpperBorder = np.inf
    rhoSvalley = []
    debtsValley = []
    LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
        FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(0, \
                    debt_bottom, debt_top, debt_bottom, shareDiff, \
                    UpperBorder, LowerBorder, rhoSvalley, debtsValley)
    
    
    # initialize figure showing rhoS vs. necessary debt to reach probS
    from ModelCode.GeneralSettings import figsize
    fig = plt.figure(figsize = figsize)  
    
    # plot and report
    plt.scatter(0, debt_bottom, s = 14, color = "blue")
    ReportProgressFindingRho(0, meta_sol, accuracy, durations, \
                            "S", interval, debt = debt_bottom, 
                            console_output = console_output, logs_on = logs_on)
    
    # checking for high rhoS
    rhoSnew = 100
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, console_output = False, logs_on = False) 
    necessary_debt = meta_sol["necessary_debt"]
 
    # update information
    LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
        FinalrhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew, \
                    necessary_debt, debt_top, debt_bottom, shareDiff, \
                    UpperBorder, LowerBorder, rhoSvalley, debtsValley)
         
    # plot and report
    plt.scatter(rhoSnew, necessary_debt, s = 10)
    debt_report = DebtReport(necessary_debt, debt_bottom, debt_top)
    ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                             "S", interval, debt = debt_report, 
                             console_output = console_output, logs_on = logs_on)
        
    while True:
        # get next guess
        rhoSnew = UpdateRhoDebtOutside(debt_top, debt_bottom, UpperBorder, \
                                    LowerBorder, rhoSvalley, debtsValley)
        
        # if we are in the "valley" we need to change searching technique
        if rhoSnew == "Found valley!":
            break
        
        # calculate for new guess
        status, crop_alloc, meta_sol, prob, durations = \
            SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, console_output = False, logs_on = False) 
        necessary_debt = meta_sol["necessary_debt"]
        
        # update information
        LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
            FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew, \
                        necessary_debt, debt_top, debt_bottom, shareDiff, \
                        UpperBorder, LowerBorder, rhoSvalley, debtsValley)
                    
        # report
        debt_report = DebtReport(necessary_debt, debt_bottom, debt_top)
        ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                                 "S", interval, debt = debt_report, 
                                 console_output = console_output, logs_on = logs_on)
        
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
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew1, console_output = False, logs_on = False) 
            necessary_debt1 = meta_sol1["necessary_debt"]
        
            # plot
            plt.scatter(rhoSnew1, necessary_debt1, s = 10, color = "blue")
            
            # update information
            LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
                FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew1, \
                            necessary_debt1, debt_top, debt_bottom, shareDiff, \
                            UpperBorder, LowerBorder, rhoSvalley, debtsValley)
                    
            # report
            debt_report = DebtReport(necessary_debt, debt_bottom, debt_top)
            ReportProgressFindingRho(rhoSnew1, meta_sol1, accuracy, durations1, \
                            "S", interval, debt = debt_report, \
                            prefix = "1. ", console_output = console_output,
                            logs_on = logs_on)
                
            # are we accurate enough?    
            if FinalRhoS is not None:
                break
            
            # if rhoSnew1 is the new minimum rhoSnew2 cannot be better
            if rhoSvalley[debtsValley.index(min(debtsValley))] == rhoSnew1:
                continue
                
            # calculating results for second point
            status, crop_alloc, meta_sol2, prob, durations2 = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew2, console_output = False, logs_on = False) 
            necessary_debt2 = meta_sol2["necessary_debt"] 
        
            # plot
            plt.scatter(rhoSnew2, necessary_debt2, s = 10, color = "blue")
            
            # update information
            LowerBorder, UpperBorder, rhoSvalley, debtsValley, interval, \
                FinalRhoS, FinalNecessaryDebt = UpdateDebtInformation(rhoSnew2, \
                            necessary_debt2, debt_top, debt_bottom, shareDiff, \
                            UpperBorder, LowerBorder, rhoSvalley, debtsValley)
                    
            # report
            debt_report = DebtReport(necessary_debt2, debt_bottom, debt_top)
            ReportProgressFindingRho(rhoSnew2, meta_sol2, accuracy, durations2, \
                            "S", interval, debt = debt_report, \
                            prefix = "2. ", console_output = console_output, logs_on = logs_on)
               
            # are we accurate enough?    
            if FinalRhoS is not None:
                break
            
    # finish and save plot
    plt.xlabel(r"$\rho_\mathrm{S}$ [\$/\$]", fontsize = 24)
    plt.ylabel(r"Necessary debt to reach $\alpha_\mathrm{S}$ [10^9\$]", fontsize = 24)
    plt.title(r"Necessary debt for different $\rho_\mathrm{S}$", fontsize = 30, pad = 20)
    fig.savefig("Figures/rhoSvsDebts/CropAlloc_" + file + ".jpg", \
                bbox_inches = "tight", pad_inches = 1)
    plt.close()	    
	        
    return(FinalRhoS, FinalNecessaryDebt)

    
# %% ####################### AUXILIARY FUNCTIONS RHOS #########################
    
def CheckRhoSiniDebt(args, probS, rhoSini, debt_top, debt_bottom, shareDiff, accuracy,
                     console_output = None, logs_on = None):
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
        Initial guess for rhoS.
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).
    debt_bottom : float
        The debt that would be necessary for rhoS = 0.
    shareDiff : int
        The share of the final rhoS that the accuracy interval can have as 
        size (i.e. if size(accuracy interval) < 1/shareDiff * rhoS, for rhoS
        the current best guess for the correct penalty, then we use rhoS).
    accuracy : int
        Desired decimal places of accuracy of the obtained probS. (Here this
        is only used for rounding of output for the console as the correct
        probS can't be reached anyway.)
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    rhoS : float or None
        If the guess worked, this is the rhoS, else it is None.
    necessary_debt : float or None
        If the guess worked, this is the necessary debt, else it is None.

    """
        
    # the demanded accuracy in the debt is given as a share of the difference
    # between debt_top and debt_bottom
    from ModelCode.GeneralSettings import accuracy_debt
    accuracy_diff_debt = np.abs(debt_top - debt_bottom) * accuracy_debt  
    
    # TODO I think this case doesn't happen - if it does I could still
    # optimize this to use the guess to improve computational time for 
    # rhoSini == 0
    if rhoSini != 0:
        printing("     Checking guess from run with other N", console_output = console_output, logs_on = logs_on)
        status, crop_alloc, meta_sol, prob, durations = \
            SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, console_output = False, logs_on = False) 
        necessary_debt = meta_sol["necessary_debt"]
        ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                 "S", debt = necessary_debt, prefix = "Guess - ", 
                                 console_output = console_output, logs_on = logs_on)
         
        # if solution was in the top part
        if  np.abs(necessary_debt - debt_top) < accuracy_diff_debt:
            rhoScheck = rhoSini - rhoSini/shareDiff
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck, console_output = False, logs_on = False) 
            necessary_debt_check = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoScheck, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt_check, prefix = "Check - ", 
                                     console_output = console_output, logs_on = logs_on)
                
            if (necessary_debt_check - debt_top) > accuracy_diff_debt:
                return(rhoSini, necessary_debt)
            
        # if solution was in the valley
        elif np.abs(necessary_debt - debt_bottom) < accuracy_diff_debt: 
            rhoScheck1 = rhoSini - rhoSini/shareDiff
            rhoScheck2 = rhoSini + rhoSini/shareDiff
            
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck1, console_output = False, logs_on = False) 
            necessary_debt_check1 = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoScheck1, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt_check1, prefix = "Check 1 - ", 
                                     console_output = console_output, logs_on = logs_on)
                
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck2, console_output = False, logs_on = False) 
            necessary_debt_check2 = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoScheck2, meta_sol, accuracy, durations, \
                                     "S", debt = necessary_debt_check2, prefix = "Check 2 - ", 
                                     console_output = console_output, logs_on = logs_on)
                
            if necessary_debt_check1 > necessary_debt and \
                necessary_debt_check2 > necessary_debt:
                return(rhoSini, necessary_debt)
    printing("     Oops, that guess didn't work - starting from scratch\n", console_output = console_output, logs_on = logs_on)
    return(None, None)

def UpdateDebtInformation(rhoSnew, necessary_debt, debt_top, debt_bottom, \
                          shareDiff, UpperBorder, LowerBorder, \
                          rhoSvalley, debtsValley):
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
    debt_bottom : float
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
    necessary_debt :float or None
        If the current best guess for the correct rhoS is accurate enough the
        corresponding necessary debt is returned, else None.

    """

    # the demanded accuracy in the debt is given as a share of the difference
    # between debt_top and debt_bottom
    from ModelCode.GeneralSettings import accuracy_debt
    accuracy_diff_debt = np.abs(debt_top - debt_bottom) * accuracy_debt    

    # Update inforamtion on Borders 
    # if np.round(necessary_debt, accuracy_debt) == np.round(debt_bottom, accuracy_debt):
    #     LowerBorder = rhoSnew
        
    # Update inforamtion on Borders 
    if np.abs(necessary_debt - debt_bottom) < accuracy_diff_debt:
        LowerBorder = rhoSnew
    
    elif (np.abs(necessary_debt - debt_top) < accuracy_diff_debt
            and (len(rhoSvalley) == 0 or rhoSnew > rhoSvalley[-1])):
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
            interval = UpperBorder - LowerBorder
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
            interval = rhoSvalley[1] - LowerBorder
        elif debtsValley[-1] == min(debtsValley):
            rhoS = UpperBorder
            necessary_debt = debt_top
            interval = UpperBorder - rhoSvalley[-2]
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

def DebtReport(necessary_debt, debt_bottom, debt_top):
    """
    For reporting: if the debt equals to either the debt for rhoS = 0 or 
    for rhoS = inf, we want to state that.

    Parameters
    ----------
    necessary_debt : float
        The necessary debt when using rhoSnew.
    debt_bottom : float
        The debt that would be necessary for rhoS = 0.
    debt_top : float
        The debt that would be necessary for rhoS -> inf (aproximated by
        setting rhoS = 1e9).

    Returns
    -------
    debt_report : float or str
        If the debt equals either debt_top or debt_bottom, this is a str 
        with the corresponding information. Else it is the value of 
        necessary_debt.

    """
    # the demanded accuracy in the debt is given as a share of the difference
    # between debt_top and debt_bottom
    from ModelCode.GeneralSettings import accuracy_debt
    accuracy_diff_debt = np.abs(debt_top - debt_bottom) * accuracy_debt    
    
    if  np.abs(necessary_debt - debt_top) < accuracy_diff_debt:
        debt_report = "1e9"
    elif np.abs(necessary_debt - debt_bottom) < accuracy_diff_debt:
        debt_report = "0"
    else:
        debt_report = necessary_debt
    
    return(debt_report)        

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
    debt_bottom : float
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
            print("Found Valley", flush = True)
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
        First next guess for rhoS (between the current best guess and its 
        "better" neighbor).
    rhoSnew2 : float
        Second next guess for rhoS (between the current best guess and its 
        other neighbor).

    """
    
    i = debtsValley.index(min(debtsValley))
    if debtsValley[i-1] < debtsValley[i+1]:
        rhoSnew1 = (rhoSvalley[i] + rhoSvalley[i-1])/2
        rhoSnew2 = (rhoSvalley[i] + rhoSvalley[i+1])/2
    else:     
        rhoSnew1 = (rhoSvalley[i] + rhoSvalley[i+1])/2
        rhoSnew2 = (rhoSvalley[i] + rhoSvalley[i-1])/2
    
    return(rhoSnew1, rhoSnew2)  
    
# %% ####################### AUXILIARY FUNCTIONS RHOS #########################

def UpdatedRhoGuessImports(meta_sol, rhoLastUp, rhoLastDown, rhoOld, min_import, accuracy):
    """
    For MinimizeNecessaryImport this provides the next guess for the penalty.

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
    minImport : float
        The minimal import (when using the full area for the more productive 
        crop in all years)
    accuracy : int
        Desired decimal places of accuracy of the obtained probability. 

    Returns
    -------
    rhoNew : float
        The next penalty guess.
    rhoLastDown : float
        Updated version of rhoLastDown
    rhoLastUp : float
        Updated version of rhoFLastUp

    """
    
    # find next guess
    if np.round(meta_sol["necessary_import"], accuracy) > min_import:
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


# %% ###################### JOINT AUXILIARY FUNCTIONS #########################

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
        currentProb = meta_sol["probF"]
    elif probType == "S":
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
    
    return(rhoNew, rhoLastDown, rhoLastUp)

def ReportProgressFindingRho(rhoOld, meta_sol, accuracy, durations, \
                             ProbType, accuracy_int = False, debt = False, imports = False, \
                             prefix = "", console_output = None, logs_on = None):
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
    ProbType : string,F" or "S"
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
    
    from ModelCode.GeneralSettings import accuracy_import
        
    # get correct probability and unit
    if ProbType == "F":
        currentProb = meta_sol["probF"]
        unit = " $/10^3kcal"
    elif ProbType == "S":
        currentProb = meta_sol["probS"]
        unit = " $/$"
    
    # if debt is given create corresponding text piece
    if type(debt) is str:
        debt_text = ", nec. debt as for rhoS = " + debt
    elif debt:
        debt_text = ", nec. debt: " + str(np.round(debt, 4)) + " 10^9$"
    else:
        debt_text = ""
        
    if imports:
        if np.round(meta_sol["necessary_import"], accuracy_import) == imports:
            import_text = ", nec. import at min."
        else:
            import_text = ", nec. import: " + str(np.round(meta_sol["necessary_import"], accuracy_import + 5)) + " 10^12kcal"
    else:
        import_text = ""
        
        
    # if length of accuracy interval is given create corresponding text piece
    if accuracy_int:
        accuracy_text = " (current accuracy interval: " + str(np.round(accuracy_int, 3)) + ")"
    else:
        accuracy_text = ""
        
    # print information (if console_output = True)
    printing("     " + prefix + "rho" + ProbType + ": " + str(rhoOld) + unit + \
          ", prob" + ProbType + ": " + str(np.round(currentProb * 100, \
                                                    accuracy -1)) + \
          "%" + debt_text + import_text + ", time: " + str(np.round(durations[2], 2)) + "s" + accuracy_text, \
              console_output = console_output, logs_on = logs_on)
    
    return(None)

    