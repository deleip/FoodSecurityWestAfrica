#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:04:14 2021

@author: Debbora Leip
"""
import numpy as np
import pickle
import sys

from ModelCode.Auxiliary import _printing
from ModelCode.GetPenalties import _GetInitialGuess
from ModelCode.SettingsParameters import SetParameters
from ModelCode.GetPenalties import _GetRhoWrapper

# %% ############### FUNCTIONS RUNNING MODEL TO GET EXP INCOME ################

def GetExpectedIncome(settings, console_output = None, logs_on = None):
    """
    Either loading expected income if it was already calculated for these 
    settings, or calling the function to calculate the expected income. 

    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    expected_incomes :  np.array of size (T, len(k_using))
        The expected income of farmers in a scenario where the government is
        not involved.

    """   
        
    # not all settings affect the expected income (as no government is 
    # included)
    SettingsAffectingGuaranteedIncome = "k" + str(settings["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "Crops" + str(settings["num_crops"]) + \
                "Start" + str(settings["sim_start"]) + \
                "N" + str(settings["N"])
    
    # open dict with all expected incomes that were calculated so far
    with open("PenaltiesAndIncome/ExpectedIncomes.txt", "rb") as fp:    
        dict_incomes = pickle.load(fp)
    
    # if expected income was already calculated for these settings, fetch it
    if SettingsAffectingGuaranteedIncome in dict_incomes.keys():
        _printing("\nFetching expected income", console_output = console_output, logs_on = logs_on)
        expected_incomes = dict_incomes[SettingsAffectingGuaranteedIncome]
    # else calculate (and save) it
    else:
        expected_incomes = _CalcExpectedIncome(settings, \
                        SettingsAffectingGuaranteedIncome, console_output = console_output, logs_on = logs_on)
        dict_incomes[SettingsAffectingGuaranteedIncome] = expected_incomes
        with open("PenaltiesAndIncome/ExpectedIncomes.txt", "wb") as fp:    
             pickle.dump(dict_incomes, fp)
     
    # should not happen
    if np.sum(expected_incomes < 0) > 0:
        sys.exit("Negative expected income")
        
    _printing("     Expected income per cluster in " + \
             str(settings["sim_start"] - 1) + ": " + \
             str(np.round(expected_incomes, 3)),
             console_output = console_output, logs_on = logs_on)
        
    return(expected_incomes)
       
def _CalcExpectedIncome(settings, SettingsAffectingGuaranteedIncome,
                       console_output = None, logs_on = None):
    """
    Calculating the expected income in the scenario corresponding to the 
    settings but without government.

    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    SettingsAffectingGuaranteedIncome : str
        Combining all settings that influence the expected income, used to 
        save the result for further runs.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    -------
    expected_incomes :  np.array of size (len(k_using),)
        The expected income of farmers in a scenario where the government is
        not involved.

    """
    
    _printing("\nCalculating expected income ", console_output = console_output, logs_on = logs_on)
    settings_ExpIn = settings.copy()

    # change some settings: we are interested in the expected income in 2016
    # (no need to change start year, as we set scenarios to fixed)
    settings_ExpIn["yield_projection"] = "fixed"
    settings_ExpIn["pop_scenario"] = "fixed"
    settings_ExpIn["T"] = 1
    probF = 0.99
    
    # settings affecting the food demand penalty
    SettingsBasics = "k" + str(settings_ExpIn["k"]) + \
                "Using" +  '_'.join(str(n) for n in settings_ExpIn["k_using"]) + \
                "Crops" + str(settings_ExpIn["num_crops"]) + \
                "Yield" + str(settings_ExpIn["yield_projection"]).capitalize() + \
                "Start" + str(settings_ExpIn["sim_start"]) + \
                "Pop" + str(settings_ExpIn["pop_scenario"]).capitalize() + \
                "T" + str(settings_ExpIn["T"]) + \
                "Import" + str(settings["import"]) + \
                "Seed" + str(settings["seed"])
    SettingsFirstGuess =  SettingsBasics + "ProbF" + str(probF)
    SettingsProbF = SettingsBasics + "N" + str(settings["N"])
    SettingsFinalRhoF = SettingsFirstGuess + "N" + str(settings_ExpIn["N"])
    
    # first guess
    with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
        dict_rhoFs = pickle.load(fp)
    with open("PenaltiesAndIncome/crop_allocF.txt", "rb") as fp:    
        dict_crop_allocF = pickle.load(fp)
        
    rhoFini, checkedGuess = _GetInitialGuess(dict_rhoFs, SettingsFirstGuess, settings["N"])
    
    # we assume that without government farmers aim for 99% probability of 
    # food security, therefore we find the right penalty for probF = 99%.
    # As we want the income in a scenario without government, the resulting run
    # of GetRhoF (with rohS = 0) automatically is the right run
    args, yield_information, population_information = \
        SetParameters(settings_ExpIn, console_output = False, logs_on = False)
    
    rhoF, meta_solF, crop_allocF = _GetRhoWrapper(args, probF, rhoFini, checkedGuess, "F", SettingsProbF,
                  SettingsFinalRhoF, console_output = False, logs_on = False)
          
    dict_rhoFs[SettingsFinalRhoF] = rhoF
    dict_crop_allocF[SettingsFinalRhoF] = crop_allocF
    
    # saving updated dicts
    with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:    
         pickle.dump(dict_rhoFs, fp)
    with open("PenaltiesAndIncome/crop_allocF.txt", "wb") as fp:     
         pickle.dump(dict_crop_allocF, fp)
        
    return(meta_solF["avg_profits_preTax"].flatten())
