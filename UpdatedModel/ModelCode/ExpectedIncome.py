#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:04:14 2021

@author: Debbora Leip
"""
import numpy as np
import pickle
import sys

from ModelCode.Auxiliary import printing
from ModelCode.GetPenalties import GetInitialGuess
from ModelCode.SettingsParameters import SetParameters
from ModelCode.GetPenalties import GetRhoWrapper

# %% ############### FUNCTIONS RUNNING MODEL TO GET EXP INCOME ################

def GetExpectedIncome(settings, console_output = None):
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

    Returns
    -------
    expected_incomes :  np.array of size (T, len(k_using))
        The expected income of farmers in a scenario where the government is
        not involved.

    """   
        
    # not all settings affect the expected income (as no government is 
    # included)
    SettingsAffectingGuaranteedIncome = "k" + str(settings["k"]) + \
            "using" +  '_'.join(str(n) for n in settings["k_using"]) + \
            "num_crops" + str(settings["num_crops"]) + \
            "sim_start" + str(settings["sim_start"]) + \
            "N" + str(settings["N"]) 
    
    # open dict with all expected incomes that were calculated so far
    with open("PenaltiesAndIncome/ExpectedIncomes.txt", "rb") as fp:    
        dict_incomes = pickle.load(fp)
    
    # if expected income was already calculated for these settings, fetch it
    if SettingsAffectingGuaranteedIncome in dict_incomes.keys():
        printing("\nFetching expected income", console_output = console_output)
        expected_incomes = dict_incomes[SettingsAffectingGuaranteedIncome]
    # else calculate (and save) it
    else:
        expected_incomes = CalcExpectedIncome(settings, \
                        SettingsAffectingGuaranteedIncome, console_output = console_output)
        dict_incomes[SettingsAffectingGuaranteedIncome] = expected_incomes
        with open("PenaltiesAndIncome/ExpectedIncomes.txt", "wb") as fp:    
             pickle.dump(dict_incomes, fp)
        
    if np.sum(expected_incomes < 0) > 0:
        sys.exit("Negative expected income")
        
    printing("     Expected income per cluster in " + \
             str(settings["sim_start"] - 1) + ": " + \
             str(np.round(expected_incomes, 3)), console_output = console_output)
        
    return(expected_incomes)
       
def CalcExpectedIncome(settings, SettingsAffectingGuaranteedIncome,
                       console_output = None):
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

    Returns
    -------
    expected_incomes :  np.array of size (len(k_using),)
        The expected income of farmers in a scenario where the government is
        not involved.

    """
    
    printing("\nCalculating expected income ", console_output = console_output)
    settings_ExpIn = settings.copy()

    # change some settings: we are interested in the expected income in 2016
    settings_ExpIn["seed"] = 201120
    settings_ExpIn["yield_projection"] = "fixed"
    settings_ExpIn["pop_scenario"] = "fixed"
    settings_ExpIn["T"] = 1
    probF = 0.99
    
    # settings affecting the food demand penalty
    SettingsBasics = "k" + str(settings_ExpIn["k"]) + \
            "using" +  '_'.join(str(n) for n in settings_ExpIn["k_using"]) + \
            "num_crops" + str(settings_ExpIn["num_crops"]) + \
            "yield_projection" + str(settings_ExpIn["yield_projection"]) + \
            "sim_start" + str(settings_ExpIn["sim_start"]) + \
            "pop_scenario" + str(settings_ExpIn["pop_scenario"]) + \
            "T" + str(settings_ExpIn["T"])
    SettingsFirstGuess =  SettingsBasics + "probF" + str(probF)
    SettingsAffectingRhoF = SettingsFirstGuess + "N" + str(settings_ExpIn["N"])
    
    # first guess
    with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
        dict_rhoFs = pickle.load(fp)
    with open("PenaltiesAndIncome/resProbFs.txt", "rb") as fp:    
        dict_probFs = pickle.load(fp)
    with open("PenaltiesAndIncome/resAvgImport.txt", "rb") as fp:    
        dict_import = pickle.load(fp)
        
    rhoFini, checkedGuess = GetInitialGuess(dict_rhoFs, SettingsFirstGuess, settings["N"])
    
    # we assume that without government farmers aim for 95% probability of 
    # food security, therefore we find the right penalty for probF = 95%.
    # As we want the income in a scenario without government, the final run of
    # GetRhoF (with rohS = 0) automatically is the right run
    args, yield_information, population_information = \
        SetParameters(settings_ExpIn, console_output = False, logs_on = False)
    
    rhoF, meta_sol = GetRhoWrapper(args, probF, rhoFini, checkedGuess, "F",
                  SettingsAffectingRhoF, console_output = None, logs_on = None)
          
    dict_rhoFs[SettingsAffectingRhoF] = rhoF
    dict_probFs[SettingsAffectingRhoF] = meta_sol["probF"]
    dict_import[SettingsAffectingRhoF] = meta_sol["avg_nec_import"]
    
    # saving updated dicts
    with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:    
         pickle.dump(dict_rhoFs, fp)
    with open("PenaltiesAndIncome/resProbFs.txt", "wb") as fp:    
         pickle.dump(dict_probFs, fp)
    with open("PenaltiesAndIncome/resAvgImport.txt", "wb") as fp:     
         pickle.dump(dict_import, fp)
        
    return(meta_sol["expected_incomes"].flatten())
