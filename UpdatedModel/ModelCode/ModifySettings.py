#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:12:12 2021

@author: Debbora Leip
"""

from termcolor import colored
from datetime import datetime

# %% ########### FUNCTIONS TO REPORT OR MODIFY GENERAL SETTINGS ############

def ReturnGeneralSettings():
    """
    Reports the current values of the general settings defined in 
    ModelCode/GeneralSettings to the console.

    Returns
    -------
    None.

    """
    
    from ModelCode.GeneralSettings import accuracyF_demandedProb
    from ModelCode.GeneralSettings import accuracyS_demandedProb
    from ModelCode.GeneralSettings import accuracyF_maxProb
    from ModelCode.GeneralSettings import accuracyS_maxProb
    from ModelCode.GeneralSettings import accuracyF_rho
    from ModelCode.GeneralSettings import accuracyS_rho
    from ModelCode.GeneralSettings import accuracy_help
    from ModelCode.GeneralSettings import logs_on
    from ModelCode.GeneralSettings import console_output
    from ModelCode.GeneralSettings import figsize
    from ModelCode.GeneralSettings import close_plots
    
    print("General settings are", flush = True)
    print("\u033F "*20)
    print("  - accuracyF_demandedProb: accuracy that is demanded from the food security " + \
          "probability as share of target probability. " + \
          "Current value: " + colored(str(accuracyF_demandedProb), "cyan"), flush = True)
    print("  - accuracyS_demandedProb: accuracy that is demanded from the solvency " + \
          "probability as share of target probability. " + \
          "Current value: " + colored(str(accuracyS_demandedProb), "cyan"), flush = True)
    print("  - accuracyF_maxProb: accuracy that is demanded from the food security " + \
          "probability as share of maximum probability. " + \
          "Current value: " + colored(str(accuracyF_maxProb), "cyan"), flush = True)
    print("  - accuracyS_maxProb: accuracy that is demanded from the solvency " + \
          "probability as share of maximum probability. " + \
          "Current value: " + colored(str(accuracyS_maxProb), "cyan"), flush = True)
    print("  - accuracyF_rho: accuracy of the food security penalty relative to the final rhoF. " + \
          "Current value: " + colored(str(accuracyF_rho), "cyan"), flush = True)
    print("  - accuracyS_rho: accuracy of the solvency penalty relative to the final rhoS. " + \
          "Current value: " + colored(str(accuracyS_rho), "cyan"), flush = True)
    print("  - accuracy_help: accuracy of import/debt in cases where probF/S cannot be reached. " + \
          "Current value: " + colored(str(accuracy_help), "cyan"), flush = True)
    print("  - logs_on: should model progress be logged? " + \
         "Current value: " + colored(str(logs_on), "cyan"), flush = True)
    print("  - console_output: should model progress be reported in console? " + \
          "Current value: " + colored(str(console_output), "cyan"), flush = True)
    print("  - figsize: default figsize used for figures. " + \
          "Current value: " + colored(str(figsize), "cyan"), flush = True)
    print("  - close_plots: should figures be closed after plotting? " + \
          "Current value: " + colored(str(close_plots), "cyan"), flush = True)
    
    return(None)

def ModifyGeneralSettings(accuracyF_demandedProb = None, 
                          accuracyS_demandedProb = None,
                          accuracyF_maxProb = None, 
                          accuracyS_maxProb = None,
                          accuracyF_rho = None,
                          accuracyS_rho = None, 
                          accuracy_help = None,
                          logs_on = None,
                          console_output = None,
                          figsize = None,
                          close_plots = None):
    """
    Updates the values for the given general settings by rewriting the
    ModelCode/GeneralSettings.py file, keeping the last value that was given
    for unspecified settings (i.e. None).

    Parameters
    ----------
    accuracyF_demandedProb : float, optional
        Accuracy demanded from the food demand probability as share of demanded
        probability (for probability method). The default is None.
    accuracyS_demandedProb : float, optional
        Accuracy demanded from the solvency probability as share of demanded
        probability (for probability method). The default is None.
    accuracyF_maxProb : float, optional
        Accuracy demanded from the food demand probability as share of maximum
        probability (for maxProb method). The default is None.
    accuracyS_maxProb : float, optional
        Accuracy demanded from the solvency probability as share of maximum
        probability (for maxProb method). The default is None.
    accuracyF_rho : float, optional
        Accuracy of the food security penalty given thorugh size of the accuracy
        interval: the size needs to be smaller than final rhoF * accuracyF_rho. 
        The default is None.
    accuracyS_rho : float, optional
        Accuracy of the solvency penalty given thorugh size of the accuracy
        interval: the size needs to be smaller than final rhoS * accuracyS_rho. 
        The default is None.
    accuracy_help : float, optional
        If method "MinHelp" is used to find the correct penalty, this defines the 
        accuracy demanded from the resulting necessary help in terms distance
        to the minimal necessary help, given this should be the accuracy demanded from the 
        final average necessary help (given as share of the difference between 
        final necessary help and the minimum nevessary help). The default is None.
    logs_on : boolean, optional
        Should model progress be logged?. The default is None.
    console_output : boolean, optional
        Should model progress be reported in console?. The default is None.
    figsize : tuple, optional
        figsize used for all figures. The default is None.
    close_plots: boolean, optional
        Should plots be closed after plotting (and saving)? The default is None.

    Returns
    -------
    None.

    """
    
    from ModelCode.GeneralSettings import accuracyF_targetProb as accuracyF_demandedProbFbefore
    from ModelCode.GeneralSettings import accuracyS_targetProb as accuracyS_demandedProbFbefore
    from ModelCode.GeneralSettings import accuracyF_maxProb as accuracyF_maxProbFbefore
    from ModelCode.GeneralSettings import accuracyS_maxProb as accuracyS_maxProbFbefore
    from ModelCode.GeneralSettings import accuracyF_rho as accuracyF_rhobefore
    from ModelCode.GeneralSettings import accuracyS_rho as accuracyS_rhobefore
    from ModelCode.GeneralSettings import accuracy_help as accuracy_helpbefore
    from ModelCode.GeneralSettings import logs_on as logs_onFbefore
    from ModelCode.GeneralSettings import console_output as console_outputbefore
    from ModelCode.GeneralSettings import figsize as figsizebefore
    from ModelCode.GeneralSettings import close_plots as close_plotsbefore
    
    report = "Changed settings for "
    
    settings = open("ModelCode/GeneralSettings.py", "w")
    settings.write("# General Settings \n")
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n\n")
    settings.write("# accuracy demanded from the target probabilities (given as share of\n")
    settings.write("# target probability)\n")
    if accuracyF_demandedProb is None:
        settings.write("accuracyF_demandedProb = " + str(accuracyF_demandedProbFbefore) + "\n")
    else:
        settings.write("accuracyF_demandedProb = " + str(accuracyF_demandedProb) + "\n")
        if accuracyF_demandedProb != accuracyF_demandedProbFbefore:
            report += "accuracyF_demandedProb, "
    if accuracyS_demandedProb is None:
        settings.write("accuracyS_demandedProb = " + str(accuracyS_demandedProbFbefore) + "\n\n")
    else:
        settings.write("accuracyS_demandedProb = " + str(accuracyS_demandedProb) + "\n\n")
        if accuracyS_demandedProb != accuracyS_demandedProbFbefore:
            report += "accuracyS_demandedProb, "
    settings.write("# accuracy demanded from the maximum probabilities (given as share of\n")
    settings.write("# maximum probability))\n")
    if accuracyF_maxProb is None:
        settings.write("accuracyF_maxProb = " + str(accuracyF_maxProbFbefore) + "\n")
    else:
        settings.write("accuracyF_maxProb = " + str(accuracyF_maxProb) + "\n")
        if accuracyF_maxProb != accuracyF_maxProbFbefore:
            report += "accuracyF_maxProb, "
    if accuracyS_maxProb is None:
        settings.write("accuracyS_maxProb = " + str(accuracyS_maxProbFbefore) + "\n\n")
    else:
        settings.write("accuracyS_maxProb = " + str(accuracyS_maxProb) + "\n\n")
        if accuracyS_maxProb != accuracyS_maxProbFbefore:
            report += "accuracyS_maxProb, "
    settings.write("# accuracy of the penalties given thorugh size of the accuracy interval:\n")
    settings.write("# the size needs to be smaller than final rho * shareDiff\n")
    if accuracyF_rho is None:
        settings.write("accuracyF_rho = " + str(accuracyF_rhobefore) + "\n")
    else:
        settings.write("accuracyF_rho = " + str(accuracyF_rho) + "\n")
        if accuracyF_rho != accuracyF_rhobefore:
            report += "accuracyF_rho, "
    if accuracyS_rho is None:
        settings.write("accuracyS_rho = " + str(accuracyS_rhobefore) + "\n\n")
    else:
        settings.write("accuracyS_rho = " + str(accuracyS_rho) + "\n\n")
        if accuracyS_rho != accuracyS_rhobefore:
            report += "accuracyS_rho, "
    settings.write("# if penalty is found according to import/debt, what accuracy should be used \n")
    settings.write("# (share of diff between max and min import/debt)\n")
    if accuracy_help is None:
        settings.write("accuracy_help = " + str(accuracy_helpbefore) + "\n\n")
    else:
        settings.write("accuracy_help = " + str(accuracy_help) + "\n\n")
        if accuracy_help != accuracy_helpbefore:
             report += "accuracy_help, "
    settings.write("# should model progress be logged?\n")
    if logs_on is None:
        settings.write("logs_on = " + str(logs_onFbefore) + "\n")
    else:
        settings.write("logs_on = " + str(logs_on) + "\n")
        if logs_on != logs_onFbefore:
            report += "logs_on, "
    settings.write("# should model progress be reported in console?" + "\n")
    if console_output is None:
        settings.write("console_output = " + str(console_outputbefore) + "\n\n")
    else:
        settings.write("console_output = " + str(console_output) + "\n\n")
        if console_output != console_outputbefore:
            report += "console_output, "
    settings.write("# figsize used for all figures\n")
    if figsize is None:
        settings.write("figsize = " + str(figsizebefore) + "\n\n")
    else:
        settings.write("figsize = " + str(figsize) + "\n\n")
        if figsize != figsizebefore:
            report += "figsize, "
    settings.write("# close figures after plotting\n")
    if close_plots is None:
        settings.write("close_plots = " + str(close_plotsbefore))
    else:
        settings.write("close_plots = " + str(close_plots))
        if close_plots != close_plotsbefore:
            report += "close_plots, "
    settings.close()
    
    if report == "Changed settings for ":
        print("No changes.")
    else:
        report = report[:-2]
        print(report + ".")
   
    return(None)

def ResetGeneralSettings():
    """
    Rewrites the ModelCode/GeneralSettings.py with the original values.

    Returns
    -------
    None.

    """
    
    settings = open("ModelCode/GeneralSettings.py", "w")
    settings.write("# General Settings \n")
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n")
    settings.write("# (reset to original values)\n\n")
    settings.write("# accuracy demanded from the target probabilities (given as share of\n")
    settings.write("# target probability)\n")
    settings.write("accuracyF_demandedProb = 0.002\n")
    settings.write("accuracyS_demandedProb = 0.002\n\n")
    settings.write("# accuracy demanded from the maximum probabilities (given as share of\n")
    settings.write("# maximum probability)\n")
    settings.write("accuracyF_maxProb = 0.005\n")
    settings.write("accuracyS_maxProb = 0.005\n\n")
    settings.write("# accuracy of the penalties given thorugh size of the accuracy interval:\n")
    settings.write("# the size needs to be smaller than final rho * accuracy_rho\n")
    settings.write("accuracyF_rho = 0.05\n")
    settings.write("accuracyS_rho = 0.05\n\n")
    settings.write("# if penalty is found according to import/debt, what accuracy should be used \n")
    settings.write("# (share of diff between max and min import/debt)\n")
    settings.write("accuracy_help = 0.01\n\n")
    settings.write("# should model progress be logged?\n")
    settings.write("logs_on = True\n")
    settings.write("# should model progress be reported in console?" + "\n")
    settings.write("console_output = True\n\n")
    settings.write("# figsize used for all figures\n")
    settings.write("figsize = (24, 13.5)\n\n")
    settings.write("# close figures after plotting\n")
    settings.write("close_plots = True")
    settings.close()
    
    print("Settings reset to original values.")
    
    return(None)

# %% ####### FUNCTIONS TO REPORT OR MODIFY DEFAULT MDEOL SETTINGS #######

def ReturnDefaultModelSettings():
    """
    Reports the current values of the general settings defined in 
    ModelCode/DefaultModelSettings to the console.

    Returns
    -------
    None.

    """
       
    from ModelCode.DefaultModelSettings import PenMet
    from ModelCode.DefaultModelSettings import probF
    from ModelCode.DefaultModelSettings import probS
    from ModelCode.DefaultModelSettings import rhoF
    from ModelCode.DefaultModelSettings import rhoS
    from ModelCode.DefaultModelSettings import k
    from ModelCode.DefaultModelSettings import k_using
    from ModelCode.DefaultModelSettings import num_crops
    from ModelCode.DefaultModelSettings import yield_projection
    from ModelCode.DefaultModelSettings import sim_start
    from ModelCode.DefaultModelSettings import pop_scenario
    from ModelCode.DefaultModelSettings import risk
    from ModelCode.DefaultModelSettings import N
    from ModelCode.DefaultModelSettings import validation_size
    from ModelCode.DefaultModelSettings import T
    from ModelCode.DefaultModelSettings import seed
    from ModelCode.DefaultModelSettings import tax
    from ModelCode.DefaultModelSettings import perc_guaranteed
    from ModelCode.DefaultModelSettings import ini_fund
    from ModelCode.DefaultModelSettings import food_import

    
    print("Defaul model settings are", flush = True)
    print("\u033F "*25)
    print("  - PenMet: " + PenMet, flush = True)
    print("  - probF: " + str(probF), flush = True)
    print("  - probS: " + str(probS), flush = True)
    print("  - rhoF: " + str(rhoF), flush = True)
    print("  - rhoS: " + str(rhoS), flush = True)
    print("  - k: " + str(k), flush = True)
    print("  - k_using: " + str(k_using), flush = True)
    print("  - num_crops: " + str(num_crops), flush = True)
    print("  - yield_projection: " + yield_projection, flush = True)
    print("  - sim_start: " + str(sim_start), flush = True)
    print("  - pop_scenario: " + pop_scenario, flush = True)
    print("  - risk: " + str(risk), flush = True)
    print("  - N: " + str(N), flush = True)
    print("  - validation_size: " + str(validation_size), flush = True)
    print("  - T: " + str(T), flush = True)
    print("  - seed: " + str(seed), flush = True)
    print("  - tax: " + str(tax), flush = True)
    print("  - perc_guaranteed: " + str(perc_guaranteed), flush = True)
    print("  - ini_fund: " + str(ini_fund), flush = True)
    print("  - food_import: " + str(food_import), flush = True)
    
    return(None)

def ModifyDefaultModelSettings(PenMet = None, \
                          probF = None, \
                          probS = None, \
                          rhoF = None, \
                          rhoS = None, \
                          k = None, \
                          k_using = None, \
                          num_crops = None, \
                          yield_projection = None, \
                          sim_start = None, \
                          pop_scenario = None, \
                          risk = None, \
                          N = None, \
                          validation_size = None, \
                          T = None, \
                          seed = None, \
                          tax = None, \
                          perc_guaranteed = None, \
                          ini_fund = None,
                          food_import = None):
    """
    Updates the values for the given default model settings by rewriting the
    ModelCode/DefaultModelSettings.py file, keeping the last value that was given
    for unspecified settings (i.e. None).

    Parameters
    ----------
    PenMet : str, optional
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly. The default is None.
    probF : float, optional
        demanded probability of keeping the food demand constraint.
        The default is defined in
        ModelCode/DefaultModelSettings.py.
    probS : float, optional
        demanded probability of keeping the solvency constraint).
    The default is defined in 
        ModelCode/DefaultModelSettings.py.
    rhoF : float, optional 
        The initial value for rhoF. The default is None.
    rhoS : float or None, optional 
        The initial value for rhoS. The default is None.
    k : int, optional
        Number of clusters in which the area is to be devided. 
        The default is None.
    k_using : "all" or a list of int i\in{1,...,k}, optional
        Specifies which of the clusters are to be considered in the model. 
        The default is None.
    num_crops : int, optional
        The number of crops that are used. The default is None.
    yield_projection : "fixed" or "trend", optional
        Specifies which yield scenario should be used. 
        The default is None.
    sim_start : int, optional
        The first year of the simulation. The default is None.
    pop_scenario : str, optional
        Specifies which population scenario should be used. 
        The default is None.
    risk : int, optional
        The risk level that is covered by the government. 
        The default is None.
    N : int, optional
        Number of yield samples to be used to approximate the expected value
        in the original objective function. The default is None.
    validation_size : int, optional
        Sample size used for validation.
        The default is None.
    T : int, optional
        Number of years to cover in the simulation. The default is None.
    seed : int, optional
        Seed to allow for reproduction of the results. The default is None.
    tax : float, optional
        Tax rate to be paied on farmers profits. The default is None
    perc_guaranteed : float, optional
        The percentage that determines how high the guaranteed income will be 
        depending on the expected income of farmers in a scenario excluding
        the government. The default is None.
    ini_fund : float, optional
        Initial fund size. The default is None.
    food_import : float, optional
        Amount of food that is imported (and therefore substracted from the
        food demand). The default is None.
        
    Returns
    -------
    None.

    """
    
    from ModelCode.DefaultModelSettings import PenMet as PenMetbefore
    from ModelCode.DefaultModelSettings import probF as probFbefore
    from ModelCode.DefaultModelSettings import probS as probSbefore
    from ModelCode.DefaultModelSettings import rhoF as rhoFbefore
    from ModelCode.DefaultModelSettings import rhoS as rhoSbefore
    from ModelCode.DefaultModelSettings import k as kbefore
    from ModelCode.DefaultModelSettings import k_using as k_usingbefore
    from ModelCode.DefaultModelSettings import num_crops as num_cropsbefore
    from ModelCode.DefaultModelSettings import yield_projection as yield_projectionbefore
    from ModelCode.DefaultModelSettings import sim_start as sim_startbefore
    from ModelCode.DefaultModelSettings import pop_scenario as pop_scenariobefore
    from ModelCode.DefaultModelSettings import risk as riskbefore
    from ModelCode.DefaultModelSettings import N as Nbefore
    from ModelCode.DefaultModelSettings import validation_size as validation_sizebefore
    from ModelCode.DefaultModelSettings import T as Tbefore
    from ModelCode.DefaultModelSettings import seed as seedbefore
    from ModelCode.DefaultModelSettings import tax as taxbefore
    from ModelCode.DefaultModelSettings import perc_guaranteed as perc_guaranteedbefore
    from ModelCode.DefaultModelSettings import ini_fund as ini_fundbefore
    from ModelCode.DefaultModelSettings import food_import as food_importbefore
    
    report = "Changed settings for "
    
    settings = open("ModelCode/DefaultModelSettings.py", "w")
    settings.write("# Default Model Settings \n")
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n\n")
    
    settings.write("# \"pro\" if desired probabilities are given and penalties are to be \n")
    settings.write("# calculated accordingly. \"penalties\" if input penalties are to be used \n")
    settings.write("# directly \n")
    if PenMet is None:
        settings.write("PenMet = \"" + str(PenMetbefore) + "\"\n\n")
    else:
        settings.write("PenMet = \"" + str(PenMet) + "\"\n\n")
        if PenMet != PenMetbefore:
            report += "PenMet, "   
            
    settings.write("# demanded probabilities for food security and solvency \n")
    if probF is None:
        settings.write("probF = " + str(probFbefore) + "\n")
    else:
        settings.write("probF = " + str(probF) + "\n")
        if probF != probFbefore:
            report += "probF, "   
    
    if probS is None:
        settings.write("probS = " + str(probSbefore) + "\n\n")
    else:
        settings.write("probS = " + str(probS) + "\n\n")
        if probS != probSbefore:
            report += "probS, " 
    
    settings.write("# penalties (if PenMet == \"prob\" used as initial guesses to calculate the \n")
    settings.write("# correct penalties) \n")
    if rhoF is None:
        settings.write("rhoF = " + str(rhoFbefore) + "\n")
    else:
        settings.write("rhoF = " + str(rhoF) + "\n")
        if rhoF != rhoFbefore:
            report += "rhoF, "   
    
    if rhoS is None:
        settings.write("rhoS = " + str(rhoSbefore) + "\n\n")
    else:
        settings.write("rhoS = " + str(rhoS) + "\n\n")
        if rhoS != rhoSbefore:
            report += "rhoS, " 
            
    settings.write("# number of clusters in which the area is devided \n")
    if k is None:
        settings.write("k = " + str(kbefore) + "\n\n")
    else:
        settings.write("k = " + str(k) + "\n\n")
        if k != kbefore:
            report += "k, "  

    settings.write("# clusters considered in the model run \n")
    if k_using is None:
        settings.write("k_using = " + str(k_usingbefore) + "\n\n")
    else:
        settings.write("k_using = " + str(k_using) + "\n\n")
        if k_using != k_usingbefore:
            report += "k_using, " 

    settings.write("# number of crops considered \n")
    if num_crops is None:
        settings.write("num_crops = " + str(num_cropsbefore) + "\n\n")
    else:
        settings.write("num_crops = " + str(num_crops) + "\n\n")
        if num_crops != num_cropsbefore:
            report += "num_crops, " 

    settings.write("# yield projections to use (\"fixed\" or \"trend\") \n")
    if yield_projection is None:
        settings.write("yield_projection = \"" + str(yield_projectionbefore) + "\"\n\n")
    else:
        settings.write("yield_projection = \"" + str(yield_projection) + "\"\n\n")
        if yield_projection != yield_projectionbefore:
            report += "yield_projection, "

    settings.write("# first year of simulation \n")
    if sim_start is None:
        settings.write("sim_start = " + str(sim_startbefore) + "\n\n")
    else:
        settings.write("sim_start = " + str(sim_start) + "\n\n")
        if sim_start != sim_startbefore:
            report += "sim_start, "

    settings.write("# population scenario to use ('fixed', 'Medium', 'High', 'Low', \n")
    settings.write("# 'ConstantFertility', 'InstantReplacement', 'ZeroMigration' \n")
    settings.write("# 'ConstantMortality', 'NoChange' and 'Momentum') \n")
    if pop_scenario is None:
        settings.write("pop_scenario = \"" + str(pop_scenariobefore) + "\"\n\n")
    else:
        settings.write("pop_scenario = \"" + str(pop_scenario) + "\"\n\n")
        if pop_scenario != pop_scenariobefore:
            report += "pop_scenario, "

    settings.write("# risk level that is covered by the government \n")
    if risk is None:
        settings.write("risk = " + str(riskbefore) + "\n\n")
    else:
        settings.write("risk = " + str(risk) + "\n\n")
        if risk != riskbefore:
            report += "risk, "

    settings.write("# sample size \n")
    if N is None:
        settings.write("N = " + str(Nbefore) + "\n\n")
    else:
        settings.write("N = " + str(N) + "\n\n")
        if N != Nbefore:
            report += "N, "

    settings.write("# sample siize for validation (if None, no validation is done) \n")
    if validation_size is None:
        settings.write("validation_size = " + str(validation_sizebefore) + "\n\n")
    else:
        settings.write("validation_size = " + str(validation_size) + "\n\n")
        if validation_size != validation_sizebefore:
            report += "validation_size, "

    settings.write("# number of years to simulate \n")
    if T is None:
        settings.write("T = " + str(Tbefore) + "\n\n")
    else:
        settings.write("T = " + str(T) + "\n\n")
        if T != Tbefore:
            report += "T, "

    settings.write("# seed for generation of yield samples \n")
    if seed is None:
        settings.write("seed = " + str(seedbefore) + "\n\n")
    else:
        settings.write("seed = " + str(seed) + "\n\n")
        if seed != seedbefore:
            report += "seed, "

    settings.write("# tax rate to be paied on farmers profit \n")
    if tax is None:
        settings.write("tax = " + str(taxbefore) + "\n\n")
    else:
        settings.write("tax = " + str(tax) + "\n\n")
        if tax != taxbefore:
            report += "tax, "

    settings.write("# the percentage that determines how high the guaranteed income will be \n")
    settings.write("# depending on the expected income \n")
    if perc_guaranteed is None:
        settings.write("perc_guaranteed = " + str(perc_guaranteedbefore) + "\n\n")
    else:
        settings.write("perc_guaranteed = " + str(perc_guaranteed) + "\n\n")
        if perc_guaranteed != perc_guaranteedbefore:
            report += "perc_guaranteed, "

    settings.write("# tax rate to be paied on farmers profit \n")
    if ini_fund is None:
        settings.write("ini_fund = " + str(ini_fundbefore) + "\n\n")
    else:
        settings.write("ini_fund = " + str(ini_fund) + "\n\n")
        if ini_fund != ini_fundbefore:
            report += "ini_fund, "
    settings.close()
    
    settings.write("# food import that will be subtracted from demand in each year \n")
    if food_import is None:
        settings.write("food_import = " + str(food_importbefore) + "\n\n")
    else:
        settings.write("food_import = " + str(food_import) + "\n\n")
        if food_import != food_importbefore:
            report += "food_import, "
    settings.close()
    
    if report == "Changed settings for ":
        print("No changes.")
    else:
        report = report[:-2]
        print(report + ".")
   
    return(None)


def ResetDefaultModelSettings():
    """
    Rewrites the ModelCode/DefaultModelSettings.py with the original values.

    Returns
    -------
    None.
    """
    
    settings = open("ModelCode/DefaultModelSettings.py", "w")
    settings.write("# Default Model Settings \n")
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n")
    settings.write("# (reset to original values)\n\n")
    
    settings.write("# \"prob\" if desired probabilities are given and penalties are to be \n")
    settings.write("# calculated accordingly. \"penalties\" if input penalties are to be used \n")
    settings.write("# directly \n")
    settings.write("PenMet = \"prob\" \n\n")
            
    settings.write("# demanded probabilities for food security and solvency \n")
    settings.write("probF = 0.99 \n")
    settings.write("probS = 0.95 \n\n")
    
    settings.write("# penalties (if PenMet == \"prob\" used as initial guesses to calculate the \n")
    settings.write("# correct penalties) \n")
    settings.write("rhoF = None \n")
    settings.write("rhoS = None \n\n")
            
    settings.write("# number of clusters in which the area is devided \n")
    settings.write("k = 9 \n\n")

    settings.write("# clusters considered in the model run \n")
    settings.write("k_using = [3] \n\n")

    settings.write("# number of crops considered \n")
    settings.write("num_crops = 2 \n\n")

    settings.write("# yield projections to use (\"fixed\" or \"trend\") \n")
    settings.write("yield_projection = \"fixed\" \n\n")

    settings.write("# first year of simulation \n")
    settings.write("sim_start = 2017 \n\n")

    settings.write("# population scenario to use ('fixed', 'Medium', 'High', 'Low', \n")
    settings.write("# 'ConstantFertility', 'InstantReplacement', 'ZeroMigration' \n")
    settings.write("# 'ConstantMortality', 'NoChange' and 'Momentum') \n")
    settings.write("pop_scenario = \"fixed\" \n\n")

    settings.write("# risk level that is covered by the government \n")
    settings.write("risk = 0.05 \n\n")

    settings.write("# sample size \n")
    settings.write("N = 10000 \n\n")

    settings.write("# sample siize for validation (if None, no validation is done) \n")
    settings.write("validation_size = None \n\n")

    settings.write("# number of years to simulate \n")
    settings.write("T = 20 \n\n")

    settings.write("# seed for generation of yield samples \n")
    settings.write("seed = 201120 \n\n")

    settings.write("# tax rate to be paied on farmers profit \n")
    settings.write("tax = 0.01 \n\n")

    settings.write("# the percentage that determines how high the guaranteed income will be \n")
    settings.write("# depending on the expected income \n")
    settings.write("perc_guaranteed = 0.9 \n\n")

    settings.write("# tax rate to be paied on farmers profit \n")
    settings.write("ini_fund = 0 \n\n")
    
    settings.write("# food import that will be subtracted from demand in each year \n")
    settings.write("food_import = 0 \n\n")
    settings.close()
    
    print("Settings reset to original values.")
    
    return(None)
