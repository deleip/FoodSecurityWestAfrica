#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:12:12 2021

@author: Debbora Leip
"""

from termcolor import colored
from datetime import datetime

def ReturnGeneralSettings():
    """
    Prints the current values of the general settings defined in 
    ModelCode/GeneralSettings to the console.

    Returns
    -------
    None.

    """
    
    from ModelCode.GeneralSettings import accuracyF
    from ModelCode.GeneralSettings import accuracyS
    from ModelCode.GeneralSettings import shareDiffF
    from ModelCode.GeneralSettings import shareDiffS
    from ModelCode.GeneralSettings import accuracy_debt
    from ModelCode.GeneralSettings import accuracy_import
    from ModelCode.GeneralSettings import logs_on
    from ModelCode.GeneralSettings import console_output
    from ModelCode.GeneralSettings import figsize
    from ModelCode.GeneralSettings import close_plots
    
    print("General Settings are", flush = True)
    print("\u033F "*20)
    print("  - accuracyF: accuracy that is demanded from the food security " + \
          "probability as decimal places. " + \
          "Current value: " + colored(str(accuracyF), "cyan"), flush = True)
    print("  - accuracyS: accuracy that is demanded from the solvency " + \
          "probability as decimal places. " + \
          "Current value: " + colored(str(accuracyS), "cyan"), flush = True)
    print("  - shareDiffF: accuracy of the food security penalty relative to the final rhoF. " + \
          "Current value: " + colored(str(shareDiffF), "cyan"), flush = True)
    print("  - shareDiffS: accuracy of the solvency penalty relative to the final rhoS. " + \
          "Current value: " + colored(str(shareDiffS), "cyan"), flush = True)
    print("  - accuracy_debt: accuracy of debts in cases where probS cannot be reached. " + \
          "Current value: " + colored(str(accuracy_debt), "cyan"), flush = True)
    print("  - accuracy_import: accuracy of imports in cases where probF cannot be reached. " + \
          "Current value: " + colored(str(accuracy_import), "cyan"), flush = True)
    print("  - logs_on: should model progress be logged? " + \
         "Current value: " + colored(str(logs_on), "cyan"), flush = True)
    print("  - console_output: should model progress be reported in console? " + \
          "Current value: " + colored(str(console_output), "cyan"), flush = True)
    print("  - figsize: default figsize used for figures. " + \
          "Current value: " + colored(str(figsize), "cyan"), flush = True)
    print("  - close_plots: should figures be closed after plotting? " + \
          "Current value: " + colored(str(close_plots), "cyan"), flush = True)
    
    return(None)

def ModifyGeneralSettings(accuracyF = None, \
                          accuracyS = None, \
                          shareDiffF = None, \
                          shareDiffS = None, \
                          accuracy_debt = None, \
                          accuracy_import = None, \
                          logs_on = None, \
                          console_output = None, \
                          figsize = None, \
                          close_plots = None):
    """
    Updates the values for the given general settings by rewriting the
    ModelCode/GeneralSettings.py file, keeping the last value that was given
    for unspecified settings (i.e. None).

    Parameters
    ----------
    accuracyF : float, optional
        Accuracy demanded from the food demand probability as decimal places
        (given as float, not as percentage). The default is None.
    accuracyS : float, optional
        Accuracy demanded from the solvency probability as decimal places
        (given as float, not as percentage). The default is None.
    shareDiffF : float, optional
        Accuracy of the food security penalty given thorugh size of the 
        accuracy interval: the size needs to be smaller than final rho/shareDiff. 
        The default is None.
    shareDiffS : float, optional
        Accuracy of the solvency penalty given thorugh size of the 
        accuracy interval: the size needs to be smaller than final rho/shareDiff. 
        The default is None.
    accuracy_debt : float, optional
        Accuracy of debts used in the algorithm to find the right rhoS in
        cases where probS cannot be reached (given as the share of the 
        difference between debt_bottom and debt_top). The default is None.
    accuracy_import : float, optional
        Accuracy of imports used in the algorithm to find the right rhoF in
        cases where probF cannot be reached (given as number of decimal places).
        The default is None.
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
    
    from ModelCode.GeneralSettings import accuracyF as accuracyFbefore
    from ModelCode.GeneralSettings import accuracyS as accuracySbefore
    from ModelCode.GeneralSettings import shareDiffF as shareDiffFbefore
    from ModelCode.GeneralSettings import shareDiffS as shareDiffSbefore
    from ModelCode.GeneralSettings import accuracy_debt as accuracy_debtbefore
    from ModelCode.GeneralSettings import accuracy_import as accuracy_importbefore
    from ModelCode.GeneralSettings import logs_on as logs_onFbefore
    from ModelCode.GeneralSettings import console_output as console_outputbefore
    from ModelCode.GeneralSettings import figsize as figsizebefore
    from ModelCode.GeneralSettings import close_plots as close_plotsbefore
    
    report = "Changed settings for "
    
    settings = open("ModelCode/GeneralSettings.py", "w")
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n\n")
    settings.write("# accuracy demanded from the probabilities as decimal places (given as float,\n")
    settings.write("# not as percentage)\n")
    if accuracyF is None:
        settings.write("accuracyF = " + str(accuracyFbefore) + "\n")
    else:
        settings.write("accuracyF = " + str(accuracyF) + "\n")
        if accuracyF != accuracyFbefore:
            report += "accuracyF, "
    if accuracyS is None:
        settings.write("accuracyS = " + str(accuracySbefore) + "\n\n")
    else:
        settings.write("accuracyS = " + str(accuracyS) + "\n\n")
        if accuracyS != accuracySbefore:
            report += "accuracyS, "
    settings.write("# accuracy of the penalties given thorugh size of the accuracy interval:\n")
    settings.write("# the size needs to be smaller than final rho / shareDiff\n")
    if shareDiffF is None:
        settings.write("shareDiffF = " + str(shareDiffFbefore) + "\n")
    else:
        settings.write("shareDiffF = " + str(shareDiffF) + "\n")
        if shareDiffF != shareDiffFbefore:
            report += "shareDiffF, "
    if shareDiffS is None:
        settings.write("shareDiffS = " + str(shareDiffSbefore) + "\n\n")
    else:
        settings.write("shareDiffS = " + str(shareDiffS) + "\n\n")
        if shareDiffS != shareDiffS:
            report += "shareDiffS, "
    settings.write("# accuracy of debts used in the algorithm to find the right rhoS in cases where\n")
    settings.write("# probS cannot be reached (given as the share of the difference between\n")
    settings.write("# debt_bottom and debt_top)\n")
    if accuracy_debt is None:
        settings.write("accuracy_debt = " + str(accuracy_debtbefore) + "\n\n")
    else:
        settings.write("accuracy_debt = " + str(accuracy_debt) + "\n\n")
        if accuracy_debt != accuracy_debtbefore:
             report += "accuracy_debt, "
    settings.write("# accuracy of imports used in the algorithm to find the right rhoF in cases\n")
    settings.write("# where probF cannot be reached (given as number of decimal places)\n")
    if accuracy_import is None:
        settings.write("accuracy_import = " + str(accuracy_importbefore) + "\n\n")
    else:
        settings.write("accuracy_import = " + str(accuracy_import) + "\n\n")
        if accuracy_import != accuracy_importbefore:
             report += "accuracy_debt, "
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
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n")
    settings.write("# (reset to original values)\n\n")
    settings.write("# accuracy demanded from the probabilities as decimal places (given as float,\n")
    settings.write("# not as percentage)\n")
    settings.write("accuracyF = 3\n")
    settings.write("accuracyS = 3\n\n")
    settings.write("# accuracy of the penalties given thorugh size of the accuracy interval:\n")
    settings.write("# the size needs to be smaller than final rho / shareDiff\n")
    settings.write("shareDiffF = 10\n")
    settings.write("shareDiffS = 10\n\n")
    settings.write("# accuracy of debts used in the algorithm to find the right rhoS in cases where\n")
    settings.write("# probS cannot be reached (given as the share of the difference between\n")
    settings.write("# debt_bottom and debt_top)\n")
    settings.write("accuracy_debt = 0.005\n\n")
    settings.write("# accuracy of imports used in the algorithm to find the right rhoF in cases\n")
    settings.write("# where probF cannot be reached (given as number of decimal places)\n")
    settings.write("accuracy_import = 3\n\n")
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

####

def ReturnDefaultModelSettings():
    """
    Prints the current values of the general settings defined in 
    ModelCode/GeneralSettings to the console.

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

    
    print("Defaul Model Settings are", flush = True)
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
                          ini_fund = None):
    """
    Updates the values for the given general settings by rewriting the
    ModelCode/GeneralSettings.py file, keeping the last value that was given
    for unspecified settings (i.e. None).

    Parameters
    ----------

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
    
    if report == "Changed settings for ":
        print("No changes.")
    else:
        report = report[:-2]
        print(report + ".")
   
    return(None)


def ResetDefaultModelSettings():
    """
    Rewrites the ModelCode/GeneralSettings.py with the original values.

    Returns
    -------
    None.
    """
    
    settings = open("ModelCode/DefaultModelSettings.py", "w")
    settings.write("# Default Model Settings \n")
    settings.write("# Last modified " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n\n")
    settings.write("# (reset to original values)\n\n")
    
    settings.write("# \"pro\" if desired probabilities are given and penalties are to be \n")
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
    settings.close()
    
    print("Settings reset to original values.")
    
    return(None)
