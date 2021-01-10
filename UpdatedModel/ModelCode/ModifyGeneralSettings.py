#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:12:12 2021

@author: Debbora Leip
"""

from termcolor import colored
from datetime import datetime

def ReturnGeneralSettings():
    
    from ModelCode.GeneralSettings import accuracyF
    from ModelCode.GeneralSettings import accuracyS
    from ModelCode.GeneralSettings import shareDiffF
    from ModelCode.GeneralSettings import shareDiffS
    from ModelCode.GeneralSettings import accuracy_debt
    from ModelCode.GeneralSettings import logs_on
    from ModelCode.GeneralSettings import console_output
    from ModelCode.GeneralSettings import figsize
    
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
    print("  - logs_on: should model progress be logged? " + \
         "Current value: " + colored(str(logs_on), "cyan"), flush = True)
    print("  - console_output: should model progress be reported in console? " + \
          "Current value: " + colored(str(console_output), "cyan"), flush = True)
    print("  - figsize: default figsize used for figures. " + \
          "Current value: " + colored(str(figsize), "cyan"), flush = True)
    
    return(None)

def ModifyGeneralSettings(accuracyF = None, \
                          accuracyS = None, \
                          shareDiffF = None, \
                          shareDiffS = None, \
                          accuracy_debt = None, \
                          logs_on = None, \
                          console_output = None, \
                          figsize = None):
    
    from ModelCode.GeneralSettings import accuracyF as accuracyFbefore
    from ModelCode.GeneralSettings import accuracyS as accuracySbefore
    from ModelCode.GeneralSettings import shareDiffF as shareDiffFbefore
    from ModelCode.GeneralSettings import shareDiffS as shareDiffSbefore
    from ModelCode.GeneralSettings import accuracy_debt as accuracy_debtbefore
    from ModelCode.GeneralSettings import logs_on as logs_onFbefore
    from ModelCode.GeneralSettings import console_output as console_outputbefore
    from ModelCode.GeneralSettings import figsize as figsizebefore
    
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
        settings.write("figsize = " + str(figsizebefore))
    else:
        settings.write("figsize = " + str(figsize))
        if figsize != figsizebefore:
            report += "figsize, "
    settings.close()
    
    if report == "Changed settings for ":
        print("No changes.")
    else:
        report = report[:-2]
        print(report + ".")
    
    # # opening and closing all code files, such that the local settings will
    # # be updated
    # open("ModelCode/AnalysisAndComparison.txt", "a").close()
    # open("ModelCode/Auxiliary.txt", "a").close()
    # open("ModelCode/CompleteModelCall.txt", "a").close()
    # open("ModelCode/ExpectedIncome.txt", "a").close()
    # open("ModelCode/GetPenalties.txt", "a").close()
    # open("ModelCode/GroupingClusters.txt", "a").close()
    # open("ModelCode/GroupingClusters.txt", "a").close()
    # open("ModelCode/GroupingClusters.txt", "a").close()
    # open("ModelCode/GroupingClusters.txt", "a").close()
    # open("ModelCode/GroupingClusters.txt", "a").close()
    
    
    return(None)

def ResetGeneralSettings():
    
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
    settings.write("# should model progress be logged?\n")
    settings.write("logs_on = True\n")
    settings.write("# should model progress be reported in console?" + "\n")
    settings.write("console_output = True\n\n")
    settings.write("# figsize used for all figures\n")
    settings.write("figsize = (24, 13.5)")
    settings.close()
    
    print("Settings reset to original values.")
    
    return(None)