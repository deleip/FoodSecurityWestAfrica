#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:12:12 2021

@author: Debbora Leip
"""

# accuracy demanded from the probabilities as decimal places (given as float,
# not as percentage)
accuracyF = 3
accuracyS = 3

# accuracy of the penalties given thorugh size of the accuracy interval: 
# the size needa to be smaller than final rho / shareDiff
shareDiffF= 10
shareDiffS = 10

# accuracy of debts used in the algorithm to find the right rhoS in cases where
# probS cannot be reached (given as the share of the difference between 
# debt_bottom and debt_top)
accuracy_debt = 0.0005

# should model progress be logged?
logs_on = True

# figsize used for all figures
figsize = (24, 13.5)

from termcolor import colored

# TODO
def ReturnGeneralSettings():
    print("General Settings are", flush = True)
    print("\u033F "*20)
    print("     accuracyF: " + str(accuracyF) + \
          "\naccuracy that is demanded from the probabilities as decimal " + \
          "places (for probabilities given as float, not as percentage)" + \
          " - for food security probability", flush = True)
    print("     accuracyS: " + str(accuracyS) + \
          "\naccuracy that is demanded from the probabilities as decimal " + \
          "places (for probabilities given as float, not as percentage)" + \
          " - for solvency probability", flush = True)
    print("     shareDiffF: " + str(shareDiffF) + \
          "\naccuracy of the penalties given thorugh size of the accuracy " + \
          "interval: the size needa to be smaller than final rho/shareDiff" + \
          " - for food security penalty", flush = True)
    print("     shareDiffS: " + str(shareDiffS) + \
          "\naccuracy of the penalties given thorugh size of the accuracy " + \
          "interval: the size needa to be smaller than final rho/shareDiff" + \
          " - for solvency penalty", flush = True)
    print("     accuracy_debt: " + str(accuracy_debt) + \
          "\naccuracy of debts used in the algorithm to find the right " + \
          "rhoS in cases where probS cannot be reached", flush = True)
    print("     logs_on: " + str(logs_on) + \
          "\nshould model progress be logged?", flush = True)
    print("     figsize: " + str(figsize) + \
          "\ndefault figsize used for figures", flush = True)
    
    return(None)

def ChangeGeneralSettings():
    print(colored("Sorry, not implemented yet.", 'red'))
    return(None)