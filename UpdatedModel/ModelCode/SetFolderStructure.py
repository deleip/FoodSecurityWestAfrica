#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:44:33 2021

@author: Debbora Leip
"""

import os
import warnings as warn
import pickle
import shutil
from termcolor import colored

from ModelCode.PandaGeneration import SetUpPandaDicts
from ModelCode.PandaGeneration import CreateEmptyPanda

# %% ################ SETTING UP FOLDER STRUCTURE FOR RESULTS #################

def CheckFolderStructure():
    """
    This functions checks whether all folders that are needed are present and
    generates them if not. If dictionaries to save expected income, penalties 
    or validation values are not yet present empty dictionaries are put in 
    place. If input data is missing the function throws an error.
    """
    if not os.path.isdir("ModelLogs"):
        os.mkdir("ModelLogs")
        
    if not os.path.isdir("Figures"):
        os.mkdir("Figures")
    
    def __GroupingPlotFolders(main, a = True):
        if not os.path.isdir("Figures/" + main):
            os.mkdir("Figures/" + main)
        if a:
            if not os.path.isdir("Figures/" + main + "/CompareCropAllocs"):
                os.mkdir("Figures/" + main + "/CompareCropAllocs")
            if not os.path.isdir("Figures/" + main + "/CompareCropAllocsRiskPooling"):
                os.mkdir("Figures/" + main + "/CompareCropAllocsRiskPooling")
            if not os.path.isdir("Figures/" + main + "/PandaPlots"):
                os.mkdir("Figures/" + main + "/PandaPlots")
            main = main + "/PandaPlots"
        if not os.path.isdir("Figures/" + main + "/Aggregated"):
            os.mkdir("Figures/" + main + "/Aggregated")
        if not os.path.isdir("Figures/" + main + "/All"):
            os.mkdir("Figures/" + main + "/All")
        if not os.path.isdir("Figures/" + main + "/Median"):
            os.mkdir("Figures/" + main + "/Median")
        if a:
            if not os.path.isdir("Figures/" + main + "/Other"):
                os.mkdir("Figures/" + main + "/Other")
        
    if not os.path.isdir("Figures/ClusterGroups"):
        os.mkdir("Figures/ClusterGroups")
    if not os.path.isdir("Figures/CropAllocs"):
        os.mkdir("Figures/CropAllocs")
    if not os.path.isdir("Figures/Samples"):
        os.mkdir("Figures/Samples")
    if not os.path.isdir("Figures/GetPenaltyFigures"):
        os.mkdir("Figures/GetPenaltyFigures")
    if not os.path.isdir("Figures/GetPenaltyFigures/rhoF"):
        os.mkdir("Figures/GetPenaltyFigures/rhoF")
    if not os.path.isdir("Figures/GetPenaltyFigures/rhoS"):
        os.mkdir("Figures/GetPenaltyFigures/rhoS")
    __GroupingPlotFolders("DissimilarAdjacent")
    __GroupingPlotFolders("SimilarAdjacent")
    __GroupingPlotFolders("DissimilarNonAdjacent")
    __GroupingPlotFolders("SimilarNonAdjacent")
    __GroupingPlotFolders("ComparingScenarios", a = False)
        
    if not os.path.isdir("InputData"):
        warn.warn("You are missing the input data")
        exit()
        
    if not os.path.isdir("InputData/Clusters"):
        warn.warn("You are missing the input data on the clusters")
        exit()
    if not os.path.isdir("InputData/Clusters/AdjacencyMatrices"):
        os.mkdir("InputData/Clusters/AdjacencyMatrices")
    if not os.listdir("InputData/Clusters/AdjacencyMatrices"): 
        warn.warn("You don't have any adjacency matrices - you won't be " + \
                  "able to run GroupingClusters(). Adjacency matrices " + \
                  "currently need to be added by hand")
    if not os.path.isdir("InputData/Clusters/ClusterGroups"):
        os.mkdir("InputData/Clusters/ClusterGroups")
    if not os.path.isdir("InputData/Clusters/Clustering"):
        warn.warn("You are missing the clustering data")
        exit()
        
    if not os.path.isdir("InputData/Other"):
        warn.warn("You are missing the input data on cloric demand, lats " + \
                  "and lons of the considered area, the mask specifying " + \
                  "cells to use, and the pearson distances between all cells.")
        exit()
        
    if not os.path.isdir("InputData/Prices"):
        warn.warn("You are missing the input data on the farm gate prices.")
        exit()
            
    if not os.path.isdir("InputData/YieldTrends"):
        warn.warn("You are missing the input data on yield trends.")
        exit()
        
    if not os.path.isdir("ModelOutput"):
        os.mkdir("ModelOutput")
    if not os.path.isdir("ModelOutput/SavedRuns"):
        os.mkdir("ModelOutput/SavedRuns")
    if not os.path.exists("ModelOutput/validation.txt"):
        with open("ModelOutput/validation.txt", "wb") as fp:
            pickle.dump({}, fp)
    if not os.path.isdir("ModelOutput/Pandas"):
        os.mkdir("ModelOutput/Pandas")
    if not os.path.exists("ModelOutput/Pandas/ColumnNames.txt"):
        SetUpPandaDicts()
    if not os.path.exists("ModelOutput/Pandas/current_panda.csv"):
        CreateEmptyPanda()
        
    if not os.path.isdir("PenaltiesAndIncome"):
            os.mkdir("PenaltiesAndIncome")
    
    if not os.path.exists("PenaltiesAndIncome/RhoFs.txt"):
        with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:
            pickle.dump({}, fp)
    if not os.path.exists("PenaltiesAndIncome/resProbFs.txt"):
        with open("PenaltiesAndIncome/resProbFs.txt", "wb") as fp:
            pickle.dump({}, fp)
    if not os.path.exists("PenaltiesAndIncome/resAvgImport.txt"):
        with open("PenaltiesAndIncome/resAvgImport.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/RhoSs.txt"):
        with open("PenaltiesAndIncome/RhoSs.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/resProbSs.txt"):
        with open("PenaltiesAndIncome/resProbSs.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/resAvgDebt.txt"):
        with open("PenaltiesAndIncome/resAvgDebt.txt", "wb") as fp:
            pickle.dump({}, fp) 
            
    if not os.path.exists("PenaltiesAndIncome/ExpectedIncomes.txt"):
        with open("PenaltiesAndIncome/ExpectedIncomes.txt", "wb") as fp:
            pickle.dump({}, fp) 
            
    return(None)
            
            
def CleanFolderStructure():
    """
    This function removes all folders originally generated by running 
    CheckFolderStructure and reruns CheckFolderStructure (thus removing all
    model results that werw collected so far).

    """
    print(colored("Warning, you are about to reset the folders " + \
           "Figures, ModelLogs, ModelOutput, and PenaltiesAndIncome. " + \
            "Do you wish to proceed?", "cyan"))
    
    proceed = input("Please enter yes/no: ")
    while len(proceed) == 0 or (proceed[0] != "y" and proceed[0] != "n"):
        proceed = input("Unknown input. Please enter yes or no: ")
        
    if proceed[0] == "y":
        if os.path.isdir("Figures"):
            shutil.rmtree("Figures")
            
        if os.path.isdir("ModelLogs"):
            shutil.rmtree("ModelLogs")
            
        if os.path.isdir("ModelOutput"):
            shutil.rmtree("ModelOutput")
            
        if os.path.isdir("PenaltiesAndIncome"):
            shutil.rmtree("PenaltiesAndIncome")
    
    CheckFolderStructure()
    return(None)