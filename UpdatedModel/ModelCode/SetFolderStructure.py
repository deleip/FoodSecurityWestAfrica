#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:44:33 2021

@author: debbora
"""

import os
import warnings as warn
import pickle

# %% ################ SETTING UP FOLDER STRUCTURE FOR RESULTS #################

def CheckFolderStructure():
    """
    This functions checks whether all folders that are needed are present and
    generates them if not. If dictionaries to save expected income, penalties 
    or validation values are not yet present empty dictionaries are put in 
    place. If input data is missing the function throws an error.
    """
    if not os.path.isdir("Figures"):
        os.mkdir("Figures")
    
    if not os.path.isdir("Figures/ClusterGroups"):
        os.mkdir("Figures/ClusterGroups")
    if not os.path.isdir("Figures/CropAllocs"):
        os.mkdir("Figures/CropAllocs")
    if not os.path.isdir("Figures/CompareCropAllocs"):
        os.mkdir("Figures/CompareCropAllocs")
    if not os.path.isdir("Figures/CompareCropAllocsRiskPooling"):
        os.mkdir("Figures/CompareCropAllocsRiskPooling")
    if not os.path.isdir("Figures/rhoSvsDebts"):
        os.mkdir("Figures/rhoSvsDebts")
        
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
        os.mkdir("ModelOutput/SavedRuns")
        with open("ModelOutput/validation.txt", "wb") as fp:
            pickle.dump({}, fp)
        
    if not os.path.isdir("OtherResults"):
        os.mkdir("OtherResults")    
        
    if not os.path.isdir("PenaltiesAndIncome"):
            os.mkdir("PenaltiesAndIncome")
    
    if not os.path.exists("PenaltiesAndIncome/ExpectedIncomes.txt"):
        with open("PenaltiesAndIncome/ExpectedIncomes.txt", "wb") as fp:
            pickle.dump({}, fp)
    if not os.path.exists("PenaltiesAndIncome/RhoFs.txt"):
        with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:
            pickle.dump({}, fp)
    if not os.path.exists("PenaltiesAndIncome/RhoSs.txt"):
        with open("PenaltiesAndIncome/RhoSs.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/Imports.txt"):
        with open("PenaltiesAndIncome/Imports.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/MaxProbFforAreaF.txt"):
        with open("PenaltiesAndIncome/MaxProbFforAreaF.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/MaxProbSforAreaS.txt"):
        with open("PenaltiesAndIncome/MaxProbSforAreaS.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/MaxProbFforAreaS.txt"):
        with open("PenaltiesAndIncome/MaxProbFforAreaS.txt", "wb") as fp:
            pickle.dump({}, fp) 
    if not os.path.exists("PenaltiesAndIncome/MaxProbSforAreaF.txt"):
        with open("PenaltiesAndIncome/MaxProbSforAreaF.txt", "wb") as fp:
            pickle.dump({}, fp)
    if not os.path.exists("PenaltiesAndIncome/MinimizedNecessaryDebt.txt"):
        with open("PenaltiesAndIncome/MinimizedNecessaryDebt.txt", "wb") as fp:
            pickle.dump({}, fp)
            