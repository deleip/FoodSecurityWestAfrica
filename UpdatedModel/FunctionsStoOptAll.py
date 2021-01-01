#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 23:31:27 2020

@author: Debbora Leip
"""

# %% #################### IMPORTING NECESSARY PACKAGES  #######################

import numpy as np
from scipy.special import comb
import scipy.stats as stats
import pickle
import pandas as pd
import time as tm 
import itertools as it
import warnings as warn
import gurobipy as gp
import sys
import matplotlib.pyplot as plt
import math
import os
# from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.colors as col
from termcolor import colored
import matplotlib.gridspec as gs

figsize = (24, 13.5)

# %% ############# DEFINING ERROR FOR INFEASIBLE PROBABILITIES ################

class PenaltyException(Exception):
    def __init__(self, message='Probability cannot be reached'):
        # Call the base class constructor with the parameters it needs
        super(PenaltyException, self).__init__(message)

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
            
# %% ############## WRAPPING FUNCTIONS FOR FOOD SECURITY MODEL ################

def FoodSecurityProblem(PenMet = "prob", probF = 0.99, probS = 0.95, \
                        rhoF = None, rhoS = None, prints = True, \
                        validation = None, save = True, plotTitle = None, \
                        **kwargs):
    """
        
    Setting up and solving the food security problem. Returns model output
    and additional information on the solution, as well as the VSS and a 
    validation of the model output. The results are also saved. If the model
    has already been solved for the exact settings, results are loaded instead
    of recalculated.
    

    Parameters
    ----------
    PenMet : "prob" or "penalties", optional
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly. The default is "prob".
    probF : float, optional
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.99.
    probS : float, optional
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float or None, optional 
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoF is None, a initial guess for rhoF will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for reaching 
        food demand. The default is None.
    rhoS : float or None, optional 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoS is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
        The default is None.
    prints : boolean, optional
        whether output should be written to the console while running function.
        The default is True.
    validation : None or int, optional
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.
    save : boolean, optional
        whether the results should be saved. The default is True.
    plotTitle : str or None
        If not None, a plot of the resulting crop allocations will be made 
        with that title and saved to Figures/CropAllocs.
    **kwargs
        settings for the model, passed to DefaultSettingsExcept()

    Returns
    -------
    crop_alloc : np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'S', 'exp_incomes', 'profits', 'exp_shortcomings', 
        'fd_penalty', 'avg_fd_penalty', 'sol_penalty', 'final_fund', 
        'prob_staying_solvent', 'prob_food_security', 'payouts', 
        'yearly_fixed_costs', 'num_years_with_losses')
    status : int
        status of solver (optimal: 2)
    durations : list
        time for setting up the model, time for solving, and total time (in sec.)
    settings : dict
        the model settings that were used     
    args : dict
        Dictionary of arguments needed as model input.
    rhoF : float
        The penalty for food shortcomings (depending on PenMet either the one 
        given or the one calculated to match the probability)    
    rhoS : float 
        The penalty for insolvency (depending on PenMet either the one 
        given or the one calculated to match the probability)         
    VSS_value : float
        VSS calculated as the difference between total costs using 
        deterministic solution for crop allocation and stochastic solution
        for crop allocation         
    crop_alloc_vss : np.array
        deterministic solution for optimal crop areas        
    meta_sol_vss : dict
        additional information on the deterministic solution        
    validation_values : dict
        total costs and penalties for the model result and a higher sample 
        size for validation ("sample_size", "total_costs", "total_costs_val", 
        "fd_penalty", "fd_penalty_val", "sol_penalty", "sol_penalty_val", 
        "total_penalties", "total_penalties_val", "deviation_penalties")
    probSnew : float
        The new probability for solvency (different from the input probS in 
        case that probability can't be reached).
    fn : str
        all settings combined to a single file name to save/load results
    """    
 
    settings = DefaultSettingsExcept(**kwargs)
    fn = filename(settings, PenMet, validation, probF, probS, rhoF, rhoS)
    if PenMet == "penalties":
        probS = None
        probF = None
    elif PenMet != "prob":
        sys.exit("A non-valid penalty method was chosen (PenMet must " + \
                 "be either \"prob\" or \"penalties\").")
    
    
    if not os.path.isfile("ModelOutput/SavedRuns/" + fn + ".txt"):
        try:
            crop_alloc, meta_sol, status, durations, settings, args, other, \
            rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
            validation_values = OptimizeModel(PenMet = PenMet,  
                                              probF = probF, 
                                              probS = probS, 
                                              rhoFini = rhoF,
                                              rhoSini = rhoS,
                                              prints = prints,
                                              validation = validation,
                                              save = save,
                                              **kwargs)
            if plotTitle is not None:
                PlotModelOutput(PlotType = "CropAlloc", title = plotTitle, \
                                file = fn, crop_alloc = crop_alloc, k = settings["k"], \
                                k_using = settings["k_using"], max_areas = args["max_areas"])
        except PenaltyException as e:
            with open("ModelOutput/SavedRuns/" + fn + ".txt", "wb"): pass
            raise PenaltyException(message =  e) 
    else:
        if os.path.getsize("ModelOutput/SavedRuns/" + fn + ".txt") == 0:
            raise PenaltyException(message =  "This setting has been tried but is not feasible")
            
        printing("Loading results", prints = prints)
        with open("ModelOutput/SavedRuns/" + fn + ".txt", "rb") as fp:
            pickle.load(fp) # info
            crop_alloc = pickle.load(fp)
            settings = pickle.load(fp)
            args = pickle.load(fp)
            other = pickle.load(fp)
            rhoF = pickle.load(fp)
            rhoS = pickle.load(fp)
            status = pickle.load(fp)
            durations = pickle.load(fp)
            VSS_value = pickle.load(fp)
            crop_alloc_vss = pickle.load(fp)
            validation_values = pickle.load(fp)
            if PenMet == "prob":
                probF = pickle.load(fp)
                probS = pickle.load(fp)
                
        meta_sol = GetMetaInformation(crop_alloc, args, rhoF, rhoS, probS)
        meta_sol_vss =  GetMetaInformation(crop_alloc_vss, args, rhoF, rhoS, probS)
        
        if plotTitle is not None:
            PlotModelOutput(PlotType = "CropAlloc", title = plotTitle, \
                            file = fn, crop_alloc = crop_alloc, k = settings["k"], \
                            k_using = settings["k_using"], max_areas = args["max_areas"])
    
    return(crop_alloc, meta_sol, status, durations, settings, args, other, \
           rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
           validation_values, fn)          

def OptimizeModel(PenMet = "prob", probF = 0.99, probS = 0.95, \
                                rhoFini = None, rhoSini = None, prints = True, \
                                validation = None, save = True, **kwargs):
    """
    Function combines setting up and solving the model, calculating additional
    information, and saving the results.

    Parameters
    ----------
    PenMet : "prob" or "penalties", optional
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly. The default is "prob".
    probF : float, optional
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.99.
    probS : float, optional
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float or None, optional
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoF is None, a initial guess for rhoF will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for reaching 
        food demand. The default is None.
    rhoS : float or None, optional
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoS is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
        The default is None.
    prints : boolean, optional
        whether output should be written to the console while running function.
        The default is True.
    validation : None or int, optional
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.
    save : boolean, optional
        whether the results should be saved. The default is True.
    **kwargs
        settings for the model, passed to DefaultSettingsExcept()

    Returns
    -------
    crop_alloc : np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'S', 'exp_incomes', 'profits', 'exp_shortcomings', 
        'fd_penalty', 'avg_fd_penalty', 'sol_penalty', 'final_fund', 
        'prob_staying_solvent', 'prob_food_security', 'payouts', 
        'yearly_fixed_costs', 'num_years_with_losses')
    status : int
        status of solver (optimal: 2)
    durations : lis
        time for setting up the model, time for solving, and total time (in sec.)
    settings : dict
        the model settings that were used    
    args : dict
        Dictionary of arguments needed as model input.      
    rhoF : float
        the penalty for food shortcomings        
    rhoS : float 
        the penalty for insolvency        
    VSS_value : float
        VSS calculated as the difference between total costs using 
        deterministic solution for crop allocation and stochastic solution
        for crop allocation         
    crop_alloc_vss : np.array
        deterministic solution for optimal crop areas        
    meta_sol_vss : dict
        additional information on the deterministic solution        
    validation_values : dict
        total costs and penalties for the model result and a higher sample 
        size for validation ("sample_size", "total_costs", "total_costs_val", 
        "fd_penalty", "fd_penalty_val", "sol_penalty", "sol_penalty_val", 
        "total_penalties", "total_penalties_val", "deviation_penalties")
    probSnew : float
        The new probability for solvency (different from the input probS in 
        case that probability can't be reached).

    """
    # timing
    all_start  = tm.time()
    
    # create dictionary of all settings (includes calculating or loading the
    # correct expected income)
    printing("Defining Settings\n", prints = prints)
    settings = DefaultSettingsExcept(**kwargs)
    
    # get parameters for the given settings
    printing("Getting parameters", prints = prints)
    args, other = SetParameters(settings)   
    expected_incomes = args["expected_incomes"]
    
    # get the right penalties
    if PenMet == "prob":
        rhoF, rhoS, necessary_debt, needed_import = GetPenalties(settings, args, other, probF, probS, \
                                  rhoFini, rhoSini, prints = prints)
        settings["import"] = needed_import
        if needed_import > 0:
            args["import"] = needed_import
        settings["neccessary_debt"] = necessary_debt
    
    # run the optimizer
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, rhoF, rhoS, probS, prints = prints)
        
    printing("\nResulting probabilities:\n" + \
    "     probF: " + str(np.round(meta_sol["prob_food_security"]*100, 2)) + "%\n" + \
    "     probS: " + str(np.round(meta_sol["prob_staying_solvent"]*100, 2)) + "%", prints)
        
    # VSS
    printing("\nCalculating VSS", prints = prints)
    crop_alloc_vss, meta_sol_vss = VSS(settings, args, rhoF, rhoS, probS)
    VSS_value = meta_sol_vss["exp_tot_costs"] - meta_sol["exp_tot_costs"]
    
    # out of sample validation
    validation_values = []
    if validation is not None:
        printing("\nOut of sample validation", prints = prints)
        meta_sol_val = OutOfSampleVal(crop_alloc, settings, rhoF, \
                              rhoS, expected_incomes, validation, probS, prints)
        validation_values = {"sample_size": validation,
               "total_costs": meta_sol["exp_tot_costs"], 
               "total_costs_val": meta_sol_val["exp_tot_costs"],
               "fd_penalty": np.nanmean(np.sum(meta_sol["fd_penalty"], axis = 1)),
               "fd_penalty_val": np.nanmean(np.sum(meta_sol_val["fd_penalty"], axis = 1)),
               "sol_penalty": np.nanmean(meta_sol["sol_penalty"]),
               "sol_penalty_val": np.nanmean(meta_sol_val["sol_penalty"])}
        validation_values["total_penalties"] = validation_values["fd_penalty"] + \
                        validation_values["sol_penalty"]
        validation_values["total_penalties_val"] = validation_values["fd_penalty_val"] + \
                        validation_values["sol_penalty_val"]
        validation_values["deviation_penalties"] = 1 - \
            (validation_values["total_penalties"]/validation_values["total_penalties_val"])  
        with open("ModelOutput/validation.txt", "rb") as fp:    
            val_dict = pickle.load(fp)
        val_name = str(len(settings["k_using"])) + "of" + \
                    str(settings["k"]) + "_N" + str(settings["N"]) + \
                    "_M" + str(validation)
        if val_name in val_dict.keys():
            tmp = val_dict[val_name]
            tmp.append(validation_values["deviation_penalties"])
            val_dict[val_name] = tmp  
        else:
            val_dict[val_name] = [validation_values["deviation_penalties"]]
        with open("ModelOutput/validation.txt", "wb") as fp:    
             pickle.dump(val_dict, fp)            
   
    # saving results
    if save:
        fn = filename(settings, PenMet, validation, probF, probS, rhoFini, rhoSini)
        
        info = ["crop_alloc", "settings", "args","meta_sol", "rhoF", "rhoS", \
                "status", "durations", "VSS_value", "crop_alloc_vss", \
                "meta_sol_vss", "validation_values"]
        # args.pop("ylds")
        # args.pop("cat_clusters")
        if PenMet == "prob":
            info.append("probF")
            info.append("probS")
        with open("ModelOutput/SavedRuns/" + fn + ".txt", "wb") as fp:
            pickle.dump(info, fp)
            pickle.dump(crop_alloc, fp)
            pickle.dump(settings, fp)
            pickle.dump(args, fp)
            pickle.dump(other, fp)
            pickle.dump(rhoF, fp)
            pickle.dump(rhoS, fp)
            pickle.dump(status, fp)
            pickle.dump(durations, fp)
            pickle.dump(VSS_value, fp)
            pickle.dump(crop_alloc_vss, fp)
            pickle.dump(validation_values, fp)
            if PenMet == "prob":
                pickle.dump(probF, fp)
                pickle.dump(probS, fp)
     
    # timing
    all_end  = tm.time()   
    full_time = all_end - all_start
    printing("\nTotal time: " + str(np.round(full_time, 2)) + "s", prints = prints)
            
    return(crop_alloc, meta_sol, status, durations, settings, args, other, \
           rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
           validation_values)          

# %% ############# FUNCTIONS TO GET INPUT FOR FOOD SECURITY MODEL #############

def DefaultSettingsExcept(k = 9,     
                          k_using = [3],
                          num_crops = 2,
                          yield_projection = "fixed",   
                          sim_start = 2017,
                          pop_scenario = "fixed",
                          risk = 0.05,                          
                          N = 3500, 
                          T = 25,
                          seed = 201120,
                          tax = 0.01,
                          perc_guaranteed = 0.9,
                          expected_incomes = None,
                          needed_import = 0,
                          ini_fund = 0):     
    """
    Using the default for all settings not specified, this creates a 
    dictionary of all settings.

    Parameters
    ----------
    k : int, optional
        Number of clusters in which the area is to be devided. 
        The default is 9.
    k_using : "all" or a list of int i\in{1,...,k}, optional
        Specifies which of the clusters are to be considered in the model. 
        The default is the representative cluster [3].
    num_crops : int, optional
        The number of crops that are used. The default is 2.
    yield_projection : "fixed" or "trend", optional
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
        The default is "fixed".
    sim_start : int, optional
        The first year of the simulation. The default is 2017.
    pop_scenario : str, optional
        Specifies which population scenario should be used. "fixed" uses the
        population of the year prior to the first year of the simulation for
        all years. The other options are 'Medium', 'High', 'Low', 
        'ConstantFertility', 'InstantReplacement', 'ZeroMigration', 
        'ConstantMortality', 'NoChange' and 'Momentum', referring to different
        UN_WPP population scenarios. All scenarios have the same estimates up 
        to (including) 2019, scenariospecific predictions start from 2020
        The default is "fixed".
    risk : int, optional
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic. The default is 5%.
    N : int, optional
        Number of yield samples to be used to approximate the expected value
        in the original objective function. The default is 3500.
    T : int, optional
        Number of years to cover in the simulation. The default is 25.
    seed : int, optional
        Seed to allow for reproduction of the results. The default is 201120.
    tax : float, optional
        Tax rate to be paied on farmers profits. The default is 1%.
    perc_guaranteed : float, optional
        The percentage that determines how high the guaranteed income will be 
        depending on the expected income of farmers in a scenario excluding
        the government. The default is 90%.
    expected_incomes : "None" or a np.array of shape (len(k_using),), optional
        If None, the expected income will be calculated by solving the model
        for the corresponding scenario without government. I an array is given,
        this will be assumed to have the correct values. The default is None.
    needed_import : float
        If PenMet = "prob", this will be caluclated such that the probability
        for food security can be reached. If PenMet = "penalties" this will 
        not be changed from the input value. The default is 0.
    ini_fund : float
        If PenMet = "prob", this will be caluclated such that the probability
        for solvency can be reached. If PenMet = "penalties" this will 
        not be changed from the input value. The default is 0.
        
        
    Returns
    -------
    settings : dict
        A dictionary that includes all of the above settings.
    

    """

    if type(k_using) is int:
        k_using = [k_using]
        
    if type(k_using) is tuple:
        k_using = list(k_using)
        
    k_using_tmp = k_using.copy()
    

    if k_using_tmp == "all":
        k_using_tmp = list(range(1, k + 1))
    
    # This will always be True except when the function is called from 
    # GetResultsToCompare() for multiple subsets of clusters 
    # (e.g. k_using = [(1,2),(3,6)])
    if sum([type(i) is int for i in k_using_tmp]) == len(k_using_tmp):
        k_using_tmp.sort()
            
    # create dictionary of settings
    settings =  {"k": k,
                 "k_using": k_using_tmp,
                 "num_crops": num_crops,
                 "yield_projection": yield_projection,
                 "sim_start": sim_start, 
                 "pop_scenario": pop_scenario,
                 "risk": risk,
                 "N": N,
                 "T": T,
                 "seed": seed, 
                 "tax": tax,
                 "perc_guaranteed": perc_guaranteed,
                 "expected_incomes": expected_incomes,
                 "import": needed_import,
                 "ini_fund": ini_fund}   
     

    # return dictionary of all settings
    return(settings)

def SetParameters(settings, wo_yields = False, VSS = False, prints = True):
    """
    
    Based on the settings, this sets all parameters needed as input to the
    model.    
    
    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    wo_yields : boolean, optional
        If True, the function will do everything execept generating the yield
        samples (and return an empty list as placeholder for the ylds 
        parameter). The default is False.
    VSS : boolean, optional
        If True, instead of yield samples only the average yields will be 
        returned and all clusters will be indicated as non-catastrophic by 
        cat_clusters, as needed to calculate the deterministic solution on 
        which the VSS is based. The default is False.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    args : dict
        Dictionary of arguments needed as model input. 
        
        - k: number of clusters in which the area is to be devided.
        - k_using: specifies which of the clusters are to be considered in 
          the model. 
        - num_crops: the number of crops that are used
        - N: number of yield samples to be used to approximate the 
          expected value in the original objective function
        - cat_cluster: np.array of size (N, T, len(k_using)), 
          indicating clusters with yields labeled as catastrophic with 1, 
          clusters with "normal" yields with 0
        - terminal years: np.array of size (N,) indicating the year in 
          which the simulation is terminated (i.e. the first year with a 
          catastrophic cluster) for each sample
        - ylds: np.array of size (N, T, num_crops, len(k_using)) of
          yield samples in 10^6t/10^6ha according to the presence of 
          catastrophes
        - costs: np array of size (num_crops,) giving cultivation costs 
          for each crop in 10^9$/10^6ha
        - demand: np.array of size (T,) giving the total food demand
          for each year in 10^12kcal
        - import: import needed to reach the desired probability for food
          security in 10^12kcal
        - ini_fund: initial fund size in 10^9$
        - tax: tax rate to be paied on farmers profits
        - prices: np.array of size (num_crops,) giving farm gate prices 
          farmers earn in 10^9$/10^6t
        - T: number of years covered in model
        - expected_incomes: expected income in secnario without government and
          probF = 95% for each clutser.               
        - guaranteed_income: np.array of size (T, len(k_using)) giving 
          the income guaranteed by the government for each year and cluster
          in case of catastrophe in 10^9$
        - crop_cal : np.array of size (num_crops,) giving the calorie content
          of the crops in 10^12kcal/10^6t
        - max_areas: np.array of size (len(k_using),) giving the upper 
          limit of area available for agricultural cultivation in each
          cluster
    other : dict
        Dictionary giving some additional inforamtion
        
        - slopes: slopes of the yield trends
        - constants: constants of the yield trends
        - yld_means: average yields 
        - residual_stds: standard deviations of the residuals of the yield 
          trends
        - prob_cat_year: probability for a catastrophic years given the 
          covered risk level
        - share_no_cat: share of samples that don't have a catastrophe within 
          the considered timeframe
        - y_profit: minimal yield per crop and cluster needed to not have 
          losses
        - share_rice_np: share of cases where rice yields are too low to 
          provide profit
        - share_maize_np: share of cases where maize yields are too low to 
          provide profit
        - exp_profit: expected profits in 10^9$/10^6ha per year, crop, cluster

    """
    
    crops = ["Rice", "Maize"]
    
# 0. extract settings from dictionary
    k = settings["k"]
    k_using = settings["k_using"]
    num_crops = settings["num_crops"]
    yield_projection = settings["yield_projection"]
    sim_start = settings["sim_start"]
    pop_scenario = settings["pop_scenario"]
    risk = settings["risk"]
    if VSS:
        N = 1
    else:
        N = settings["N"]
    T = settings["T"]
    seed = settings["seed"]
    tax = settings["tax"]
    perc_guaranteed = settings["perc_guaranteed"]
    expected_incomes = settings["expected_incomes"]
    
# 1. get cluster information (clusters given by k-Medoids on SPEI data) 
    with open("InputData/Clusters/Clustering/kMediods" + \
                        str(k) + "_PearsonDistSPEI_ProfitableArea.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
    
# 2. calculate total area and area proportions of the different clusters
    with open("InputData/Population/land_area.txt", "rb") as fp:    
        land_area = pickle.load(fp)
    cluster_areas = np.zeros(len(k_using))
    for i, cl in enumerate(k_using):
        cluster_areas[i] = np.nansum(land_area[clusters == cl])
    cluster_areas = cluster_areas * 100 # convert sq km to ha
                                        
# 3. Share of population in the area we use (in 2015):
    with open("InputData/Population/GPW_WA.txt", "rb") as fp:    
        gridded_pop = pickle.load(fp)[3,:,:]
    with open("InputData/Population/" + \
                  "UN_PopTotal_Prospects_WesternAfrica.txt", "rb") as fp:    
        total_pop = pickle.load(fp)
        scenarios = pickle.load(fp)
    total_pop_scen = total_pop[np.where(scenarios == "Medium")[0][0],:]
    total_pop_UN = total_pop_scen[2015-1950]
    cluster_pop = np.zeros(len(k_using))
    for i, cl in enumerate(k_using):
        cluster_pop[i] = np.nansum(gridded_pop[clusters == cl])
    total_pop_GPW = np.sum(cluster_pop)
    pop_ratio = total_pop_GPW/total_pop_UN
    
# 4. Country shares of area (for price calculation)
    with open("InputData/Prices/CountryCodesGridded.txt", "rb") as fp:    
        country_codes_gridded = pickle.load(fp)
    country_codes = pd.read_csv("InputData/Prices/CountryCodes.csv")
    with open("InputData/Other/MaskProfitableArea.txt", "rb") as fp:   
        yld_mask = pickle.load(fp)
    country_codes_gridded[yld_mask == 0] = np.nan
    total_land_cells = np.sum(yld_mask == 1)
    
    # liberia has no price data
    liberia_landcells = np.sum(country_codes_gridded == country_codes.loc \
                       [(country_codes.CountryName=="Liberia")]["Code"].values)
    total_cells_for_average = total_land_cells - liberia_landcells 
                    
    # Mauritania has no rice price data
    mauritania_landcells = np.sum(country_codes_gridded == country_codes.loc \
                    [(country_codes.CountryName=="Mauritania")]["Code"].values)
    total_cells_for_average_excl_mauritania = \
                            total_cells_for_average - mauritania_landcells 
                            
    country_codes["Shares"] = 0
    country_codes["Shares excl. Mauritania"] = 0
    
    country_codes_gridded[country_codes_gridded == country_codes.loc \
             [(country_codes.CountryName=="Liberia")]["Code"].values] = np.nan
    for idx, c in enumerate(country_codes["Code"].values):
        country_codes.iloc[idx, 2] = \
                    np.sum(country_codes_gridded == c)/total_cells_for_average
    
    country_codes_gridded[country_codes_gridded == country_codes.loc \
           [(country_codes.CountryName=="Mauritania")]["Code"].values] = np.nan
    for idx, c in enumerate(country_codes["Code"].values):
        country_codes.iloc[idx, 3] = np.sum(country_codes_gridded == c)/ \
                                        total_cells_for_average_excl_mauritania
    
    # removing countries with no share as they don't show up in prices df below
    country_codes = country_codes.drop(axis = 0, labels = [4, 5, 9]) 

# 5. Per person/day demand       
    # we use data from a paper on food waste 
    # (https://doi.org/10.1371/journal.pone.0228369) 
    # it includes country specific vlues for daily energy requirement per 
    # person (covering five of the countries in West Africa), based on data
    # from 2003. the calculation of the energy requirement depends on 
    # country specific values on age/gender of the population, body weight, 
    # and Physical Avticity Level.
    # Using the area shares as weights, this results in an average demand
    # per person and day of 2952.48kcal in the area we use.
    ppdemand = pd.read_csv("InputData/Other/CaloricDemand.csv")                           
    ppdemand["Shares"] = 0
    for c in ppdemand["Country"]:
        ppdemand.loc[ppdemand["Country"] == c, "Shares"] = \
            country_codes.loc[country_codes["CountryName"] == c, "Shares"].values
    ppdemand["Shares"] = ppdemand["Shares"] / np.sum(ppdemand["Shares"])        
    ppdemand = np.sum(ppdemand["Demand"] * ppdemand["Shares"])
    
# 6. cultivation costs of crops
    # RICE:
    # Liberia: "The cost of production of swampland Nerica rice (farming 
    # and processing of the paddy) is $308 per metric tons [...]. 
    # Swampland Nerica rice has a yield of 2.8 metric tons per hectare in 
    # Liberia."  
    # https://ekmsliberia.info/document/liberia-invest-agriculture/
    # => 862.4USD/ha (2015)
    # Nigeria: "COST AND RETURNS OF PADDY RICE PRODUCTION IN KADUNA STATE"
    # (Online ISSN: ISSN 2054-6327(online)) 
    # 1002USD/ha 
    # Benin: 
    # 105 FCFA/kg, 3t/ha =>  315000 FCFA/ha => 695.13USD/ha (2011)
    # Burkina-Faso: 
    # Rainfed: 50 FCFA/kg, 1t/ha => 50000 FCFA/ha => 104.62USD/ha (2011)
    # Lowland (bas-fonds): 55 FCFA/kg, 2t/ha => 110000 FCFA/ha 
    #                                               => 230.17 USD/ha
    # (I exclude the value for irrigated areas, as in West Africa 
    # agriculture is mainly rainfed)
    # Mali:
    # 108FCFA/kg, 2.7 t/ha => 291600 FCFA/ha => 589.27 USD/ha
    # Senegal: 
    # 101FCFA/kg, 5 t/ha => 505000 FCFA/ha => 1020.51 USD/ha
    # For Benin, Burkina-Faso, Mali, Senegal:
    # http://www.roppa-afrique.org/IMG/pdf/
    #                       rapport_final_synthese_regionale_riz_finale.pdf
    # in 2011 average exchange rate to USD 477.90 FCFA for 1 USD 
    # in 2014 average exchange rate to USD 494.85 FCFA for 1 USD
    # (https://www.exchangerates.org.uk/
    #                   USD-XOF-spot-exchange-rates-history-2011.html)
    # On average: 
    # (862.4+1002+695.13+104.62+230.17+589.27+1020.51)/7 = 643.44
    # MAIZE
    # "Competiveness of Maize Value Chains for Smallholders in West Africa"
    # DOI: 10.4236/as.2017.812099
    # Benin: 304.6 USD/ha (p. 1384, rainfed)
    # Ivory Coast: 305 USD/ha (p. 1384)
    # Ghana: 301.4 USD/ha (p. 1392) 
    # Nigeria: Field surey 2010 (e-ISSN: 2278-4861)
    # 32079.00 ₦/ha => 213.86 USD/ha
    # (https://www.exchangerates.org.uk/USD-NGN-spot-exchange-rates-
    # history-2010.html)
    # On average: (304.6 + 305 + 301.4 + 213.86)/4 = 281.22 
    costs = np.transpose(np.tile(np.array([643.44, 281.22]), \
                                                     (len(k_using), 1)))
    # in 10^9$/10^6ha
    costs = 1e-3 * costs 
        
# 7. Energy value of crops
    # https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/
    # beltsville-human-nutrition-research-center/methods-and-application-
    # of-food-composition-laboratory/mafcl-site-pages/sr11-sr28/
    # Rice: NDB_No 20450, "RICE,WHITE,MEDIUM-GRAIN,RAW,UNENR" [kcal/100g]
    kcal_rice = 360 * 10000             # [kcal/t]
    # Maize: NDB_No 20014, "CORN GRAIN,YEL" (kcal/100g)
    kcal_maize = 365 * 10000            # [kcal/t]
    crop_cal = np.array([kcal_rice, kcal_maize])
    # in 10^12kcal/10^6t
    crop_cal = 1e-6 * crop_cal
    
# 8. Food demand
    # based on the demand per person and day (ppdemand) and assuming no change
    # of per capita daily consumption we use UN population scenarios for West 
    # Africa and scale them down to the area we use, using the ratio from 2015 
    # (from gridded GPW data)
    with open("InputData/Population/" + \
                  "UN_PopTotal_Prospects_WesternAfrica.txt", "rb") as fp:    
        total_pop = pickle.load(fp)
        scenarios = pickle.load(fp)
    if pop_scenario == "fixed":
        total_pop_scen = total_pop[np.where(scenarios == "Medium")[0][0],:]
        demand = np.repeat(ppdemand*365*total_pop_scen[(sim_start-1)-1950],\
                   T) # for fixed we use pop of 2016 if run starts 2017
        demand = demand * pop_ratio
    else:
        total_pop_scen = total_pop[np.where(scenarios ==  \
                                                pop_scenario)[0][0],:]
        demand = ppdemand*365*total_pop_scen[(sim_start-1950): \
                                                (sim_start-1950+T)]
        demand = demand * pop_ratio
    # in 10^12 kcal
    demand = 1e-12 * demand
        
# 9. guaranteed income as share of expected income w/o government
    # if expected income is not given...
    if expected_incomes is None:    
        # get expected income for the given settings
        expected_incomes = GetExpectedIncome(settings, prints = prints)
        if np.sum(expected_incomes < 0) > 0:
            sys.exit("Negative expected income")
    guaranteed_income = np.repeat(expected_incomes[np.newaxis, :], T, axis=0)    
    guaranteed_income = perc_guaranteed * guaranteed_income                     
    # guaraneteed income per person assumed to be constant over time, 
    # therefore scale with population size
    if not pop_scenario == "fixed":
        total_pop_ratios = total_pop_scen / \
                                total_pop_scen[(sim_start-1)-1950]
        guaranteed_income = (guaranteed_income.swapaxes(0,1) * \
                            total_pop_ratios[(sim_start-1950): \
                                                (sim_start-1950+T)]) \
                            .swapaxes(0,1)
    printing("     Guaranteed income per cluster (in first year): " + str(np.round(guaranteed_income[0,:].flatten(), 3)), prints)
    
# 10. prices for selling crops, per crop and cluster
    with open("InputData//Prices/CountryAvgFarmGatePrices.txt", "rb") as fp:    
        country_avg_prices = pickle.load(fp)
    # Gambia is not included in our area
    country_avg_prices = country_avg_prices.drop(axis = 0, labels = [4])             
        
    price_maize = np.nansum(country_avg_prices["Maize"].values * \
                                            country_codes["Shares"].values)
    price_rice = np.nansum(country_avg_prices["Rice"].values * \
                          country_codes["Shares excl. Mauritania"].values)
    
    prices = np.transpose(np.tile(np.array([price_rice, \
                                            price_maize]), (len(k_using), 1)))
    # in 10^9$/10^6t
    prices = 1e-3 * prices 
    
# 11. thresholds for yields being profitable
    y_profit = costs/prices
        
# 12. Agricultural Areas: 
    # Landscapes of West Africa - A Window on a Changing World: 
    # "Between 1975 and 2013, the area covered by crops doubled in West 
    # Africa, reaching a total of 1,100,000 sq km, or 22.4 percent, of the 
    # land surface." 
    # (https://eros.usgs.gov/westafrica/agriculture-expansion)
    # => Approximate max. agricultural area by 22.4% of total cluster land 
    #   area (assuming agricultural area is evenly spread over West Africa)
    # Their area extends a bit more north, where agricultural land is 
    # probably reduced due to proximity to desert, so in our region the 
    # percentage of agricultural might be a bit higher in reality. But 
    # assuming evenly spread agricultural area over the area we use is 
    # already a big simplification, hence this will not be that critical.
    max_areas = cluster_areas * 0.224        
    # in 10^6ha
    max_areas = 1e-6 * max_areas

# 13. generating yield samples
    # using historic yield data from GDHY  
    with open("InputData/YieldTrends/DetrYieldAvg_k" + \
                              str(k) + "_ProfitableArea.txt", "rb") as fp:   
         pickle.load(fp) # yields_avg 
         pickle.load(fp) # avg_pred
         pickle.load(fp) # residuals
         pickle.load(fp) # residual_means
         residual_stds = pickle.load(fp)
         pickle.load(fp) # fstat
         constants = pickle.load(fp)
         slopes = pickle.load(fp)
         pickle.load(fp) # crops
    residual_stds = residual_stds[:, [i - 1 for i in k_using]]
    constants = constants[:, [i - 1 for i in k_using]] 
    slopes = slopes[:, [i - 1 for i in k_using]]

    # get yield realizations:
    # what is the probability of a catastrophic year for given settings?
    printing("\nOverview on yield samples", prints = prints)
    prob_cat_year = RiskForCatastrophe(risk, len(k_using))
    printing("     Prob for catastrophic year: " + str(np.round(prob_cat_year*100, 2)) + "%", prints = prints)    
    # create realizations of presence of catastrophic yields and corresponding
    # yield distributions
    np.random.seed(seed)
    cat_clusters, terminal_years, ylds, yld_means = \
          YieldRealisations(slopes, constants, residual_stds, sim_start, \
                           N, risk, T, len(k_using), num_crops, \
                           yield_projection, VSS, wo_yields)
    # probability to not have a catastrophe
    no_cat = np.sum(terminal_years == -1) / N
    printing("     Share of samples without catastrophe: " + str(np.round(no_cat*100, 2)), prints = prints) 
    # share of non-profitable crops
    share_rice_np = np.sum(ylds[:,:,0,:] < y_profit[0,:])/np.sum(~np.isnan(ylds[:,:,0,:]))
    printing("     Share of cases with rice yields too low to provide profit: " + \
             str(np.round(share_rice_np * 100, 2)), prints = prints)
    share_maize_np = np.sum(ylds[:,:,1,:] < y_profit[1,:])/np.sum(~np.isnan(ylds[:,:,1,:]))
    printing("     Share of cases with maize yields too low to provide profit: " + \
             str(np.round(share_maize_np * 100, 2)), prints = prints)
    # in average more profitable crop
    exp_profit = yld_means * prices - costs
    avg_time_profit = np.nanmean(exp_profit, axis = 0)
    more_profit = np.argmax(avg_time_profit)
    printing("     On average more profit: " + crops[more_profit], prints = prints)
    # in average more productive crop
    avg_time_production = np.nanmean(yld_means, axis = 0)
    more_food = np.argmax(avg_time_production)
    printing("     On average higher productivity: " + crops[more_food] + "\n", prints = prints)
    
# 14. group output into different dictionaries
    # arguments that are given to the objective function by the solver
    args = {"k": k,
            "k_using": k_using,
            "num_crops": num_crops,
            "N": N,
            "cat_clusters": cat_clusters,
            "terminal_years": terminal_years,
            "ylds": ylds,
            "costs": costs,
            "demand": demand,
            "import": settings["import"],
            "ini_fund": settings["ini_fund"],
            "tax": tax,
            "prices": prices,
            "T": T,
            "expected_incomes": expected_incomes,
            "guaranteed_income": guaranteed_income,
            "crop_cal": crop_cal,
            "max_areas": max_areas}
        
    # information not needed by the solver but potentially interesting 
    other = {"slopes": slopes,
             "constants": constants,
             "yld_means": yld_means,
             "residual_stds": residual_stds,
             "prob_cat_year": prob_cat_year,
             "share_no_cat": no_cat,
             "y_profit": y_profit,
             "share_rice_np": share_rice_np,
             "share_maize_np": share_maize_np,
             "exp_profit": exp_profit}
        
    return(args, other)

def RiskForCatastrophe(risk, num_clusters):
    """
    The risk level covered by the government is an input parameter to the 
    model. The resulting probability for a catastrophic year depends on the
    number of clusters used by the model.

    Parameters
    ----------
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    num_clusters : int
        The number of crops that are used.

    Returns
    -------
    prob : The probability for a year to be catastrophic.

    """
    res = 0
    for i in range(1, num_clusters + 1):
        res += comb(num_clusters, i) * (risk**i) * (1-risk)**(num_clusters-i)
             
    return(res)

def YieldRealisations(yld_slopes, yld_constants, resid_std, sim_start, N, \
                      risk, T, num_clusters, num_crops, \
                      yield_projection, VSS = False, wo_yields = False):  
    """
    Depending on the risk level the occurence of catastrophes is generated,
    which is then used to generate yield samples from the given yield 
    distributions.
    
    Parameters
    ----------
    yld_slopes : np.array of size (num_crops, len(k_using))
        Slopes of the yield trends.
    yld_constants : np.array of size (num_crops, len(k_using))
        Constants of the yield trends.
    resid_std : np.array of size (num_crops, len(k_using))
        Standard deviations of the residuals of the yield trends.
    sim_start : int
        The first year of the simulation..
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function.
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    T : int
        Number of years to cover in the simulation.
    num_clusters : int
        Number of clusters considered by model (corresponding to len(k_using))
    num_crops : int
        The number of crops that are used.
    yield_projection : "fixed" or "trend"
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
    VSS : boolean, optional
        If True, all clusters will be indicated as non-catastrophic, as needed 
        to calculate the deterministic solution on which the VSS is based. 
        The default is False.
    wo_yields : boolean, optional
        If True, the function will do everything execept generating the yield
        samples (and return an empty list as placeholder for the ylds 
        parameter). The default is False.
        
    Returns
    -------
    cat_cluters : np.array of size (N, T, len(k_using)) 
        indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0
    terminal_years : np.array of size (N,) 
        indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample, with 0 for 
        catastrophe in first year, 1 for second year, etc. If no catastrophe 
        happens before T, this is indicated by -1
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        yield samples in 10^6t/10^6ha according to the presence of 
        catastrophes
    yld_means : np.array of size (T, num_crops, len(k_using))
        Average yields in 10^6t/10^6ha.

    """
    
    # generating catastrophes 
    cat_clusters, terminal_years = CatastrophicYears(risk, \
                                    N, T, num_clusters, VSS)
    
    # generating yields according to catastrophes
    if wo_yields:
        ylds = []
    else:
        ylds, yld_means = ProjectYields(yld_slopes, yld_constants, resid_std, sim_start, \
                             N, cat_clusters, terminal_years, T, risk, \
                             num_clusters, num_crops, yield_projection, \
                             VSS)
        
    # comparing with np.nans leads to annoying warnings, so we turn them off        
    np.seterr(invalid='ignore')
    ylds[ylds<0] = 0
    np.seterr(invalid='warn')
    
    return(cat_clusters, terminal_years, ylds, yld_means)

def CatastrophicYears(risk, N, T, num_clusters, VSS):
    """
    Given the risk level that is to be covered, this creates a np.array 
    indicating which clusters should be catastrophic for each year.

    Parameters
    ----------
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function. The default is 3500.
    T : int
        Number of years to cover in the simulation. The default is 25.
    k : int
        Number of clusters in which the area is to be devided. 
        The default is 1.
    VSS : boolean
        If True, all clusters will be indicated as non-catastrophic, and 
        average yields will be returned instead of yield samples, as needed 
        to calculate the deterministic solution on which the VSS is based. 

    Returns
    -------
    cat_cluters : np.array of size (N, T, len(k_using)) 
        indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0
    terminal_years : np.array of size (N,) 
        indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample

    """

    if VSS:
        cat_clusters = np.zeros((N, T, num_clusters))
        terminal_years = np.repeat(-1, N) 
        return(cat_clusters, terminal_years)
    
    # generating uniform random variables between 0 and 1 to define years
    # with catastrophic cluster (catastrohic if over threshold)
    threshold = 1 - risk
    cat_clusters = np.random.uniform(0, 1, [N, T, num_clusters])
    cat_clusters[cat_clusters > threshold] = 1
    cat_clusters[cat_clusters <= threshold] = 0
    
    # year is catastrophic if min. one cluster is catastrophic
    cat_years = np.sum(cat_clusters, axis = 2)
    cat_years[cat_years < 1] = 0
    cat_years[cat_years >= 1] = 1
    
    # terminal year is first catastrophic year. If no catastrophic year before
    # T, we set terminal year to -1
    terminal_years = np.sum(cat_years, axis = 1)
    for i in range(0, N):
        if terminal_years[i] == 0:
            terminal_years[i] = -1
        else:
            terminal_years[i] = np.min(np.where(cat_years[i, :] == 1))
    return(cat_clusters, terminal_years)

def ProjectYields(yld_slopes, yld_constants, resid_std, sim_start, \
                  N, cat_clusters, terminal_years, T, risk, \
                  num_clusters, num_crops, yield_projection, \
                  VSS = False):
    """
    Depending on the occurence of catastrophes yield samples are generated
    from the given yield distributions.    

    Parameters
    ----------
    yld_slopes : np.array of size (num_crops, len(k_using))
        Slopes of the yield trends.
    yld_constants : np.array of size (num_crops, len(k_using))
        Constants of the yield trends.
    resid_std : np.array of size (num_crops, len(k_using))
        Standard deviations of the residuals of the yield trends.
    sim_start : int
        The first year of the simulation..
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function.
    cat_cluters : np.array of size (N, T, len(k_using)) 
        indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0
    terminal_years : np.array of size (N,) 
        indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample, with 0 for 
        catastrophe in first year, 1 for second year, etc. If no catastrophe 
        happens before T, this is indicated by -1
    T : int
        Number of years to cover in the simulation.
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    num_clusters : int
        Number of clusters considered by model (corresponding to len(k_using))
    num_crops : int
        The number of crops that are used.
    yield_projection : "fixed" or "trend"
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
    VSS : boolean, optional
        If True, average yields will be returned instead of yield samples, as 
        needed to calculate the deterministic solution on which the VSS is
        based. The default is False.

    Returns
    -------
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        yield samples in 10^6t/10^6ha according to the presence of 
        catastrophes
    yld_means : np.array of size (T, num_crops, len(k_using))
        Average yields in 10^6t/10^6ha.

    """    
    # project means of yield distributions (either repeating fixed year or by
    # using trend)
    if yield_projection == "fixed": # for fixed we use the year before start
        year_rel = (sim_start - 1) - 1981
        yld_means = yld_constants + year_rel * yld_slopes
        yld_means = np.repeat(yld_means[np.newaxis, :, :], T, axis = 0)
            
    elif yield_projection == "trend":
        year_rel = sim_start - 1981
        years = np.transpose(np.tile(np.array(range(year_rel, year_rel + \
                                        T)), (num_clusters, num_crops, 1)))
        yld_means = yld_constants + years * yld_slopes
    resid_std = np.repeat(resid_std[np.newaxis,:, :], T, axis=0)
    
    # needed for VSS
    if VSS == True:
        return(np.expand_dims(yld_means, axis = 0), yld_means)
    
    # initializing yield array
    ylds = np.empty([N, T, num_crops, num_clusters])
    ylds.fill(np.nan)
    
    # calculating quantile of standard normal distribution corresponding to 
    # the catastrophic yield quantile for truncnorm fct
    quantile_low = stats.norm.ppf(risk, 0, 1)
    
    # generating yields: for catastrophic clusters from lower quantile, for
    # normal years from upper quantile. Dependence between crops, i.e. if 
    # cluster is catastrophic, both crops get yields from lower quantile of 
    # their distributions
    for run in range(0, N):
        if int(terminal_years[run]) == -1:
            ylds[run, :, :, :] = stats.truncnorm.rvs(quantile_low, np.inf, \
                                                     yld_means, resid_std)
        else:
            term_year = int(terminal_years[run])
            ylds[run, :(term_year+1), :, :] = \
                stats.truncnorm.rvs(quantile_low, np.inf, \
                                    yld_means[:(term_year+1), :, :], \
                                    resid_std[:(term_year+1), :, :])
            for cl in range(0, num_clusters):
                if cat_clusters[run, term_year, cl] == 1:
                    ylds[run, term_year, :, cl] = \
                        stats.truncnorm.rvs(- np.inf, quantile_low, \
                                            yld_means[term_year, :, cl], \
                                            resid_std[term_year, :, cl])
    
    return(ylds, yld_means)  

# %% ############### FUNCTIONS RUNNING MODEL TO GET EXP INCOME ################

def GetExpectedIncome(settings, prints = True):
    """
    Either loading expected income if it was already calculated for these 
    settings, or calling the function to calculate the expected income. 

    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

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
        printing("\nFetching expected income", prints = prints)
        expected_incomes = dict_incomes[SettingsAffectingGuaranteedIncome]
    # else calculate (and save) it
    else:
        expected_incomes = CalcExpectedIncome(settings, \
                                 SettingsAffectingGuaranteedIncome)
        dict_incomes[SettingsAffectingGuaranteedIncome] = expected_incomes
        with open("PenaltiesAndIncome/ExpectedIncomes.txt", "wb") as fp:    
             pickle.dump(dict_incomes, fp)
            
    return(expected_incomes)
       
def CalcExpectedIncome(settings, SettingsAffectingGuaranteedIncome,
                       prints = True):
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
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    expected_incomes :  np.array of size (T, len(k_using))
        The expected income of farmers in a scenario where the government is
        not involved.

    """
    printing("\nCalculating expected income ", prints = prints)
    settings_ExpIn = settings.copy()

    # change some settings: we are interested in the expected income in 2016
    settings_ExpIn["seed"] = 201120
    settings_ExpIn["yield_projection"] = "fixed"
    settings_ExpIn["pop_scenario"] = "fixed"
    settings_ExpIn["T"] = 1
    settings_ExpIn["expected_incomes"] = np.zeros(len(settings["k_using"]))
    probF = 0.99
    
    # settings affecting the food demand penalty
    SettingsBasics = "k" + str(settings_ExpIn["k"]) + \
            "using" +  '_'.join(str(n) for n in settings_ExpIn["k_using"]) + \
            "num_crops" + str(settings_ExpIn["num_crops"]) + \
            "yield_projection" + str(settings_ExpIn["yield_projection"]) + \
            "sim_start" + str(settings_ExpIn["sim_start"]) + \
            "pop_scenario" + str(settings_ExpIn["pop_scenario"]) + \
            "T" + str(settings_ExpIn["T"])
    SettingsMaxProbS = SettingsBasics + \
            "risk" + str(settings["risk"]) + \
            "tax" + str(settings["tax"]) + \
            "perc_guaranteed" + str(settings["perc_guaranteed"]) + \
            "N" + str(settings["N"])
    SettingsMaxProbF = SettingsBasics + "N" + str(settings_ExpIn["N"])
    SettingsFirstGuess =  SettingsBasics + "probF" + str(probF)
    SettingsAffectingRhoF = SettingsFirstGuess + "N" + str(settings_ExpIn["N"])
    
    # first guess
    with open("PenaltiesAndIncome/RhoFs.txt", "rb") as fp:    
        dict_rhoFs = pickle.load(fp)
    with open("PenaltiesAndIncome/Imports.txt", "rb") as fp:    
        dict_imports = pickle.load(fp)
    with open("PenaltiesAndIncome/MaxProbFforAreaF.txt", "rb") as fp:    
        dict_maxProbF = pickle.load(fp)
    with open("PenaltiesAndIncome/MaxProbSforAreaF.txt", "rb") as fp:    
        dict_maxProbS = pickle.load(fp)
    rhoFini = GetInitialGuess(dict_rhoFs, SettingsFirstGuess)
    
    # we assume that without government farmers aim for 95% probability of 
    # food security, therefore we find the right penalty for probF = 95%.
    # As we want the income in a scenario without government, the final run of
    # GetRhoF (with rohS = 0) automatically is the right run
    args, other = SetParameters(settings_ExpIn, prints = False)
    try:
        rhoF, maxProbF, max_probS, needed_import, crop_alloc, meta_sol = \
               GetRhoF(args, other, probF = probF, rhoFini = rhoFini, prints = False) 
    except PenaltyException as e:
        raise PenaltyException(message =  str(e) + " (called from CalcExpectedIncome)")
          
    dict_rhoFs[SettingsAffectingRhoF] = rhoF
    dict_imports[SettingsAffectingRhoF] = needed_import
    dict_maxProbF[SettingsMaxProbF] = maxProbF
    dict_maxProbS[SettingsMaxProbS] = max_probS
    
    # saving updated dicts
    with open("PenaltiesAndIncome/RhoFs.txt", "wb") as fp:    
         pickle.dump(dict_rhoFs, fp)
    with open("PenaltiesAndIncome/Imports.txt", "wb") as fp:    
         pickle.dump(dict_imports, fp)
    with open("PenaltiesAndIncome/MaxProbFforAreaF.txt", "wb") as fp:    
         pickle.dump(dict_maxProbF, fp)
    with open("PenaltiesAndIncome/MaxProbSforAreaF.txt", "wb") as fp:    
         pickle.dump(dict_maxProbS, fp)
        
    return(meta_sol["exp_incomes"].flatten())

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
    
    bestN = 0
    bestFile = None
    rho = None
    for file in dictGuesses.keys():
        if file.startswith(name + "N"):
            N = int(file[len(name)+1:])
            if N > bestN:
                bestN = N
                bestFile = file
    if bestFile is not None: 
        rho = dictGuesses[bestFile]
    return(rho)

# %% ############### FUNCTIONS RUNNING MODEL TO GET PENALTIES #################

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
        The default is 3.
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
    
    
    status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, 1e9, probS, prints = False)   
    
    
    # run obective function for this area and the given settings
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

def CheckPotential(args, other, probF = None, probS = None, accuracy = 3, prints = True):
    """
    Wrapper function for finding potential of area either for food security 
    or for solvency. Only one probF and probS should differ from None, thus
    deciding which potential we want to analyze.
    
    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    other : dict
        Other information on the model setup (on the yield distributions).
    probF : float or None, optional
        The desired probability for food security. The default is None.
    probS : float or None, optional
        The desired probability for solvency. The default is None.
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probF. 
        The default is 3.
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
        return(CheckOptimalProbF(args, other, probF, accuracy, prints))
    elif probS is not None and probF is None:
        return(CheckOptimalProbS(args, other, probS, accuracy, prints))

def GetRhoF(args, other, probF, rhoFini, shareDiff = 10, accuracy = 4, prints = True):
    """
    Finding the correct rhoF given the probability probF, based on a bisection
    search algorithm.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input.  
    other : dict
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
    accuracy : int, optional
        Desired decimal places of accuracy of the obtained probF. 
        The default is 4.
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
    maxProbF, maxProbS, needed_import = CheckPotential(args_tmp, other, probF = probF, prints = prints)
    
    if needed_import > 0:
        args_tmp["import"] = needed_import
        
    # accuracy information
    printing("     accuracy we demand for probF: " + str(accuracy - 2) + " decimal places", prints = prints)
    printing("     accuracy we demand for rhoF: 1/" + str(shareDiff) + " of final rhoF\n", prints = prints)
    
    # check if rhoF from run with smaller N works here as well:
    # if we get the right probF for our guess, and a lower probF for rhoFcheck 
    # at the lower end of our accuracy-interval, we know that the correct 
    # rhoF is in that interval and can return our guess
    if rhoFini is not None:
        printing("     Checking guess from run with lower N", prints = prints)
        status, crop_alloc, meta_sol, prob, durations = \
                        SolveReducedcLinearProblemGurobiPy(args_tmp, rhoFini, 0, prints = False) 
        ReportProgressFindingRho(rhoFini, meta_sol, accuracy, durations, \
                                 ProbType = "F", prefix = "Guess: ", prints = prints) 
        if np.round(meta_sol["prob_food_security"], accuracy) == probF:
            rhoFcheck = rhoFini - rhoFini/shareDiff
            status, crop_alloc_check, meta_sol_check, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args_tmp, rhoFcheck, 0, prints = False)  
            ReportProgressFindingRho(rhoFcheck, meta_sol_check, accuracy, durations, \
                                     ProbType = "F", prefix = "Check: ", prints = prints) 
            if np.round(meta_sol_check["prob_food_security"], accuracy) < probF:
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
    
    if np.round(meta_sol["prob_food_security"], accuracy) == probF:
        lowestCorrect = rhoFini
                
    # remember guess
    rhoFold = rhoFini
    
    # report
    accuracy_int = lowestCorrect - rhoFLastUp
    ReportProgressFindingRho(rhoFold, meta_sol, accuracy, durations, \
                             accuracy_int, ProbType = "F", prints = prints)
        
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
                                 accuracy_int, ProbType = "F", prints = prints)
            
        # remember guess
        rhoFold = rhoFnew
        if np.round(meta_sol["prob_food_security"], accuracy) == probF \
            and lowestCorrect > rhoFnew:
            lowestCorrect = rhoFnew
            
    
    # last report
    ReportProgressFindingRho(rhoFnew, meta_sol, accuracy, durations, \
                             accuracy_int, ProbType = "F", prints = prints)    
        
    printing("\n     Final rhoF: " + str(rhoF), prints = prints)
    
    return(rhoF, maxProbF, maxProbS, needed_import, crop_alloc, meta_sol)

def GetRhoS_Wrapper(args, other, probS, rhoSini, file, shareDiff = 10, accuracy = 3, prints = True):
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
        Desired decimal places of accuracy of the obtained probF. 
        The default is 3.
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
    
    if maxProbS >= probS:
        printing("     Finding corresponding penalty\n", prints)
        rhoS = GetRhoS(args, probS, rhoSini, shareDiff, accuracy, prints)
        necessary_debt = 0
    else:
        printing("     Finding lowest penalty minimizing necessary debt\n", prints)
        rhoS, necessary_debt = MinimizeNecessaryDebt(args, probS, rhoSini, necessary_debt,  shareDiff, accuracy, file, prints)
    
    printing("\n     Final rhoS: " + str(rhoS))
    
    return(rhoS, necessary_debt, maxProbS, maxProbF)

def UpdateRhoDebtOutside(necessary_debt, debt_top, debt_bottom, rhoSold, \
                  UpperBorder, LowerBorder, rhoSdip, debtsDip, shareDiff):
    if len(rhoSdip) == 0:
        if necessary_debt == debt_bottom:
            if UpperBorder is None:
                rhoSnew = rhoSold * 2
                LowerBorder = rhoSold
                diff = False
            else:
                rhoSnew = rhoSold + (UpperBorder - rhoSold)/2
                LowerBorder = rhoSold
                diff = (UpperBorder - rhoSold)/2
                if diff > rhoSnew/shareDiff:
                    diff = False
        # attention, we do not take care of the case of hitting exactly this value 
        # in the "dip" of the function => das koennen wir loesen indem wir mit 
        # gerundeten debt werten arbeiten - ah ne doch nicht, weil dann muesste 
        # ich die rand debts auch runden, und das macht es noch wahrscheinlicher
        # sie zu treffen...
        elif necessary_debt == debt_top:
            rhoSnew = rhoSold - (rhoSold - LowerBorder)/2
            UpperBorder = rhoSold
            diff = (rhoSold - LowerBorder)/2
            if diff > rhoSnew/shareDiff:
                diff = False
    else:
        if necessary_debt == debt_bottom:
            rhoSnew = rhoSold + (rhoSdip[0] - rhoSold)/2
            LowerBorder = rhoSold
            diff = (rhoSdip[0] - rhoSold)/2
            if diff > rhoSnew/shareDiff:
                diff = False
        elif necessary_debt == debt_top:
            rhoSnew = rhoSold - (rhoSold - rhoSdip[-1])/2
            UpperBorder = rhoSold
            diff = (rhoSold - rhoSdip[-1])/2
            if diff > rhoSnew/shareDiff:
                diff = False
                
    if necessary_debt != debt_bottom and necessary_debt != debt_top:
        rhoSdip.append(rhoSold)
        debtsDip.append(necessary_debt)
        s = sorted(zip(rhoSdip,debtsDip))
        rhoSdip = [x for x,_ in s]
        debtsDip = [x for _,x in s]
        if debtsDip[-1] == min(debtsDip):
            rhoSnew = rhoSdip[-1] + (UpperBorder - rhoSdip[-1])/2
            diff = (UpperBorder - rhoSdip[-1])/2
            if diff > rhoSnew/shareDiff:
                diff = False
        elif debtsDip[0] == min(debtsDip):
            rhoSnew = rhoSdip[0] - (rhoSdip[0] - LowerBorder)/2
            diff = (rhoSdip[0] - LowerBorder)/2
            if diff > rhoSnew/shareDiff:
                diff = False
        else:
            rhoSnew = "Found dip!"
            diff = False
            
    # wenn am ende immer noch min(debtsDip) = debtsDip[0], dann rhoS = 0
    # wenn am ende immer noch min(debtsDip) = debtsDip[-1], dann rhoS = rhoSdip[-1] ??
    
    
    return(rhoSnew, UpperBorder, LowerBorder, rhoSdip, debtsDip, diff)

def UpdateRhoDebtDip(rhoSdip, debtsDip, rhoSold1 = None, necessary_debt1 = None, \
                     rhoSold2 = None, necessary_debt2 = None):
    if rhoSold1 is not None:
        rhoSdip.append(rhoSold1)
        debtsDip.append(necessary_debt1)
    if rhoSold2 is not None:
        rhoSdip.append(rhoSold2)
        debtsDip.append(necessary_debt2)
    s = sorted(zip(rhoSdip,debtsDip))
    rhoSdip = [x for x,_ in s]
    debtsDip = [x for _,x in s]
    
    i = debtsDip.index(min(debtsDip))
    rhoSnew1 = rhoSdip[i] + (rhoSdip[i+1] - rhoSdip[i])/2
    rhoSnew2 = rhoSdip[i] - (rhoSdip[i] - rhoSdip[i-1])/2
    
    return(rhoSnew1, rhoSnew2, rhoSdip[i], debtsDip[i], rhoSdip, debtsDip)
    

def MinimizeNecessaryDebt(args, probS, rhoSini, debt_top, shareDiff, accuracy, file, prints):
    # minimizing the absolute value of debt neccessary to reach probS, as we 
    # also don't want to be too good (as probabilities higher than probS will
    # lead to negative debt)
    
    # accuracy information
    printing("     accuracy we demand for rhoS: 1/" + str(shareDiff) + " of final rhoS\n", prints = prints)
    
    # checking for rhoS = 0
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, 0, 0, probS, prints = False) 
    debt_bottom = meta_sol["necessary_debt"]
    
    # check if rhoS from run with smaller N works here as well
    if rhoSini is not None:
        # TODO I think this case doesn't happen - if it does I could still
        # optimize this to use the guess to improve computational time for 
        # rhoSini == 0
        if rhoSini != 0:
            printing("     Checking guess from run with lower N", prints = prints)
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, probS, prints = False) 
            necessary_debt = meta_sol["necessary_debt"]
            ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                     ProbType = "S", debt = necessary_debt, prefix = "Guess: ", prints = prints)
                
            if necessary_debt == debt_top:
                rhoScheck = rhoSini - rhoSini/shareDiff
                status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck, probS, prints = False) 
                necessary_debt_check = meta_sol["necessary_debt"]
                ReportProgressFindingRho(rhoScheck, meta_sol, accuracy, durations, \
                                         ProbType = "S", debt = necessary_debt_check, prefix = "Check: ", prints = prints)
                if necessary_debt_check > necessary_debt:
                    return(rhoSini, necessary_debt)
            elif necessary_debt != debt_bottom:          
                rhoScheck1 = rhoSini - rhoSini/shareDiff
                rhoScheck2 = rhoSini + rhoSini/shareDiff
                status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck1, probS, prints = False) 
                necessary_debt_check1 = meta_sol["necessary_debt"]
                ReportProgressFindingRho(rhoScheck1, meta_sol, accuracy, durations, \
                                         ProbType = "S", debt = necessary_debt_check1, prefix = "Check 1: ", prints = prints)
                status, crop_alloc, meta_sol, prob, durations = \
                    SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck2, probS, prints = False) 
                necessary_debt_check2 = meta_sol["necessary_debt"]
                ReportProgressFindingRho(rhoScheck2, meta_sol, accuracy, durations, \
                                         ProbType = "S", debt = necessary_debt_check2, prefix = "Check 2: ", prints = prints)
                if necessary_debt_check1 > necessary_debt and \
                    necessary_debt_check2 > necessary_debt:
                    return(rhoSini, necessary_debt)
        printing("     Oops, that guess didn't work - starting from scratch\n", prints = prints)
    
    # initializing values for search algorithm
    LowerBorder = 0
    UpperBorder = None
    rhoSdip = []
    debtsDip = []
    diff = False
    
    # initialize figure showing rhoS vs. necessary debt to reach probS
    fig = plt.figure(figsize = figsize)  
    
    # plot and report
    plt.scatter(0, debt_bottom, s = 10, color = "blue")
    ReportProgressFindingRho(0, meta_sol, accuracy, durations, \
                             ProbType = "S", debt = debt_bottom, prints = prints)
    
    # checking for high rhoS
    rhoSnew = 100
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, probS, prints = False) 
    necessary_debt = meta_sol["necessary_debt"]
    
    # remember guess
    rhoSold = rhoSnew
 
    # plot and report
    plt.scatter(rhoSold, necessary_debt, s = 10)
    ReportProgressFindingRho(rhoSold, meta_sol, accuracy, durations, \
                             ProbType = "S", debt = necessary_debt, prints = prints)
        
    while not diff:
        #get next guess
        rhoSnew, UpperBorder, LowerBorder, rhoSdip, debtsDip, diff = \
            UpdateRhoDebtOutside(necessary_debt, debt_top, debt_bottom, rhoSold, \
                  UpperBorder, LowerBorder, rhoSdip, debtsDip, shareDiff)
        
        if rhoSnew == "Found dip!":
            break
        
        status, crop_alloc, meta_sol, prob, durations = \
            SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew, probS, prints = False) 
        necessary_debt = meta_sol["necessary_debt"]
        
        # report
        ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                                 ProbType = "S", debt = necessary_debt, prints = prints)
        
        # remember guess
        rhoSold = rhoSnew
        
        # plot
        plt.scatter(rhoSnew, necessary_debt, s = 10, color = "blue")
    
        
    if rhoSnew == "Found dip!":
        rhoSnew1, rhoSnew2, rhoSmin, debtMin, rhoSdip, debtsDip = UpdateRhoDebtDip(rhoSdip, debtsDip)
        
        while not diff:
            # calculating results for first point
            status, crop_alloc, meta_sol1, prob, durations1 = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew1, probS, prints = False) 
            necessary_debt1 = meta_sol1["necessary_debt"]
        
            # plot
            plt.scatter(rhoSnew1, necessary_debt1, s = 10, color = "blue")
            
            # report
            ReportProgressFindingRho(rhoSnew1, meta_sol1, accuracy, durations1, \
                                     ProbType = "S", debt = necessary_debt1, prefix = "1. ", prints = prints)
                
            # if already this debt is the new minimum, the other can't be
            # smaller (as there can only be one global minimum)
            if necessary_debt1 < debtMin:
                diff = rhoSnew1 - rhoSmin
                if diff > rhoSnew1/shareDiff:
                    diff = False
                # get new guesses
                rhoSnew1, rhoSnew2, rhoSmin, debtMin, rhoSdip, debtsDip = \
                    UpdateRhoDebtDip(rhoSdip, debtsDip, rhoSnew1, \
                                     necessary_debt1)
                continue
                
            # calculating results for second point
            status, crop_alloc, meta_sol2, prob, durations2 = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoSnew2, probS, prints = False) 
            necessary_debt2 = meta_sol2["necessary_debt"] 
        
            # plot
            plt.scatter(rhoSnew2, necessary_debt2, s = 10, color = "blue")
            
            # report
            ReportProgressFindingRho(rhoSnew2, meta_sol2, accuracy, durations2, \
                                     ProbType = "S", debt = necessary_debt2, prefix = "2. ", prints = prints)
                
            # check if we are accurate enough (we assume that we never get the
            # exact same debt twice as long as we are in the "dip" of the curve=)
            if necessary_debt2 > debtMin:
                diff = rhoSnew1 - rhoSnew2
                if diff > rhoSmin/shareDiff:
                    diff = False
            elif necessary_debt2 < debtMin:
                diff = rhoSmin - rhoSnew2
                if diff > rhoSnew2/shareDiff:
                    diff = False
            
            # get new guesses
            rhoSnew1, rhoSnew2, rhoSmin, debtMin, rhoSdip, debtsDip = \
                UpdateRhoDebtDip(rhoSdip, debtsDip, rhoSnew1, rhoSnew2, \
                                 necessary_debt1, necessary_debt2)
            
    plt.xlabel("rhoS", fontsize = 24)
    plt.ylabel("Necessary debt to reach probS", fontsize = 24)
    plt.title("Necessary debt for different rhoS", fontsize = 30)
    fig.savefig("Figures/rhoSvsDebts/CropAlloc_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    
    if len(debtsDip) <= 1:
        if debt_bottom < debt_top:
            rhoS = 0
            necessary_debt = debt_bottom
        elif debt_bottom > debt_top:
            rhoS = UpperBorder
            necessary_debt = debt_top
    else:
        if debtsDip[0] == min(debtsDip):
            rhoS = 0
            necessary_debt = debt_bottom
        elif debtsDip[-1] == min(debtsDip):
            rhoS = UpperBorder
            necessary_debt = debt_top
        else:
            rhoS = rhoSdip[debtsDip.index(min(debtsDip))]
            necessary_debt = min(debtsDip)
            
    return(rhoS, necessary_debt)

# def MinimizeNecessaryDebtTest(args, probS, rhoSini, accuracy, prints):
#     # minimizing the absolute value of debt neccessary to reach probS, as we 
#     # also don't want to be too good (as probabilities higher than probS will
#     # lead to negative debt)
#     rhoSs = [0, 10, 20, 30, 40, 50, 60, 70]
    
#     plt.figure(figsize = figsize)   
    
#     for rhoS in rhoSs:
#         status, crop_alloc, meta_sol, prob, durations = \
#             SolveReducedcLinearProblemGurobiPy(args, 0, rhoS, probS, prints = False) 
#         necessary_debt = - np.quantile(meta_sol["final_fund"], 1 - probS)
        
#         # report
#         ReportProgressFindingRho(rhoS, meta_sol, accuracy, durations, \
#                                  ProbType = "S", necessary_debt, prints = prints)
        
#         # plot
#         plt.scatter(rhoS, necessary_debt, s = 10)
    
#     plt.xlabel("rhos")
#     plt.ylabel("necessary debt to reach probS")
    
#     return(rhoS)

def UpdatedRhoGuess(meta_sol, rhoLastUp, rhoLastDown, rhoOld, prob, accuracy, probType = "F"):
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
            rhoNew = rhoOld + (rhoLastDown - rhoOld)/ 2 
    else:
        rhoLastDown = rhoOld
        if rhoLastUp == 0:
            rhoNew = rhoOld / 4
        else:
            rhoNew = rhoOld - (rhoOld - rhoLastUp) / 2
    
    return(rhoNew, rhoLastDown, rhoLastUp)

def ReportProgressFindingRho(rhoOld, meta_sol, accuracy, durations, \
                             accuracy_int = False, ProbType = "S", debt = False, prefix = "", prints = True):
    if ProbType == "F":
        currentProb = meta_sol["prob_food_security"]
        unit = " $/10^3kcal"
    elif ProbType == "S":
        currentProb = meta_sol["prob_staying_solvent"]
        unit = " $/$"
    
    if debt:
        debt_text = ", nec. debt: " + str(np.round(debt, 3)) + " 10^9$"
    else:
        debt_text = ""
        
    if accuracy_int:
        accuracy_text = " (current accuracy interval: " + str(np.round(accuracy_int, 2)) + ")"
    else:
        accuracy_text = ""
        
    printing("     " + prefix + "rho" + ProbType + ": " + str(rhoOld) + unit + \
          ", prob" + ProbType + ": " + str(np.round(currentProb * 100, \
                                                    accuracy -1)) + \
          "%" + debt_text + ", time: " + str(np.round(durations[2], 2)) + "s" + accuracy_text, \
              prints = prints)
    

def GetRhoS(args, probS, rhoSini, shareDiff, accuracy = 3, prints = True):
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
        Desired decimal places of accuracy of the obtained probF. 
        The default is 3.
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
    if rhoSini is not None:
        printing("     Checking guess from run with lower N", prints = prints)
        status, crop_alloc, meta_sol, prob, durations = \
                        SolveReducedcLinearProblemGurobiPy(args, 0, rhoSini, probS, prints = False)  
        ReportProgressFindingRho(rhoSini, meta_sol, accuracy, durations, \
                                 ProbType = "S", prefix = "Guess: ", prints = prints)
        if np.round(meta_sol["prob_staying_solvent"], accuracy) == probS:
            rhoScheck = rhoSini - rhoSini/shareDiff
            status, crop_alloc, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args, 0, rhoScheck, probS, prints = False)  
            ReportProgressFindingRho(rhoScheck, meta_sol, accuracy, durations, \
                                     ProbType = "S", prefix = "Check: ", prints = prints)
            if np.round(meta_sol["prob_staying_solvent"], accuracy) < probS:
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
                             accuracy_int, ProbType = "S", prints = prints)

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
                                 accuracy_int, ProbType = "S", prints = prints)
            
        # remember guess
        rhoSold = rhoSnew
        if np.round(meta_sol["prob_staying_solvent"], accuracy) == probS \
            and lowestCorrect > rhoSnew:
            lowestCorrect = rhoSnew

    # last report
    ReportProgressFindingRho(rhoSnew, meta_sol, accuracy, durations, \
                             accuracy_int, ProbType = "S", prints = prints)    
    
    return(rhoS)
            
def GetPenalties(settings, args, other, probF, probS, \
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
    other : dict
        Other information on the model setup (on the yield distributions).
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
    probSnew : float
        The new probability for solvency (different from the input probS in 
        case that probability can't be reached).
    needed_import : float
        Amount of food that needs to imported to reach the probability for
        food seecurity probF.
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
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoF calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 1)
            if rhoFini is None:
                rhoFini = GetInitialGuess(dict_rhoFs, SettingsFirstGuess)
            # calculating rhoF
            printing("Calculating rhoF and import", prints = prints)
            try:
                rhoF, maxProbF, maxProbS, needed_import, crop_alloc, meta_sol = \
                        GetRhoF(args, other, probF, rhoFini, prints = prints)
            except PenaltyException as e:
                raise PenaltyException(message =  str(e) + " (called from GetPenalties)")
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
        else:
            # if this setting was calculated for a lower N and no initial
            # guess was given, we use the rhoS calculted for the lower N as 
            # initial guess (if no initial guess can be provided we set it
            # to 100)
            if rhoSini is None:
                rhoSini = GetInitialGuess(dict_rhoSs, SettingsFirstGuess)
            # calculating rhoS
            printing("\nCalculating rhoS", prints = prints)
            rhoS, necessary_debt, maxProbS, maxProbF = GetRhoS_Wrapper(args, other, probS, rhoSini, SettingsAffectingRhoS, prints = prints)
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
    
# %% ########### FUNCTIONS TO GET META INFORMATION ON MODEL OUTPUT ############

def ObjectiveFunction(x, num_clusters, num_crops, N, \
                    cat_clusters, terminal_years, ylds, costs, demand, imports, \
                    ini_fund, tax, prices, T, guaranteed_income, crop_cal, 
                    rhoF, rhoS): 
    """
    Given input parameters and a crop allocation, this calculates the value
    of the objevtive function, i.e. the total costs, and returns the different
    terms and aspects affecting the result as well.
    
    Parameters
    ----------
    x : np.array of size (T, num_crops, len(k_using),)
        Gives allocation of area to each crop in each cluster.
    num_clusters : int
        The number of crops that are used.
    num_crops : int
        The number of crops that are used.
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function.
    cat_clusters : np.array of size (N, T, len(k_using))
        Indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0.
    terminal_years : np.array of size (N,) 
        Indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample.
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        Yield samples in 10^6t/10^6ha according to the presence of 
        catastrophes.
    costs : np array of size (num_crops,) 
        Cultivation costs for each crop in 10^9$/10^6ha.
    demand : np.array of size (T,)
        Total food demand for each year in 10^12kcal.
    imports : float
        Amount of food that will be imported and therefore is substracted from
        the demand.
    ini_fund : float
        Initial fund size in 10^9$.
    tax : float
        Tax rate to be paied on farmers profits.
    prices : np.array of size (num_crops,) 
        Farm gate prices.
    T : int
        Number of years to cover in the simulation.
    guaranteed_income : np.array of size (T, len(k_using)) 
        Income guaranteed by the government for each year and cluster in case 
        of catastrophe in 10^9$.
    crop_cal : np.array of size (num_crops,)
        Calorie content of the crops in 10^12kcal/10^6t.
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.

    Returns
    -------
    exp_tot_costs :  float
        Final value of objective function, i.e. sum of cultivation and penalty
        costs in 10^9$.
    fixcosts : np.array of size (N,)
        Cultivation costs in 10^9$ for each yield sample (depends only on the 
        final year of simulation for each sample).
    shortcomings : np.array of size (N, T)
        Shortcoming of the food demand in 10^12kcal for each year in each 
        sample.
    exp_income : np.array of size (T, len(k_using))
        Average profits of farmers in 10^9$ for each cluster in each year.
    profits : np.array of size (N, T, len(k_using))
        Profits of farmers in 10^9$ per cluster and year for each sample.
    avg_shortcomings : np.array of size (T,)
        Average shortcoming of the food demand in 10^12kcal in each year.
    fp_penalties : np.array of size (N, T) 
        Penalty payed because of food shortages in each year for each sample.
    avg_fp_penalties : np.array of size (T,)
        Average penalty payed because of food shortages in each year.
    sol_penalties : np.array of size (N,)
        Penalty payed because of insolvency in each sample.
    final_fund : np.array of size (N,)
        The fund size after payouts in the catastrophic year for each sample.
    payouts : np.array of size (N, T, len(k_using))
        Payouts from the government to farmers in case of catastrope per year
        and cluster for each sample. 
    yearly_fixed_costs : np.array of size (N, T) 
        Total cultivation costs in each year for each sample.     
    """

    # preparing x for all realizations
    # x = np.reshape(x,[T, num_crops, num_clusters]) already in this format
    X = np.repeat(x[np.newaxis, :, :, :], N, axis=0)
    for c in range(0, N):
        t_c = int(terminal_years[c])                   # catastrophic year
        if t_c > -1:
            X[c, (t_c + 1) : , :, :] = np.nan  # no crop area after catastrophe
            
    # Production
    prod = X * ylds                          # nan for years after catastrophe
    kcal  = np.swapaxes(prod, 2, 3)
    kcal = kcal * crop_cal                   # relevant for realistic values
    
    # Shortcomings
    S = demand - imports - np.sum(kcal, axis = (2, 3)) # nan for years after catastrophe
    np.seterr(invalid='ignore')
    S[S < 0] = 0
    np.seterr(invalid='warn')
    
    # fixed costs for cultivation of crops
    fixed_costs =  X * costs
    
    # Yearly profits
    P =  prod*prices - fixed_costs          # still per crop and cluster, 
                                            # nan for years after catastrophe
    P = np.sum(P, axis = 2)                 # now per cluster
    # calculate expected income
    exp_income = np.nanmean(P, axis = 0) # per year and cluster
    # P[P < 0] = 0   # we removed the max(0, P) and min(I_gov, I_gov-P) for
                     # linearization prurposes
 
    # Payouts
    payouts = guaranteed_income - P  # as we set negative profit to zero,
                                     # government doesn't cover those
                                     # it only covers up to guaranteed income.
                                     # -> this is not true any more!
    np.seterr(invalid='ignore')
    payouts[(cat_clusters == 0) + (payouts < 0)] = 0
    np.seterr(invalid='warn')
                # no payout if there is no catastrophe, even if guaranteed 
                # income is not reached
                # in the unlikely case where a cluster makes more than the
                # guaranteed income despite catastrophe, no negative payouts!
                      
    # Final fund
    ff = ini_fund + tax * np.nansum(P, axis = (1,2)) - \
                                            np.nansum(payouts, axis = (1,2))
    ff[ff > 0] = 0
    
    # expected total costs
    exp_tot_costs = np.mean(np.nansum(fixed_costs, axis = (1,2,3)) + \
                            rhoF * np.nansum(S, axis = 1) + rhoS * (- ff))
    
    return(exp_tot_costs, 
        np.nansum(fixed_costs, axis = (1,2,3)), #  fixcosts (N)
        S, # shortcomings per realization and year
        exp_income, # expected income (T, k)
        P, # profits
        np.nanmean(S, axis = 0) , # yearly avg shortcoming (T)
        rhoF * S, # yearly food demand penalty (N x T)
        np.nanmean(rhoF * S, axis = 0), # yearly avg fd penalty (T)
        rhoS * (- ff), # solvency penalty (N)
        ini_fund + tax * np.nansum(P, axis = (1,2)) - \
          np.nansum(payouts, axis = (1,2)), # final fund per realization
        payouts, # government payouts (N, T, k)
        np.nansum(fixed_costs, axis = (2,3)), #  fixcosts (N, T)
        ) 

def GetMetaInformation(crop_alloc, args, rhoF, rhoS, probS = None):
    """
    To get metainformation for final crop allocation after running model.

    Parameters
    ----------
    crop_alloc : np.array of size (T*num_crops*len(k_using),)
        Gives allocation of area to each crop in each cluster.
    args : dict
        Dictionary of arguments needed as model input (as given by 
        SetParameters())
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.

    Returns
    -------
    meta_sol : dict 
        additional information about the model output.
        
        - exp_tot_costs: Final value of objective function, i.e. sum of 
          cultivation and penalty costs in 10^9$.
        - fixcosts: Cultivation costs in 10^9$ for each yield sample (depends 
          only on the final year of simulation for each sample).
        - shortcomings: Shortcoming of the food demand in 10^12kcal for each year in each 
          sample.
        - exp_income: Average profits of farmers in 10^9$ for each cluster in
          each year.
        - profits: Profits of farmers in 10^9$ per cluster and year for each
          sample.
        - avg_shortcomings: Average shortcoming of the food demand in 
          10^12kcal in each year.
        - fp_penalties: Penalty payed because of food shortages in each year 
          for each sample.
        - avg_fp_penalties: Average penalty payed because of food shortages in 
          each year.
        - sol_penalties: Penalty payed because of insolvency in each sample.
        - final_fund: The fund size after payouts in the catastrophic year for 
          each sample.
        - prob_staying_solvent: Probability for solvency of the government fund
          after payouts.
        - prob_food_security: Probability for meeting the food femand.
        - payouts: Payouts from the government to farmers in case of catastrope 
          per year and cluster for each sample. 
        - yearly_fixed_costs: Total cultivation costs in each year for each 
          sample.   
        - num_years_with_losses: Number of occurences where farmers of a 
          cluster have negative profits.

    """
    
    
    # running the objective function with option meta = True to get 
    # intermediate results of the calculation
    exp_tot_costs, fix_costs, shortcomings, exp_incomes, profits, \
    exp_shortcomings,  fd_penalty, avg_fd_penalty, sol_penalty, final_fund, \
    payouts, yearly_fixed_costs = ObjectiveFunction(crop_alloc, 
                                           args["k"], 
                                           args["num_crops"],
                                           args["N"], 
                                           args["cat_clusters"], 
                                           args["terminal_years"],
                                           args["ylds"], 
                                           args["costs"], 
                                           args["demand"],
                                           args["import"],
                                           args["ini_fund"],
                                           args["tax"],
                                           args["prices"],
                                           args["T"],
                                           args["guaranteed_income"],
                                           args["crop_cal"], 
                                           rhoF, 
                                           rhoS)
    
    # calculationg additional quantities:
    # probability of solvency in case of catastrophe
    prob_staying_solvent = np.sum(final_fund >= 0) /  args["N"]
    tmp = np.copy(shortcomings)
    np.seterr(invalid='ignore')
    tmp[tmp > 0] = 1
    np.seterr(invalid='warn')
    prob_food_security = 1 - np.nanmean(tmp)
    np.seterr(invalid='ignore')
    num_years_with_losses = np.sum(profits<0)  
    np.seterr(invalid='warn')
    
    # group information into a dictionary
    meta_sol = {"exp_tot_costs": exp_tot_costs,
                "fix_costs": fix_costs,
                "shortcomings": shortcomings,
                "exp_incomes": exp_incomes,
                "profits": profits,
                "exp_shortcomings": exp_shortcomings,
                "fd_penalty": fd_penalty,
                "avg_fd_penalty": avg_fd_penalty,
                "sol_penalty": sol_penalty,
                "final_fund": final_fund,
                "prob_staying_solvent": prob_staying_solvent,
                "prob_food_security": prob_food_security,
                "payouts": payouts,
                "yearly_fixed_costs": yearly_fixed_costs,
                "num_years_with_losses": num_years_with_losses}
    
    if probS is not None:
        meta_sol["necessary_debt"] = - np.quantile(meta_sol["final_fund"], 1 - probS)
    
    return(meta_sol)  
    
# %% #################### VALUE OF THE STOCHASTIC SOLUTION ####################  
 
def VSS(settings, args, rhoF, rhoS, probS = None):
    """
    Calculating expected total costs using the optimal crop area allocation 
    assuming expected yield values as yield realization.

    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    args : dict
        Dictionary of arguments needed as model input (as given by 
        SetParameters()).
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.

    Returns
    -------
    crop_alloc_vss : np.array
        Optimal crop areas for all years, crops, clusters in the deterministic
        setting.
    meta_sol_vss : dict
        Dictionary of additional information on the solution of the
        deterministic model.

    """
    # get arguments to calculate deterministic solution (in particular the 
    # expected yields instead of yield samples)
    args_vss, other = SetParameters(settings, VSS = True, prints = False)
    
    # solve model for the expected yields
    status, crop_alloc_vss, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args_vss, rhoF, rhoS, probS, prints = False)
                
    # get information of using VSS solution in stochastic setting
    meta_sol_vss = GetMetaInformation(crop_alloc_vss, args, rhoF, rhoS, probS)
    return(crop_alloc_vss, meta_sol_vss)      
    
# %% ############ IMPLEMENTING AND SOLVING LINEAR VERSION OF MODEL ############

def SolveReducedcLinearProblemGurobiPy(args, rhoF, rhoS, probS = None, prints = True):
    """
    Sets up and solves the linear form of the food security problem.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input (as given by 
        SetParameters()).
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    status : int
        status of solver (optimal: 2)
    crop_alloc : np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output
    prob : gurobi model
        The food security model that was set up.
    durations : list
        time for setting up the model, time for solving, and total time (in
        sec.)

    """
    printing("\nSolving Model", prints = prints)
    
    start = tm.time()
    
# no output to console
    env = gp.Env(empty = True)    
    env.setParam('OutputFlag', 0)
    env.start()
    
# problem
    prob = gp.Model("SustainableFoodSecurity", env = env)
    
    # dimension stuff
    T = args["T"]
    K = len(args["k_using"])
    J = args["num_crops"]
    N = args["N"]
    termyear_p1 = args["terminal_years"] + 1
    termyear_p1[termyear_p1 == 0] = T
    termyear_p1 = termyear_p1.astype(int)
    
# index tupes for variables and constraints
    indVfood = flatten([[(t, s) for t in range(0, termyear_p1[s])] \
                        for s in range(0, N)])
    
    indW = flatten(flatten([[[(t, k, s) for t in range(0, termyear_p1[s])] \
                             for k in range(0,K)] for s in range(0, N)]))   
            
    indCultCosts = flatten([[(t, j, k) for (t,j,k) in \
              it.product(range(0,termyear_p1[s]), range(0, J), range(0, K))] \
              for s in range(0, N)])
            
    indMaxArea = list(it.product(range(0, K), range(0, T)))
    indCropsClusters = list(it.product(range(0, J), range(0, K)))
    
# variables
    x = prob.addVars(range(0, T), range(0, J), range(0, K), name = "x")
    Vfood = prob.addVars(indVfood, name = "Vfood")
    Vsol = prob.addVars(range(0, N), name = "Vsol")
    Wgov = prob.addVars(indW, name = "Wgov")


# objective function
    obj = gp.quicksum([1/N * x[t,j,k] * args["costs"][j,k] \
                        for (t,j,k) in indCultCosts] + \
                       [1/N * rhoF * Vfood[t, s] for (t, s) in indVfood] + \
                       [1/N * rhoS * Vsol[s] for s in range(0, N)] + \
                          [0 * Wgov[t, k, s] for (t, k, s) in indW])
    prob.setObjective(obj, gp.GRB.MINIMIZE)
            
         
# constraints 1
    prob.addConstrs((gp.quicksum([x[t, j, k] for j in range(0, J)]) \
                 <= args["max_areas"][k] for (k, t) in indMaxArea), "c_marea")
       
# constraints 2
    prob.addConstrs((gp.quicksum([Vfood[t, s]] + \
                [args["ylds"][s, t, j, k] * x[t, j, k] * args["crop_cal"][j] \
                          for (j, k) in indCropsClusters]) \
                 >= (args["demand"][t] - args["import"]) \
                                 for (t, s) in indVfood), "c_demand")
    
# constraints 3
    prob.addConstrs((gp.quicksum([-Vsol[s]] + \
                        [- args["tax"] * (args["ylds"][s, t, j, k] * \
                                x[t, j, k] * args["prices"][j, k] - \
                                x[t, j, k] * args["costs"][j, k]) \
                           for (j, t, k) in it.product(range(0, J), \
                                 range(0, termyear_p1[s]), range(0, K))] + \
                        [args["cat_clusters"][s, t, k] * Wgov[t, k, s] \
                           for (t, k) in it.product(range(0, termyear_p1[s]), \
                                range(0, K))]) \
                 <= args["ini_fund"] for s in range(0, N)), "c_sol")
        
# constraints 4
    prob.addConstrs((gp.quicksum([- Wgov[t, k, s]] + \
            [- args["ylds"][s, t, j, k] * x[t, j, k] * args["prices"][j, k] + \
            x[t, j, k] * args["costs"][j, k] for j in range(0, J)]) \
         <= - args["guaranteed_income"][t, k] for (t, k, s) in indW), "c_gov")

# solving
    middle = tm.time()
    
    # prob.write("../ForPublication/TestingLinearization" \
    #                                        + "/gurobipy_test.lp")
    # prob.write("../ForPublication/TestingLinearization" \
    #                                        + "/gurobipy_test.mps")
    # return()

    prob.optimize()
    
    end = tm.time()
    durationBuild = middle - start
    durationSolve = end - middle
    durationTotal = end - start
    durations = [durationBuild, durationSolve, durationTotal]
    
    status = prob.status
    
# get results
    crop_alloc = np.zeros((T, J, K))
    meta_sol = []
    
    if status != 2:
        warn.warn("Non-optimal status of solver")
        return(status, crop_alloc, meta_sol, prob, durations)
    else:        
        for t in range(0, T):
            for j in range(0, J):
                for k in range(0, K):
                    crop_alloc[t, j, k] = prob.getVarByName("x[" + str(t) + \
                                        "," + str(j) + "," + str(k) + "]").X
                  
        meta_sol = GetMetaInformation(crop_alloc, args, \
                                                    rhoF, rhoS, probS)
        
        # if meta_sol["num_years_with_losses"] > 0:
        #     warn.warn(str("Please notice that in " + \
        #               str(meta_sol["num_years_with_losses"]) + \
        #               " years/clusters profits are negative."))
            
    # printing("      " + "\u005F" * 21, prints = prints)
    printing("     Time      Setting up model: " + \
            str(np.round(durations[0], 2)) + "s", prints = prints)
    printing("               Solving model: " + \
            str(np.round(durations[1], 2)) + "s", prints = prints)
    printing("               Total: " + \
            str(np.round(durations[2], 2)) + "s", prints = prints) 
    # printing("      " + "\u0305 " * 21, prints = prints)           
                
    return(status, crop_alloc, meta_sol, prob, durations)
        
# %% ################### OUT OF SAMLE VALIDATION OF RESULT ####################  

def OutOfSampleVal(crop_alloc, settings, rhoF, rhoS, \
                   expected_incomes, M, probS = None, prints = True):
    """
    For validation, the objective function is re-evaluate, using the optimal
    crop allocation but a higher sample size.    
    
    Parameters
    ----------
    crop_alloc : np.array
        Optimal crop areas for all years, crops, clusters.
    settings : dict
        the model settings that were used     
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.
    expected_incomes : np.array 
        expected income in secnario without government and probF = 95% for 
        each clutser.               
    M : int
        Sample size used for validation.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Yields
    ------
    meta_sol_vss : dict
        Additional information on the outcome for the higher sample size.

    """
    
    # printing("      " + "\u005F" * 21, prints = prints)
    # higher sample size
    settings_val = settings.copy()
    settings_val["N"] = M
    # expected income
    settings_val["expected_incomes"] = expected_incomes
    # get yield samples
    printing("     Getting parameters and yield samples", prints = prints)
    args, other = SetParameters(settings_val, prints = False)
    
    # run objective function for higher sample size
    printing("     Objective function", prints = prints)
    meta_sol_vss = GetMetaInformation(crop_alloc, args, rhoF, rhoS, probS)
    # printing("      " + "\u0305 " * 21, prints = prints)           
    return(meta_sol_vss)      

# %% ########################## PLOTTING FUNCTIONS ############################  

def PlotModelOutput(PlotType = "CropAlloc", cols = None, cols_b = None, \
                    figsize = figsize, title = None, file = None, **kwargs):
    """
    Creating different types of plots based on the model in- and output

    Parameters
    ----------
    PlotType : str, optional
        Specifying what kind of information should be plotted. The default is 
        "CropAlloc".
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    title : str
        Basis of title for for the plot (will be completed with information on
        the clusters). The default is None.
    file : str
        Filename to save resulting figure. If None, figure will not be saved.
        The default is None.    
    **kwargs
        Additional parameters passed along to the different plotting functions.

    Returns
    -------
    None.

    """
    
    # defining colors
    if cols is None:            
        cols = ["royalblue", "darkred", "grey", "gold", \
                "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                "indigo"]
    if cols_b is None:
        cols_b = ["dodgerblue", "red", "darkgrey", "y", \
              "lime", "cyan", "orchid", "lightseagreen", "mediumpurple"]      
            
    # plotting the specified information
    if PlotType == "CropAlloc":
        PlotCropAlloc(cols = cols, cols_b = cols_b, figsize = figsize, \
                      title = title, file = file, **kwargs)
    
    return()

def PlotCropAlloc(crop_alloc, k, k_using, max_areas, cols = None, cols_b = None, \
                  figsize = figsize, title = None, file = None, sim_start = 2017):
    """
    Plots crop area allocations over the years for all clusters. Should be 
    called through PlotModelOutput(PlotType = "CropAlloc").

    Parameters
    ----------
    crop_alloc : np.array of size (T*num_crops*len(k_using),)
        Gives allocation of area to each crop in each cluster.
    k : int, optional
        Number of clusters in which the area is to be devided. 
    k_using :  "all" or a list of int
        Specifies which of the clusters are to be considered in the model. 
        The default is "all".
    max_areas : np.array of size (len(k_using),) 
        Upper limit of area available for agricultural cultivation in each
        cluster
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    title : str
        Basis of title for for the plot (will be completed with information on
        the clusters). The default is None.
    file : str
        Filename to save resulting figure. If None, figure will not be saved.
        The default is None.
    sim_start : int
        The first year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    
    if title is None:
        title = ""
    else:
        title = " - " + title
        
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    [T, J, K] = crop_alloc.shape
    years = range(sim_start, sim_start + T)
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.95,
                    wspace=0.15, hspace=0.35)
    
    # overview of all crop area allocations in one plot
    if K > 2:
        ax = fig.add_subplot(1,2,1)
        plt.plot(years, np.sum(crop_alloc, axis = (1,2)), color = "k", \
                 lw = 2, alpha = 0.7)
    else:
        ax = fig.add_subplot(1,1,1)
    for cl in range(0, K):
        plt.plot(years, np.repeat(max_areas[cl], len(years)), \
                 color = cols_b[k_using[cl]-1], lw = 5, alpha = 0.4)
        plt.plot(years, crop_alloc[:,0,cl], color = cols[k_using[cl]-1], \
                 lw = 2, linestyle = "--")
        plt.plot(years, crop_alloc[:,1,cl], color = cols[k_using[cl]-1], \
                 lw = 2, label = "Cluster " + str(k_using[cl]))
    plt.xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.yaxis.offsetText.set_fontsize(24)
    ax.xaxis.offsetText.set_fontsize(24)
    plt.xlabel("Years", fontsize = 30)
    plt.ylabel(r"Crop area in [$10^6$ ha]", fontsize = 30)
    if K > 2:
        plt.title("Overview" + title, fontsize = 32, pad = 10)
    elif K == 1:
        plt.title("Cluster " + str(k_using[0]) + title, \
                  fontsize = 32, pad = 10)
    elif K == 2:
        plt.title("Cluster " + str(k_using[0]) + " and " + str(k_using[1]) + title, \
                 fontsize = 32, pad = 10)
    
    # crop area allocations in separate subplots per cluster
    if K > 2:
        rows = math.ceil(K/2)
        whichplots = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
        for cl in range(0, K):
            ax = fig.add_subplot(rows, 4, whichplots[cl])
            plt.plot(years, np.repeat(max_areas[cl], len(years)), \
                     color = cols_b[cl], lw = 5, alpha = 0.4)
            plt.plot(years, crop_alloc[:,0,cl], color = cols[k_using[cl]-1], \
                     lw = 2, linestyle = "--")
            plt.plot(years, crop_alloc[:,1,cl], color = cols[k_using[cl]-1], \
                     lw = 2, label = "Cluster " + str(k_using[cl]))
            plt.ylim([-0.05 * np.max(max_areas), 1.1 * np.max(max_areas)])
            plt.xlim(years[0] - 0.5, years[-1] + 0.5)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.yaxis.offsetText.set_fontsize(16)
            ax.xaxis.offsetText.set_fontsize(16)
            # ax.text(0.05, 0.91, "Cluster " + str(int(k_using[cl])), \
            #         fontsize = 16, transform = ax.transAxes, \
            #         verticalalignment = 'top', bbox = props)
            if cl == 0:
                plt.title("                        Separate clusters", \
                          fontsize = 32, pad = 8) 
    
    
    if file is not None:
        fig.savefig("Figures/CropAllocs/CropAlloc_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    return()

# %% ######################### GROUPING CLUSTERS ############################## 

def GroupingClusters(k = 9, size = 5, aim = "Similar", adjacent = True, title = None, figsize = figsize):
    
    with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
        distance = pickle.load(fp)  

    with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
                 "_PearsonDistSPEI_ProfitableArea.txt", "rb") as fp:  
        pickle.load(fp) # clusters
        pickle.load(fp) # costs
        medoids = pickle.load(fp)
    DistMedoids = MedoidMedoidDist(medoids, distance)
    
    with open("InputData/Clusters/AdjacencyMatrices/k" + str(k) + "AdjacencyMatrix.txt", "rb") as fp:
        AdjacencyMatrix = pickle.load(fp)
    
    clusters = list(range(0,k))
    
    BestCosts = None
    BestGrouping = None
    valid = 0
    
    for grouping in AllGroupings(clusters, size):
        if adjacent and not CheckAdjacency(clusters, grouping, AdjacencyMatrix):
            continue
        valid += 1
        TmpCosts = CostsGrouping(grouping, DistMedoids)
        BestGrouping, BestCosts = \
           UpdateGrouping(BestCosts, TmpCosts, BestGrouping, grouping, aim)
    
    ShiftedGrouping = []
    for gr in BestGrouping:
        ShiftedGrouping.append(tuple([i + 1 for i in list(gr)]))
               
    if title is not None:
        VisualizeClusterGroups(k, size, aim, ShiftedGrouping, title, figsize, \
                          fontsizet = 20, fontsizea = 16)
    
    if adjacent:
        ad = "Adj"
    else:
        ad = ""
            
    with open("InputData/Clusters/ClusterGroups/GroupingSize" + \
                              str(size) + aim + ad + ".txt", "wb") as fp:
        pickle.dump(ShiftedGrouping, fp)
               
    return(ShiftedGrouping, BestCosts, valid)
      
def AllGroupings(lst, num):
    if len(lst) < num:
        yield []
        return
    if len(lst) % num != 0:
        # Handle odd length list
        for i in it.combinations(lst, len(lst) % num):
            lst_tmp = lst.copy()
            for j in i:
                lst_tmp.remove(j)
            for result in AllGroupings(lst_tmp, num):
                yield [i] + result
    else:
        for i in it.combinations(lst[1:], num - 1):
            i = list(i)
            i.append(lst[0])
            i.sort()
            i = tuple(i)
            lst_tmp = lst.copy()
            for j in i:
                lst_tmp.remove(j)
            for rest in AllGroupings(lst_tmp, num):
                yield [i] + rest        
                
def CheckAdjacency(clusters, grouping, AdjacencyMatrix):
    for gr in grouping:
        if len(grouping) == 1:
            continue
        ClustersNot = clusters.copy()
        for i in gr:
            ClustersNot.remove(i)
        AdjacencyMatrixRed = np.delete(AdjacencyMatrix, ClustersNot, 0)
        AdjacencyMatrixRed = np.delete(AdjacencyMatrixRed, ClustersNot, 1)
        AdjacencyMatrixRed = np.linalg.matrix_power(AdjacencyMatrixRed, len(gr)-1)
        if np.sum(AdjacencyMatrixRed[0,:] == 0) > 0:
            return(False)
    return(True)

def CostsGrouping(grouping, dist):
    costs = 0
    for gr in grouping:
        if len(gr) == 1:
            continue
        for i in it.combinations(list(gr), 2):
            costs = costs + dist[i[0], i[1]]
    return(costs)

def UpdateGrouping(BestCosts, TmpCosts, BestGrouping, grouping, aim):
    if BestCosts is None:
        return(grouping, TmpCosts)
    if aim == "Similar":
        if TmpCosts < BestCosts:
            return(grouping, TmpCosts)
        return(BestGrouping, BestCosts)
    elif aim == "Dissimilar":
        if TmpCosts > BestCosts:
            return(grouping, TmpCosts)
        return(BestGrouping, BestCosts)
            
def MedoidMedoidDist(medoids, dist):
    k = len(medoids)
    res = np.empty([k,k])
    res.fill(0)
    # get distance to all medoids
    for i in range(0, k):
        for j in range(0, k):
            if i == j:
                continue
            res[i, j] = dist[medoids[i][0]][medoids[i][1]] \
                                                [medoids[j][0], medoids[j][1]]
    return(res)

def VisualizeClusterGroups(k, size, aim, grouping, title, figsize, \
                          fontsizet = 20, fontsizea = 16):
    
    from mpl_toolkits.basemap import Basemap
        
    # shift as cisualization works with clusters 0, ..., k-1
    ShiftedGrouping = []
    for gr in grouping:
        ShiftedGrouping.append(tuple([i - 1 for i in list(gr)]))
    
    fig = plt.figure(figsize = figsize)
    
    # clusters
    with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
                 "_PearsonDistSPEI_ProfitableArea.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        
    # number of groups
    num_groups = len(ShiftedGrouping)
    
    # assignment to groups
    assign = np.empty(k)
    for idx, gr in enumerate(ShiftedGrouping):
        for j in gr:
            assign[j] = idx
    
    # getting longitudes and latitudes of region
    with open("InputData/Other/LatsLonsArea.txt", "rb") as fp:
        lats_WA = pickle.load(fp)
        lons_WA = pickle.load(fp)
        
    # initialize map
    m = Basemap(llcrnrlon=lons_WA[0], llcrnrlat=lats_WA[0], \
                urcrnrlat=lats_WA[-1], urcrnrlon=lons_WA[-1], \
                resolution='l', projection='merc', \
                lat_0=lats_WA.mean(),lon_0=lons_WA.mean())
    
    lon, lat = np.meshgrid(lons_WA, lats_WA)
    xi, yi = m(lon, lat)
    
    # Plot Data
    m.drawmapboundary(fill_color=(0.9745,0.9745,0.9857))
    cmap = cm.Paired._resample(num_groups)
    cmap = cmap(np.linspace(0,1,num_groups))
    NewMap = np.empty((0,4))
    for idx, gr in enumerate(ShiftedGrouping):
        l = len(gr)
        for j in range(0, l):
            NewColor = col.to_rgb(cmap[idx])
            NewColor = (0.6 + (j+1) * 0.35/(l+1)) * np.array(NewColor)
            NewColor = col.to_rgba(NewColor)
            NewMap = np.vstack([NewMap, NewColor])
    SortedColors = NewMap.copy()
    idx = 0
    for j in range(0,k):
        for i in [i for i, e in enumerate(assign) if e == j]:
            SortedColors[i] = NewMap[idx]
            idx +=1
            
    SortedColors = col.ListedColormap(SortedColors)
    
    c = m.pcolormesh(xi, yi, np.squeeze(clusters), cmap = SortedColors, \
                                             vmin = 0.5, vmax = k + 0.5)

    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], \
                            fontsize = fontsizea)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], \
                            fontsize = fontsizea)
    
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(linewidth=1.3, color = "dodgerblue")
    m.drawstates(linewidth=1.5)
    m.drawcountries(linewidth=1.5)
    m.drawrivers(linewidth=0.7, color='dodgerblue')
    
    # Add Title
    plt.title(title, fontsize = fontsizet, pad=20)
    plt.show()
    
    # add colorbar
    cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(c, cax = cb_ax)       
    cbar.set_ticks(range(1, k + 1))
    cbar.set_ticklabels(range(1, k + 1))
    
    # save figure
    fig.savefig("Figures/ClusterGroups/VisualizationGrouping_" + \
                "k" + str(k) + "s" + str(size) + aim + ".png", \
            bbox_inches = "tight", pad_inches = 1)
    return()

# %% ############################# ANALYSIS ###################################

def CompareCropAllocs(CropAllocs, MaxAreas, labels, title, legend_title, \
                      comparing = "clusters", cols = None, cols_b = None, filename = None, \
                      figsize = figsize, fig = None, ax = None, subplots = False, \
                      fs = "big", sim_start = 2017):
    """
    Creates a plot of the crop areas over time, either all in one plot or as
    separate subplots.

    Parameters
    ----------
    CropAllocs : list
        List of crop areas.
    MaxAreas : list
        List of maximum available areas.
    labels : list
        List of labels for the plot.
    title : str
        Title of the plot.
    legend_title : str
        Titlel of the legend.
    comparing : str, optional
        What cahnges between different results. Mainly relevant, as "clusters" 
        is treated differently. Any other string will be treated the same. The
        default is "clusters".
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    filename : str, optional
        Filename to save the resulting plot. If None, plot is not saved. The
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    fig : figure, optional
        If the function should create a plot within an already existing
        figure, the figure has to be passed to the function. The default is 
        None.
    ax : AxesSubplot, optional
        If the function should create a plot as a specific subplot of a figure, 
        the AxesSubplot has to be passed to the function. The default is None.
    subplots : boolean, optional
        If True, for each setting a new subplot is used. If False, all are 
        plotted in the same plot. The default is False.
    fs : str, optional
        Defines the fontsize, if fs is "big", larger fontsize is chosen, for 
        any other string smaller fontsizes are chosen. the smaller dontsizes
        should be used for figures with many plots. The default is "big".
    sim_start : int, optional
        First year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    
    if fs == "big":
        fs_axis = 24
        fs_label = 28
        fs_title = 30
        fs_sptitle = 18
        titlepad = 40
    else:
        fs_axis = 16
        fs_label = 18
        fs_title = 20
        fs_sptitle = 14
        titlepad = 30
        
    
    if cols is None:            
        cols = ["royalblue", "darkred", "grey", "gold", \
                "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                "indigo"]
    if cols_b is None:
        cols_b = ["dodgerblue", "red", "darkgrey", "y", \
              "lime", "cyan", "orchid", "lightseagreen", "mediumpurple"]  
    
    if fig is None:
        fig = plt.figure(figsize = figsize)
        
    if ax is None:
        ax = fig.add_subplot(1,1,1)
    if subplots:            
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        inner = gs.GridSpecFromSubplotSpec(subplots[0], subplots[1], ax, wspace=0.3, hspace=0.3)
        which = -1
        
    idx_col = -1
    for idx, cr in enumerate(CropAllocs):
        [T, J, K] = cr.shape
        years = range(sim_start, sim_start + T)
        for k in range(0, K):
            if comparing == "clusters":
                idx_col = labels[idx][k] - 1
                label = str(labels[idx][k])
            else:
                idx_col += 1
                label = labels[idx_col]
            if subplots:
                if (comparing == "clusters") and (subplots == (3,3)):
                    which = labels[idx][k] - 1
                else:
                    which += 1
                axTmp = plt.Subplot(fig, inner[which])
            else:
                axTmp = ax
            axTmp.plot(years, np.repeat(MaxAreas[idx][k], len(years)), \
                     color = cols_b[idx_col], lw = 5, alpha = 0.4)
            axTmp.plot(years, CropAllocs[idx][:,0,k], color = cols[idx_col], \
                     lw = 2, linestyle = "--")
            axTmp.plot(years, CropAllocs[idx][:,1,k], color = cols[idx_col], \
                     lw = 2, label = label)
            if subplots:
                axTmp.set_title(legend_title + str(label), fontsize = fs_sptitle)
                axTmp.set_xlim(years[0] - 0.5, years[-1] + 0.5)
                axTmp.xaxis.set_tick_params(labelsize=12)
                axTmp.yaxis.set_tick_params(labelsize=12)
                axTmp.yaxis.offsetText.set_fontsize(12)
                axTmp.xaxis.offsetText.set_fontsize(12)
                fig.add_subplot(axTmp)
                    
    if not subplots:
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.xaxis.set_tick_params(labelsize=fs_axis)
        ax.yaxis.set_tick_params(labelsize=fs_axis)
        ax.yaxis.offsetText.set_fontsize(fs_axis)
        ax.xaxis.offsetText.set_fontsize(fs_axis)
        ax.legend(title = legend_title, fontsize = fs_axis, title_fontsize = fs_label, loc = 7)
        
    ax.set_title(title, fontsize = fs_title, pad = titlepad)
    ax.set_xlabel("Years", fontsize = fs_label, labelpad = 30)
    ax.set_ylabel(r"Crop area in [$10^6$ ha]", fontsize = fs_label, labelpad = 30)
    
    if filename is not None:
        if subplots:
            filename = filename + "_sp"
        fig.savefig("Figures/CompareCropAllocs/" + filename + \
                    ".jpg", bbox_inches = "tight", pad_inches = 1, format = "jpg")
    plt.close()
    
def CompareCropAllocRiskPooling(CropAllocsPool, CropAllocsIndep, MaxAreasPool, MaxAreasIndep, 
                                labelsPool, labelsIndep, title = None, cols = None, cols_b = None, 
                                subplots = False, filename = None, figsize = figsize, 
                                sim_start = 2017):
    """
    Given two list of crop areas, max available areas and labels, this plots 
    the crop areas for comparison, either with subplots for each cluster of 
    each list, or all cluster per list in one plot.

    Parameters
    ----------
    CropAllocsPool : list
        List of crop areas of setting 1.
    CropAllocsIndep : list
        List of crop areas of setting 2.
    MaxAreasPool : list
        List of maximum available area for setting 1.
    MaxAreasIndep : list
        List of maximum available area for setting 2.
    labelsPool : list
        Labels for setting 1.
    labelsIndep : list
        Labels for setting 2.
    title : str, optional
        Title of the plot. The default is None.
    cols : list, optional
        List of colors to use. If None, a standard set of colors is used. The 
        default is None.
    cols_b : list, optional
        List of colors in a lighter shade to use. If None, a standard set of
        colors is used. The default is None.
    subplots : TYPE, optional
        DESCRIPTION. The default is False.
    filename : str, optional
        Filename to save the resulting plot. If None, plot is not saved. The
        default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    sim_start : int, optional
        First year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,2,1)
    CompareCropAllocs(CropAllocsPool, MaxAreasPool, labelsPool, "Risk pooling", "Cluster: ", \
                      fig = fig, ax = ax, subplots = subplots, fs = "small")
    ax = fig.add_subplot(1,2,2)
    CompareCropAllocs(CropAllocsIndep, MaxAreasIndep, labelsIndep, "Independent", "Cluster: ", \
                      fig = fig, ax = ax, subplots = subplots, fs = "small")
    
    if title is not None:
        fig.suptitle(title, fontsize = 30)
        
    if filename is not None:
        if subplots:
            filename = filename + "_sp"
        fig.savefig("Figures/CompareCropAllocsRiskPooling/" + filename + \
                    ".jpg", bbox_inches = "tight", pad_inches = 1)

def GetResultsToCompare(ResType = "k_using", PenMet = "prob", probF = 0.95, \
                       probS = 0.95, rhoF = None, rhoS = None, prints = True, \
                       groupSize = "", groupAim = "", adjacent = False, \
                       validation = None, **kwargs):
    """
    Function that loads results from different model runs, with one setting 
    changing while the others stay the same (e.g. different clusters for same
    settings).

    Parameters
    ----------
    ResType : str, optional
        Which setting will be changed to compare different results. Needs to 
        be the exact name of that setting. The default is "k_using".
    PenMet : "prob" or "penalties", optional
        "prob" if desired probabilities were given. "penalties" if penalties 
         were given directly. The default is "prob".
    probF : float, optional
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    probS : float, optional
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float or None, optional 
        Value that will be used as penalty for shortciómings of the food 
        demand (only relevant if PenMet == "penalties"). The default is None.
    rhoS : float or None, optional 
        Value that will be used as penalty for insolvency of the government 
        fund (only relevant if PenMet == "penalties"). The default is None.
    groupSize : int, optional
        The size of the groups. If we load reults for different cluster groups,
        this is relevant for the output filename. The default is "".
    groupAim : str ("Similar" or "Dissimilar"), optional
        The aim when grouping clusters. If we load reults for different cluster 
        groups, this is relevant for the output filename. The default is "".
    adjacent : boolean, optional
        Whether clusters within a group are adjacent. If we load reults for 
        different cluster groups, this is relevant for the output filename.
        The default is False.
    validation : None or int, optional
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.  
    **kwargs
        settings for the model, passed to DefaultSettingsExcept()  

    Returns
    -------
    CropAllocs : list
         List of crop allocations for the different settings.
    MaxAreas : list
         List of maximum available agricultural areas for the different 
         settings.
    labels : list
        List of labels for plots (given information on the setting that is 
        changing).
    fnIterate : str
        Filename to be used as basis for saving figures using this data.

    """

    settingsIterate = DefaultSettingsExcept(**kwargs)
    fnIterate = filename(settingsIterate, PenMet, validation, probF, probS, \
                     rhoF, rhoS, groupSize = groupSize, groupAim = groupAim, \
                     adjacent = adjacent)
    
    if type(kwargs["k_using"]) is tuple: 
        settingsIterate["k_using"] = [settingsIterate["k_using"]]
    if type(kwargs["k_using"] is list) and (sum([type(i) is int for \
                    i in kwargs["k_using"]]) == len(kwargs["k_using"])):
        settingsIterate["k_using"] = kwargs["k_using"]
        
        
    settingsIterate["probF"] = probF
    settingsIterate["probS"] = probS
    settingsIterate["rhoF"] = rhoF
    settingsIterate["rhoS"] = rhoS
    settingsIterate["validation"] = validation
    
    ToIterate = settingsIterate[ResType]
    
    if type(ToIterate) is not list:
        ToIterate = [ToIterate]
    
    CropAllocs = []
    MaxAreas = []
    labels = []
    
    for val in ToIterate:
        printing(ResType + ": " + str(val), prints = prints)
        if ResType == "k_using":
            if type(val) is int:
                val = [val]
            if type(val) is tuple:
                val = list(val)
                val.sort()
        settingsIterate[ResType] = val
        try:
            crop_alloc, meta_sol, status, durations, settings, args, \
            rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
            validation_values, fn = FoodSecurityProblem(PenMet = PenMet,
                                        prints = prints, **settingsIterate)
            CropAllocs.append(crop_alloc)
            MaxAreas.append(args["max_areas"])
            labels.append(val)
        except PenaltyException as e:
            print(colored("Case " + str(val) + " --- " + str(e), 'red'))
        
    return(CropAllocs, MaxAreas, labels, fnIterate)
        
# %% ############################### AUXILIARY ################################  

def flatten(ListOfLists):
    """
    Turn list of lists into a list.

    Parameters
    ----------
    ListOfLists : list
        A lits with lists as elements.

    Returns
    -------
    FlatList : list
        List made out of single elements.

    """
    return(list(it.chain(*ListOfLists)))

def MakeList(grouping):
    """
    Function to make a single list out of the clusters in a grouping (which 
    is given by a list of tuples)

    Parameters
    ----------
    grouping : list
        all groups (given by tuples of clusters within a group)

    Returns
    -------
    clusters : list
        all clusters within that grouping

    """
    res = []
    for gr in grouping:
        if type(gr) is int:
            res.append(gr)
        else:
            for i in range(0, len(gr)):
                res.append(gr[i])
    return(res)        
    
def printing(content, prints = True, flush = True):
    """
    Function to only print progress report to console if chosen.

    Parameters
    ----------
    content : str
        Message that is to be printed.
    prints : boolean, optional
        Whether message should be printed. The default is True.
    flush : bpolean, optional
        Whether to forcibly flush the stream. The default is True.

    Returns
    -------
    None.

    """
    if prints:
        print(content, flush = flush)
    
def filename(settings, PenMet, validation, probF = 0.95, probS = 0.95, \
             rhoF = None, rhoS = None, groupSize = "", groupAim = "", \
             adjacent = False, allNames = False):
    """
    Combines all settings to a single file name to save results.

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.
    PenMet : "prob" or "penalties"
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly.
    validation : None or int
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.
    probF : float
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    probS : float
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float
        The penalty for shortcomings of the food demand (only relevant if 
        PenMet == "penalties"). The default is None.
    rhoS : float
        The penalty for insolvency (only relevant if PenMet == "penalties").
        The default is None.
    groupSize : int
        in case loading data for e.g. all groups from a specific cluster 
        grouping, this is the size of the groups (relevant for filename of
        figures)
    groupSize : int
        in case loading data for e.g. all groups from a specific cluster 
        grouping, this is the aim of the grouping (relevant for filename of
        figures)
    adjacent : boolean
        in case loading data for e.g. all groups from a specific cluster 
        grouping, this states whether clusters within a group had to be 
        adjacent (relevant for filename of figures)
        
    Returns
    -------
    fn : str
        Filename combining all settings.

    """
        
    settingsTmp = settings.copy()
    if type(settingsTmp["k_using"]) is tuple:
        settingsTmp["k_using"] = list(settingsTmp["k_using"])
    if type(settingsTmp["k_using"]) is list:
        settingsTmp["k_using"] = MakeList(settingsTmp["k_using"])
        
    for key in settingsTmp.keys():
        if type(settingsTmp[key]) is not list:
            settingsTmp[key] = [settingsTmp[key]]
        
    if type(validation) is not list:
        validationTmp = [validation]
    else:
        validationTmp = validation
    
    if PenMet == "prob":
        if type(probF) is not list:
            probFTmp = [probF]
        else:
            probFTmp = probF
        if type(probS) is not list:
            probSTmp = [probS]
        else:
            probSTmp = probS
        fn = "pF" + '_'.join(str(n) for n in probFTmp) + \
             "pS" + '_'.join(str(n) for n in probSTmp)
    else:
        rhoFTmp = rhoF.copy()
        rhoSTmp = rhoS.copy()
        if type(rhoFTmp) is not list:
            rhoFTmp = [rhoFTmp]
        if type(rhoSTmp) is not list:
            rhoSTmp = [rhoSTmp]
        fn = "rF" + '_'.join(str(n) for n in rhoFTmp) + \
             "rS" + '_'.join(str(n) for n in rhoSTmp)
     
    if groupSize != "":    
        groupSize = "GS" + str(groupSize)
   
    if adjacent:
        ad = "Adj"
    else:
        ad = ""     
        
    fn = fn + "K" + '_'.join(str(n) for n in settingsTmp["k"]) + \
        "using" +  '_'.join(str(n) for n in settingsTmp["k_using"]) + \
        groupAim + groupSize + ad + \
        "Yield" + '_'.join(str(n).capitalize() for n in settingsTmp["yield_projection"]) + \
        "Pop" + '_'.join(str(n).capitalize() for n in settingsTmp["pop_scenario"]) +  \
        "Risk" + '_'.join(str(n) for n in settingsTmp["risk"]) + \
        "N" + '_'.join(str(n) for n in settingsTmp["N"]) + \
        "M" + '_'.join(str(n) for n in validationTmp) + \
        "Tax" + '_'.join(str(n) for n in settingsTmp["tax"]) + \
        "PercIgov" + '_'.join(str(n) for n in settingsTmp["perc_guaranteed"])
    
    
    if allNames:
        # all settings that affect the calculation of rhoF
        SettingsBasics = "k" + str(settings["k"]) + \
                "using" +  '_'.join(str(n) for n in settings["k_using"]) + \
                "num_crops" + str(settings["num_crops"]) + \
                "yield_projection" + str(settings["yield_projection"]) + \
                "sim_start" + str(settings["sim_start"]) + \
                "pop_scenario" + str(settings["pop_scenario"]) +  \
                "T" + str(settings["T"])
        SettingsMaxProbF = SettingsBasics + "N" + str(settings["N"])
        SettingsAffectingRhoF = SettingsBasics + "probF" + str(probF) + \
                "N" + str(settings["N"])
        
        # all settings that affect the calculation of rhoS
        SettingsBasics = SettingsBasics + \
                "risk" + str(settings["risk"]) + \
                "tax" + str(settings["tax"]) + \
                "perc_guaranteed" + str(settings["perc_guaranteed"])
        SettingsMaxProbS = SettingsBasics + "N" + str(settings["N"])
        SettingsAffectingRhoS = SettingsBasics + "probS" + str(probS) + \
                "N" + str(settings["N"])
        return(fn, SettingsMaxProbF, SettingsAffectingRhoF, \
               SettingsMaxProbS, SettingsAffectingRhoS)
    
    return(fn)

