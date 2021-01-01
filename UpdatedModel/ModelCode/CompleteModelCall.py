#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 09:30:39 2021

@author: Debbora Leip
"""

import os
import pickle
import sys
import time as tm 
import numpy as np
from datetime import datetime

from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import filename
from ModelCode.PlottingModelOutput import PlotModelOutput
from ModelCode.Auxiliary import printing
from ModelCode.MetaInformation import GetMetaInformation
from ModelCode.SettingsParameters import SetParameters
from ModelCode.GuaranteedIncome import GetExpectedIncome
from ModelCode.GetPenalties import GetPenalties
from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy
from ModelCode.VSSandValidation import VSS
from ModelCode.VSSandValidation import OutOfSampleVal


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
    yield_information : dict
        Dictionary with information on the yield samples
    population_information : dict
        Dictionary with information on the population
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
    fn : str
        all settings combined to a single file name to save/load results
    """    
    
    # defining settings
    settings = DefaultSettingsExcept(**kwargs)
    
    # get filename of model results
    fn = filename(settings, PenMet, validation, probF, probS, rhoF, rhoS)
    
    # check modus (probabilities or penalties given?)
    if PenMet == "penalties":
        probS = None
        probF = None
    elif PenMet != "prob":
        sys.exit("A non-valid penalty method was chosen (PenMet must " + \
                 "be either \"prob\" or \"penalties\").")
    
    # if model output already exists, it is loaded
    if not os.path.isfile("ModelOutput/SavedRuns/" + fn + ".txt"):
        crop_alloc, meta_sol, status, durations, settings, args, \
        yield_information, population_information, rhoF, rhoS, VSS_value, \
        crop_alloc_vss, meta_sol_vss, validation_values = \
                            OptimizeModel(PenMet = PenMet,  
                                          probF = probF, 
                                          probS = probS, 
                                          rhoFini = rhoF,
                                          rhoSini = rhoS,
                                          prints = prints,
                                          validation = validation,
                                          save = save,
                                          **kwargs)
        
    # if not, it is calculated
    else:            
        printing("Loading results", prints = prints)
        with open("ModelOutput/SavedRuns/" + fn + ".txt", "rb") as fp:
            pickle.load(fp) # info
            crop_alloc = pickle.load(fp)
            settings = pickle.load(fp)
            args = pickle.load(fp)
            yield_information = pickle.load(fp)
            population_information = pickle.load(fp)
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
        
    # if a plottitle is provided, crop allocations over time are plotted
    if plotTitle is not None:
        PlotModelOutput(PlotType = "CropAlloc", title = plotTitle, \
                    file = fn, crop_alloc = crop_alloc, k = settings["k"], \
                    k_using = settings["k_using"], max_areas = args["max_areas"])
    
    return(crop_alloc, meta_sol, status, durations, settings, args, \
           yield_information, population_information, rhoF, rhoS, \
           VSS_value, crop_alloc_vss, meta_sol_vss, validation_values, fn)          

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
    rhoFini : float or None, optional
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoFini is None, a initial guess for rhoFini 
        will be calculated in GetPenalties, else this will be used as initial 
        guess for the penalty which will give the correct probability for 
        reaching food demand. The default is None.
    rhoSini : float or None, optional
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoSini is None, a initial guess for rhoSini 
        will be calculated in GetPenalties, else this will be used as initial
        guess for the penalty which will give the correct probability for 
        solvency. The default is None.
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
    yield_information : dict
        Dictionary with information on the yield samples
    population_information : dict
        Dictionary with information on the population      
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

    """
    # timing
    all_start  = tm.time()
    
    # TODO include decision for logfile in all printing statements so console
    # initialize log file
    if os.path.exists("ModelLogs/tmp.txt"):
        os.remove("ModelLogs/tmp.txt")
    log = open("ModelLogs/tmp.txt", "a")
    log.write("Model started " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")))
    log.write("\n\nModel Input: PenMet = " + str(PenMet) + 
              "\n             probF = " + str(probF) + 
              "\n             probS = " + str(probS) + 
              "\n             rhoF = " + str(rhoFini) + 
              "\n             rhoS = " + str(rhoSini) + 
              "\n             validation = " + str(validation) + 
              "\n             save = " + str(save))
    for key in kwargs.keys():
        log.write("\n             " + key + " = " + str(kwargs[key]))
    log.close()
        
    # create dictionary of all settings (includes calculating or loading the
    # correct expected income)
    printing("\nDefining Settings", prints = prints)
    settings = DefaultSettingsExcept(**kwargs)
    
    # get parameters for the given settings
    if settings["expected_incomes"] is None:  
         settings["expected_incomes"] = GetExpectedIncome(settings, prints = False)
    printing("\nGetting parameters", prints = prints)
    args, yield_information, population_information = SetParameters(settings)
    
    # get the right penalties
    if PenMet == "prob":
        rhoF, rhoS, necessary_debt, needed_import = GetPenalties(settings, args, \
                                        yield_information, probF, probS, \
                                        rhoFini, rhoSini, prints = prints)
        settings["import"] = needed_import
        if needed_import > 0:
            args["import"] = needed_import
        settings["neccessary_debt"] = necessary_debt
    else:
        rhoF = rhoFini
        rhoS = rhoSini
    
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
    if validation is not None:
        printing("\nOut of sample validation", prints = prints)
        validation_values = OutOfSampleVal(crop_alloc, settings, rhoF, \
                              rhoS, validation, meta_sol, probS, prints)
            
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
            pickle.dump(yield_information, fp)
            pickle.dump(population_information, fp)
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
     
    # rename the temporal log file
    os.rename("ModelLogs/tmp.txt", "ModelLogs/" + fn + ".txt")    
     
    # timing
    all_end  = tm.time()   
    full_time = all_end - all_start
    printing("\nTotal time: " + str(np.round(full_time, 2)) + "s", prints = prints)
            
    return(crop_alloc, meta_sol, status, durations, settings, args, yield_information, \
           population_information, rhoF, rhoS, VSS_value, crop_alloc_vss, meta_sol_vss, \
           validation_values)          
