#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 09:30:39 2021

@author: Debbora Leip
"""

import os
import pickle
import time as tm 
import numpy as np
from datetime import datetime
from termcolor import colored

from ModelCode.SetFolderStructure import CheckFolderStructure
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import filename
from ModelCode.PandaFunctions import write_to_pandas
from ModelCode.PlottingModelOutput import PlotModelOutput
from ModelCode.Auxiliary import printing
from ModelCode.MetaInformation import GetMetaInformation
from ModelCode.SettingsParameters import SetParameters
from ModelCode.ExpectedIncome import GetExpectedIncome
from ModelCode.GetPenalties import GetPenalties
from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy
from ModelCode.VSSandValidation import VSS
from ModelCode.VSSandValidation import OutOfSampleVal

# %% ############## WRAPPING FUNCTIONS FOR FOOD SECURITY MODEL ################

def FoodSecurityProblem(console_output = None, logs_on = None, \
                        save = True, plotTitle = None, \
                        **kwargs):
    """
        
    Setting up and solving the food security problem. Returns model output
    and additional information on the solution, as well as the VSS and a 
    validation of the model output. The results are also saved. If the model
    has already been solved for the exact settings, results are loaded instead
    of recalculated.
    

    Parameters
    ----------
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        If None, the default as defined in ModelCode/GeneralSettings is used.
    save : boolean, optional
        whether the results should be saved. The default is True.
    plotTitle : str or None
        If not None, a plot of the resulting crop allocations will be made 
        with that title and saved to Figures/CropAllocs.
    **kwargs
        settings for the model, passed to DefaultSettingsExcept()
        
    Returns
    -------
    settings : dict
        The model input settings that were given by user. 
    args : dict
        Dictionary of arguments needed as direct model input.
    AddInfo_CalcParameters : dict
        Additional information from calculatings expected income and penalties
        which are not needed as model input.
    yield_information : dict
        Information on the yield distributions for the considered clusters.
    population_information : dict
        Information on the population in the considered area.
    status : int
        status of solver (optimal: 2)
    all_durations :  dict
        Information on the duration of different steps of the model framework.
    crop_alloc :  np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'S', 'exp_incomes', 'profits', 'exp_shortcomings', 
        'fd_penalty', 'avg_fd_penalty', 'sol_penalty', 'final_fund', 
        'prob_staying_solvent', 'prob_food_security', 'payouts', 
        'yearly_fixed_costs', 'num_years_with_losses')
    crop_alloc_vss : np.array
        deterministic solution for optimal crop areas    
    meta_sol_vss : dict
        additional information on the deterministic solution  
    VSS_value : float
        VSS calculated as the difference between total costs using 
        deterministic solution for crop allocation and stochastic solution
        for crop allocation       
    validation_values : dict
        total costs and penalties for the model result and a higher sample 
        size for validation ("sample_size", "total_costs", "total_costs_val", 
        "fd_penalty", "fd_penalty_val", "sol_penalty", "sol_penalty_val", 
        "total_penalties", "total_penalties_val", "deviation_penalties")
    fn : str
        all settings combined to a single file name to save/load results
    """    
    
    # set up folder structure (if not already done)
    CheckFolderStructure()
        
    # defining settings
    settings = DefaultSettingsExcept(**kwargs)
    
    # get filename of model results
    fn = filename(settings)
    
    # if model output does not exist yet it is calculated
    if not os.path.isfile("ModelOutput/SavedRuns/" + fn + ".txt"):
        try:
           settings, args, AddInfo_CalcParameters, yield_information, \
           population_information, status, all_durations, crop_alloc, meta_sol, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values = \
                                OptimizeModel(settings,
                                              console_output = console_output,
                                              save = save,
                                              logs_on = logs_on,
                                              plotTitle = plotTitle,)
        except KeyboardInterrupt:
            print(colored("\nThe model run was interupted by the user.", "red"), flush = True)
            if logs_on:
                log = open("ModelLogs/tmp.txt", "a")
                log.write("\n\nThe model run was interupted by the user.")
                log.close()
                os.rename("ModelLogs/tmp.txt", "ModelLogs/" + fn + ".txt")
        
    # if it does, it is loaded
    else:            
        printing("Loading results", console_output = console_output, logs_on = False)
        with open("ModelOutput/SavedRuns/" + fn + ".txt", "rb") as fp:
            pickle.load(fp) # info
            settings = pickle.load(fp)
            args = pickle.load(fp)
            AddInfo_CalcParameters = pickle.load(fp)
            yield_information = pickle.load(fp)
            population_information = pickle.load(fp)
            status = pickle.load(fp)
            all_durations = pickle.load(fp)
            crop_alloc = pickle.load(fp)
            crop_alloc_vss = pickle.load(fp)
            VSS_value = pickle.load(fp)
            validation_values = pickle.load(fp)
                
        meta_sol = GetMetaInformation(crop_alloc, args, args["rhoF"], args["rhoS"])
        meta_sol_vss =  GetMetaInformation(crop_alloc_vss, args, args["rhoF"], args["rhoS"])
        
    # if a plottitle is provided, crop allocations over time are plotted
    if plotTitle is not None:
        PlotModelOutput(PlotType = "CropAlloc", title = plotTitle, \
                    file = fn, crop_alloc = crop_alloc, k = settings["k"], \
                    k_using = settings["k_using"], max_areas = args["max_areas"])
    
    return(settings, args, AddInfo_CalcParameters, yield_information, \
           population_information, status, all_durations, crop_alloc, meta_sol, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn)          

def OptimizeModel(settings, console_output = None, logs_on = None, \
                  save = True, plotTitle = None):
    """
    Function combines setting up and solving the model, calculating additional
    information, and saving the results.

    Parameters
    ----------
    settings : dict
        All input settings for the model framework.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        If None, the default as defined in ModelCode/GeneralSettings is used.
    save : boolean, optional
        whether the results should be saved. The default is True.
    plotTitle : str or None
        If not None, a plot of the resulting crop allocations will be made 
        with that title and saved to Figures/CropAllocs.

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
    all_durations = {}
    
    # initialize log file
    if logs_on is None:
        from ModelCode.GeneralSettings import logs_on
        
    if logs_on:
        if os.path.exists("ModelLogs/tmp.txt"):
            os.remove("ModelLogs/tmp.txt")
        log = open("ModelLogs/tmp.txt", "a")
        log.write("Model started " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n")
        log.write("\nModel Settings: ")
        for key in settings.keys():
            log.write(key + " = " + str(settings[key]) + "\n                ")
        log.write("save = " + str(save) + "\n                ")
        log.write("plotTitle = " + str(plotTitle))
        log.close()
    
    # get parameters for the given settings
    ex_income_start  = tm.time()
    exp_incomes = GetExpectedIncome(settings, console_output = console_output)
    AddInfo_CalcParameters = {"expected_incomes": exp_incomes}
    ex_income_end  = tm.time()
    all_durations["GetExpectedIncome"] = ex_income_end - ex_income_start
    printing("\nGetting parameters", console_output = console_output)
    args, yield_information, population_information = \
                    SetParameters(settings, AddInfo_CalcParameters)
    
    # get the right penalties
    penalties_start  = tm.time()
    if settings["PenMet"] == "prob":
        rhoF, rhoS, necessary_debt, needed_import = GetPenalties(settings, args, \
                                        yield_information, console_output = console_output)
        args["rhoF"] = rhoF
        args["rhoS"] = rhoS
        
        AddInfo_CalcParameters["necessary_debt"] = necessary_debt
        AddInfo_CalcParameters["import"] = needed_import
        if needed_import > 0:
            args["import"] = needed_import
    else:
        args["rhoF"] = settings["rhoF"]
        args["rhoS"] = settings["rhoS"]
        
        AddInfo_CalcParameters["necessary_debt"] = None
        AddInfo_CalcParameters["import"] = None
    penalties_end  = tm.time()
    all_durations["GetPenalties"] = penalties_end - penalties_start
    
        
    # run the optimizer
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, \
                                           console_output = console_output, \
                                           logs_on = True)
    all_durations["MainModelRun"] = durations[2]
        
    printing("\nResulting probabilities:\n" + \
    "     probF: " + str(np.round(meta_sol["probF"]*100, 2)) + "%\n" + \
    "     probS: " + str(np.round(meta_sol["probS"]*100, 2)) + "%", console_output)
        
    # VSS
    vss_start  = tm.time()
    printing("\nCalculating VSS", console_output = console_output)
    crop_alloc_vss, meta_sol_vss = VSS(settings, AddInfo_CalcParameters, args)
    VSS_value = meta_sol_vss["exp_tot_costs"] - meta_sol["exp_tot_costs"]
    vss_end  = tm.time()
    all_durations["VSS"] = vss_end - vss_start
    
    # out of sample validation
    validation_start  = tm.time()
    if settings["validation_size"] is not None:
        printing("\nOut of sample validation", console_output = console_output)
        validation_values = OutOfSampleVal(crop_alloc, settings, AddInfo_CalcParameters, args["rhoF"], \
                              args["rhoS"], meta_sol, args["probS"], console_output)
    validation_end  = tm.time()
    all_durations["Validation"] = validation_end - validation_start

    # add results to pandas overview
    write_to_pandas(settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, crop_alloc, \
                    meta_sol, meta_sol_vss, VSS_value, validation_values, \
                    console_output)     
            
    # timing
    all_end  = tm.time()   
    full_time = all_end - all_start
    printing("\nTotal time: " + str(np.round(full_time, 2)) + "s", console_output = console_output)
    all_durations["TotalTime"] = full_time
       
    
    # saving results
    if save:
        fn = filename(settings)
        
        info = ["settings", "args", "yield_information", "population_information", \
                "status", "durations", "crop_alloc", "crop_alloc_vss", \
                "VSS_value", "validation_values"]
            
        with open("ModelOutput/SavedRuns/" + fn + ".txt", "wb") as fp:
            pickle.dump(info, fp)
            pickle.dump(settings, fp)
            pickle.dump(args, fp)
            pickle.dump(AddInfo_CalcParameters, fp)
            pickle.dump(yield_information, fp)
            pickle.dump(population_information, fp)
            pickle.dump(status, fp)
            pickle.dump(all_durations, fp)
            pickle.dump(crop_alloc, fp)
            pickle.dump(crop_alloc_vss, fp)
            pickle.dump(VSS_value, fp)
            pickle.dump(validation_values, fp)
     
    # rename the temporal log file
    if logs_on:
        if os.path.exists("ModelLogs/" + fn  + ".txt"):
            i = 1
            while os.path.exists("ModelLogs/" + fn  + "_" + str(i) + ".txt"):
                i += 1
            fn = fn + "_" + str(i)
        os.rename("ModelLogs/tmp.txt", "ModelLogs/" + fn + ".txt") 
        
    return(settings, args, AddInfo_CalcParameters, yield_information, \
           population_information, status, all_durations, crop_alloc, meta_sol, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values)          
