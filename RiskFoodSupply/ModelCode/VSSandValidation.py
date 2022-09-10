#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:06:20 2021

@author: Debbora Leip
"""
import numpy as np
import pickle

from ModelCode.SettingsParameters import SetParameters
from ModelCode.ModelCore import SolveReducedLinearProblemGurobiPy
from ModelCode.MetaInformation import GetMetaInformation
from ModelCode.Auxiliary import _printing

# %% #################### VALUE OF THE STOCHASTIC SOLUTION ####################  
 
def VSS(settings, args):
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

    Returns
    -------
    crop_alloc_vss : np.array
        Optimal crop areas for all years, crops, clusters in the deterministic
        setting.
    expected_incomes_vss : np.array of size (len(k_using),)
        The expected income of farmers in a scenario where the government is
        not involved in VSS case.
    meta_sol_vss : dict
        Dictionary of additional information on the solution of the
        deterministic model.

    """
    # 1. expected income in deterministic case:
        
    # change some settings: we are interested in the expected income in 2016
    # (no need to change start year, as we set scenarios to fixed)
    settings_ExpIn = settings.copy()
    settings_ExpIn["yield_projection"] = "fixed"
    settings_ExpIn["pop_scenario"] = "fixed"
    settings_ExpIn["T"] = 1
    
    # get arguments to calculate deterministic solution (in particular the 
    # expected yields instead of yield samples)
    args_vss_ExpIn, yield_information_ExpIn, population_information_ExpIn = \
        SetParameters(settings_ExpIn, expected_incomes = None, VSS = True, 
                      console_output = False, logs_on = False)
    
    # solve model for the expected yields
    status, crop_alloc_vss_ExpIn, meta_sol_vss_ExpIn, prob, durations = \
                SolveReducedLinearProblemGurobiPy(args_vss_ExpIn, 
                                      args["rhoF"], args["rhoS"], 
                                      console_output = False, logs_on = False)
                
    # get information of using VSS solution in stochastic setting
    meta_sol_vss_ExpIn = GetMetaInformation(crop_alloc_vss_ExpIn, 
                               args_vss_ExpIn, args["rhoF"], args["rhoS"])
    expected_incomes_vss = meta_sol_vss_ExpIn["avg_profits_preTax"].flatten()
        
    
    # 2. crop allocation and meta information for deterministic social planner 
    
    # get arguments to calculate deterministic solution (in particular the 
    # expected yields instead of yield samples)
    args_vss, yield_information, population_information = \
        SetParameters(settings, expected_incomes_vss, VSS = True, 
                      console_output = False, logs_on = False)
    
    # solve model for the expected yields
    status, crop_alloc_vss, meta_sol, prob, durations = \
                SolveReducedLinearProblemGurobiPy(args_vss, args["rhoF"],
                      args["rhoS"], console_output = False, logs_on = False)
                
    # get information of using VSS solution in stochastic setting (but with 
    # guaranteed income of deterministic setting)
    args_VSS_sto = args.copy()
    args_VSS_sto["guaranteed_income"] = args_vss["guaranteed_income"]
    meta_sol_vss = GetMetaInformation(crop_alloc_vss, args_VSS_sto, 
                                      args["rhoF"], args["rhoS"])
    return(crop_alloc_vss, expected_incomes_vss, meta_sol_vss)      
    

# %% ################### OUT OF SAMLE VALIDATION OF RESULT ####################  

def OutOfSampleVal(crop_alloc, settings, expected_incomes, rhoF, rhoS, \
                   meta_sol, console_output = None, logs_on = None):
    """
    For validation, the objective function is re-evaluate, using the optimal
    crop allocation but a higher sample size.    
    
    Parameters
    ----------
    crop_alloc : np.array
        Optimal crop areas for all years, crops, clusters.
    settings : dict
        the model settings that were used     
    expected_incomes : np.array of size (len(k_using),)
        The expected income of farmers in a scenario where the government is
        not involved.
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.   
    meta_sol : dict 
        additional information about the model output 
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.

    Returns
    ------
    validation_values : dict
        Dictionary of validation values
        
        - sample_size : Validation sample size
        - total_costs : expected total costs in model run
        - total_costs_val : expected total costs in validation run
        - fd_penalty : average total food demand penalty in model run
        - fd_penalty_val : average total food demand penalty in validation run
        - sol_penalty : average total solvency penalty in model run
        - sol_penalty_val : average total solvency penalty in validation run
        - total_penalties : fd_penalty + sol_penalty
        - total_penalties_val : fd_penalty_val + sol_penalty_val
        - deviation_penalties : 1 - (total_penalties / total_penalties_val)

    """
    
    # higher sample size
    settings_val = settings.copy()
    settings_val["N"] = settings["validation_size"]
    
    # get yield samples
    _printing("     Getting parameters and yield samples", console_output = console_output, logs_on = logs_on)
    args, yield_information, population_information = \
                SetParameters(settings_val, expected_incomes, \
                              console_output = False, logs_on = False)
    
    # run objective function for higher sample size
    _printing("     Objective function", console_output = console_output, logs_on = logs_on)
    meta_sol_val = GetMetaInformation(crop_alloc, args, rhoF, rhoS)
    
    # create dictionary with validation information
    validation_values = {"sample_size": settings["validation_size"],
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
        
    # add this dictionary to validation file
    with open("ModelOutput/validation.txt", "rb") as fp:    
        val_dict = pickle.load(fp)
        
    val_name = str(len(settings["k_using"])) + "of" + \
                str(settings["k"]) + "_N" + str(settings["N"]) + \
                "_M" + str(settings["validation_size"])
                
    if val_name in val_dict.keys():
        tmp = val_dict[val_name]
        tmp.append(validation_values["deviation_penalties"])
        val_dict[val_name] = tmp  
    else:
        val_dict[val_name] = [validation_values["deviation_penalties"]]
    with open("ModelOutput/validation.txt", "wb") as fp:    
         pickle.dump(val_dict, fp)            
    
    return(validation_values)     