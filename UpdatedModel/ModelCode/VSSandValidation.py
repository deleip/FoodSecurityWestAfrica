#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:06:20 2021

@author: Debbora Leip
"""
import numpy as np
import pickle

from ModelCode.SettingsParameters import SetParameters
from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy
from ModelCode.MetaInformation import GetMetaInformation
from ModelCode.Auxiliary import printing

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
    args_vss, yield_information, population_information = SetParameters(settings, VSS = True, console_output = False, logs_on = False)
    
    # solve model for the expected yields
    status, crop_alloc_vss, meta_sol, prob, durations = \
                SolveReducedcLinearProblemGurobiPy(args_vss, rhoF, rhoS, probS, console_output = False, logs_on = False)
                
    # get information of using VSS solution in stochastic setting
    meta_sol_vss = GetMetaInformation(crop_alloc_vss, args, rhoF, rhoS, probS)
    return(crop_alloc_vss, meta_sol_vss)      
    

# %% ################### OUT OF SAMLE VALIDATION OF RESULT ####################  

def OutOfSampleVal(crop_alloc, settings, rhoF, rhoS, \
                   M, meta_sol, probS = None, console_output = None):
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
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.

    Yields
    ------
    meta_sol_vss : dict
        Additional information on the outcome for the higher sample size.

    """
    
    # printing("      " + "\u005F" * 21, console_output = console_output)
    # higher sample size
    settings_val = settings.copy()
    settings_val["N"] = M
    # get yield samples
    printing("     Getting parameters and yield samples", console_output = console_output)
    args, yield_information, population_information = SetParameters(settings_val, \
                                            console_output = False, logs_on = False)
    
    # run objective function for higher sample size
    printing("     Objective function", console_output = console_output)
    meta_sol_val = GetMetaInformation(crop_alloc, args, rhoF, rhoS, probS)
    
    # create dictionary with validation information
    validation_values = {"sample_size": M,
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
                "_M" + str(M)
                
    if val_name in val_dict.keys():
        tmp = val_dict[val_name]
        tmp.append(validation_values["deviation_penalties"])
        val_dict[val_name] = tmp  
    else:
        val_dict[val_name] = [validation_values["deviation_penalties"]]
    with open("ModelOutput/validation.txt", "wb") as fp:    
         pickle.dump(val_dict, fp)            
    
    return(validation_values)     