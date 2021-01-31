# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:05:23 2021

@author: leip
"""
import numpy as np
import pandas as pd
import os
import pickle
from termcolor import colored

from ModelCode.Auxiliary import printing

# %% ############## FUNCTIONS DEALING WITH THE RESULTS PANDA CSV ##############

def write_to_pandas(settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, crop_alloc, \
                    meta_sol, meta_sol_vss, VSS_value, validation_values, \
                    fn_fullresults, console_output, file):
    """
    Adds information on the model run to the current pandas csv.
    
    Parameters
    ----------
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
        DESCRIPTION.
    crop_alloc :  np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'S', 'exp_incomes', 'profits', 'exp_shortcomings', 
        'fd_penalty', 'avg_fd_penalty', 'sol_penalty', 'final_fund', 
        'prob_staying_solvent', 'prob_food_security', 'payouts', 
        'yearly_fixed_costs', 'num_years_with_losses')
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
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.

    Returns
    -------
    None.

    """
 
    printing("\nAdding results to pandas", console_output = console_output)
    if settings["PenMet"] == "prob":
        dict_for_pandas = {"Input probability food security": settings["probF"],
                           "Input probability solvency": settings["probS"],
                           "Number of crops": settings["num_crops"],
                           "Number of clusters": settings["k"],
                           "Used clusters": settings["k_using"],
                           "Yield projection": settings["yield_projection"],
                           "Simulation start": settings["sim_start"],
                           "Population scenario": settings["pop_scenario"],
                           "Risk level covered": settings["risk"],
                           "Tax rate": settings["tax"],
                           "Share of income that is guaranteed": settings["perc_guaranteed"],
                           "Initial fund size": settings["ini_fund"],
                           "Sample size": settings["N"],
                           "Sample size for validation": settings["validation_size"],
                           "Number of covered years": settings["T"],
                           "Average food demand": np.mean(args["demand"]),
                           "Import (given as model input)": args["import"],
                           "Necessary add. import (excluding solvency constraint, including theoretical export)": AddInfo_CalcParameters["necessary_import"],
                           "Necessary add. import (including solvency constraint, including theoretical export)": meta_sol["necessary_import"],
                           "Total necessary import when including solvency constraint": args["import"] + meta_sol["add_needed_import"],
                           "Expected income (to calculate guaranteed income)": list(AddInfo_CalcParameters["expected_incomes"]),
                           "Penalty for food shortage": args["rhoF"],
                           "Penalty for insolvency": args["rhoS"],
                           "Max. possible probability for food security (area optimized for food security)": AddInfo_CalcParameters["maxProbFareaF"],
                           "Max. possible probability for solvency (area optimized for food security)": AddInfo_CalcParameters["maxProbSareaF"],
                           "Max. possible probability for food security (area optimized for solvency)": AddInfo_CalcParameters["maxProbFareaS"],
                           "Max. possible probability for solvency (area optimized for solvency)": AddInfo_CalcParameters["maxProbSareaS"],
                           "Necessary debt (excluding food security constraint)": AddInfo_CalcParameters["necessary_debt"],
                           "Necessary debt (including food security constraint)": meta_sol["necessary_debt"],
                           "Probability for a catastrophic year": yield_information["prob_cat_year"],
                           "Share of samples with no catastrophe": yield_information["share_no_cat"],
                           "Share of years/clusters with unprofitable rice yields": yield_information["share_rice_np"],
                           "Share of years/clusters with unprofitable maize yields": yield_information["share_maize_np"],
                           "Share of West Africa's population that is living in currently considered region (2015)": \
                               population_information["pop_area_ratio2015"],
                           "On average cultivated area per cluster": list(np.nanmean(crop_alloc, axis = (0,1))),
                           "Average yearly total cultivated area": np.nanmean(np.nansum(crop_alloc, axis = (1,2))),
                           "Average food demand penalty (over years and samples)": np.nanmean(meta_sol["fd_penalty"]),
                           "Average solvency penalty (over samples)": np.mean(meta_sol["sol_penalty"]),
                           "Average cultivation costs per cluster (over years and samples)": list(np.nanmean(meta_sol["yearly_fixed_costs"], axis = (0,1))),
                           "Average total cultivation costs": np.nanmean(np.nansum(meta_sol["yearly_fixed_costs"], axis = (1,2))),
                           "Expected total costs": meta_sol["exp_tot_costs"],
                           "Average food shortcomings (over all years and samples with shortcomings)": np.nanmean(meta_sol["shortcomings"][meta_sol["shortcomings"]>0]),
                           "Number of occurrences per cluster where farmers make losses": list(meta_sol["num_years_with_losses"]),
                           "Average income per cluster in final run (over years and samples)": list(np.nanmean(meta_sol["profits"], axis = (0,1))),
                           "Average government payouts per cluster (over samples)": list(np.nanmean(np.nansum(meta_sol["payouts"], axis = 1), axis = 0)),
                           "Average final fund (over all samples with negative final fund)": np.nanmean(meta_sol["final_fund"][meta_sol["final_fund"] < 0]),
                           "Number of samples with negative final fund": np.nansum(meta_sol["final_fund"] < 0),
                           "Resulting probability for food security": meta_sol["probF"],
                           "Resulting probability for solvency": meta_sol["probS"],
                           "Resulting probability for food security for VSS": meta_sol_vss["probF"],
                           "Resulting probability for solvency for VSS": meta_sol_vss["probS"],
                           "Value of stochastic solution": VSS_value,
                           "Validation value (deviation of total penalty costs)": validation_values["deviation_penalties"],
                           "Seed (for yield generation)": settings["seed"],
                           "Filename for full results": fn_fullresults}
            
        if not os.path.exists("ModelOutput/Pandas/" + file + ".csv"):
            CreateEmptyPanda(file)
        
        current_panda = pd.read_csv("ModelOutput/Pandas/" + file + ".csv")
        current_panda = current_panda.append(dict_for_pandas, ignore_index = True)
        
        saved = False
        while saved == False:
            try:
                current_panda.to_csv("ModelOutput/Pandas/" + file + ".csv", index = False)
            except PermissionError:
                print(colored("Could not save updated panda.", "cyan"))
                print(colored("Please close the corresponding csv if currently open.", "cyan"))
                continue
            saved = True
            
    return(None)

def SetUpNewCurrentPandas(name_old_pandas):
    """
    Renames the current pandas csv according to the given name and sets up a
    new current pandas csv.

    Parameters
    ----------
    name_old_pandas : str
        filenme for the csv.

    Returns
    -------
    None.

    """
    
    # save old panda
    current_panda = pd.read_csv("ModelOutput/Pandas/current_panda.csv")
    current_panda.to_csv("ModelOutput/Pandas/" + name_old_pandas + ".csv", index = False)
    
    # create new empty panda
    os.remove("ModelOutput/Pandas/current_panda.csv")
    CreateEmptyPanda()
    
    return(None)

def CreateEmptyPanda(file = "current_panda"):
    """
    Creating a new empty pandas object with the correct columns.

    Returns
    -------
    None.

    """
    
    with open("ModelOutput/Pandas/ColumnNames.txt", "rb") as fp:
        colnames = pickle.load(fp)
    
    new_panda = pd.DataFrame(columns = colnames)
    new_panda.to_csv("ModelOutput/Pandas/" + file + ".csv", index = False)

    return(None)

def OpenPanda(file = "current_panda"):
 
    def __ConvertListsInts(arg):
        arg = arg.strip("][").split(", ")
        res = []
        for j in range(0, len(arg)):
            res.append(int(arg[j]))
        return(res)
    
    def __ConvertListsFloats(arg):
        arg = arg.strip("][").split(", ")
        res = []
        for j in range(0, len(arg)):
            res.append(float(arg[j]))
        return(res)
    
    # open panda without conversion
    panda = pd.read_csv("ModelOutput/Pandas/" + file + ".csv")
    
    # get conversion for all columns that could be in the file
    with open("ModelOutput/Pandas/ColumnTypes.txt", "rb") as fp:
        dict_convert = pickle.load(fp)
        
    # get the subset of columns available in the demanded panda file
    dict_convert = {k:dict_convert[k] for k in  panda.columns.to_list()}
        
    # substitute conversions that need local function
    for key in dict_convert.keys():
        if dict_convert[key] == "list of floats":
            dict_convert[key] = __ConvertListsFloats
        elif dict_convert[key] == "list of ints":
            dict_convert[key] = __ConvertListsInts
        
    # re-read panda with conversions
    panda = pd.read_csv("ModelOutput/Pandas/" + file + ".csv", converters = dict_convert)
    
    return(panda)
   

def OverViewCurrentPandaVariables():
    
    with open("ModelOutput/Pandas/ColumnNames.txt", "rb") as fp:
        colnames = pickle.load(fp)
            
    return(colnames)
    
def SetUpPandaDicts():
    units = {"Input probability food security": "",
        "Input probability solvency": "",
        "Number of crops": "",
        "Number of clusters": "",
        "Used clusters": "",
        "Yield projection": "",
        "Simulation start": "[Year]",
        "Population scenario": "",
        "Risk level covered": "",
        "Tax rate": "",
        "Share of income that is guaranteed": "",
        "Initial fund size": "[$10^9\,\$$]",
        "Sample size": "",
        "Sample size for validation": "",
        "Number of covered years": "",
        "Average food demand": "[$10^{12}\,$kcal]",
        "Import (given as model input)": "[$10^{12}\,$kcal]",
        "Necessary add. import (excluding solvency constraint, including theoretical export)": "[$10^{12}\,$kcal]",
        "Necessary add. import (including solvency constraint, including theoretical export)":"[$10^{12}\,$kcal]",
        "Total necessary import when including solvency constraint": "[$10^{12}\,$kcal]",
        "Expected income (to calculate guaranteed income)": "[$10^9\,\$$]",
        "Penalty for food shortage": "[$\$/10^3\,$kcal]",
        "Penalty for insolvency": "[$\$/\$$]",
        "Max. possible probability for food security (area optimized for food security)": "",
        "Max. possible probability for solvency (area optimized for food security)": "",
        "Max. possible probability for food security (area optimized for solvency)": "",
        "Max. possible probability for solvency (area optimized for solvency)": "",
        "Necessary debt (excluding food security constraint)": "[$10^9\,\$$]",
        "Necessary debt (including food security constraint)": "[$10^9\,\$$]",
        "Probability for a catastrophic year": "",
        "Share of samples with no catastrophe": "",
        "Share of years/clusters with unprofitable rice yields": "",
        "Share of years/clusters with unprofitable maize yields": "",
        "Share of West Africa's population that is living in currently considered region (2015)": "",
        "On average cultivated area per cluster": "[$10^9\,ha$]",
        "Average yearly total cultivated area": "[$10^9\,ha$]",
        "Average food demand penalty (over years and samples)": "[$10^9\,\$$]",
        "Average solvency penalty (over samples)": "[$10^9\,\$$]",
        "Average cultivation costs per cluster (over years and samples)": "[$10^9\,\$$]",
        "Average total cultivation costs": "[$10^9\,\$$]",
        "Expected total costs": "[$10^9\,\$$]",
        "Average food shortcomings (over all years and samples with shortcomings)": "[$10^{12}\,$kcal]",
        "Number of occurrences per cluster where farmers make losses": "",
        "Average income per cluster in final run (over years and samples)": "[$10^9\,\$$]",
        "Average government payouts per cluster (over samples)": "[$10^9\,\$$]",
        "Average final fund (over all samples with negative final fund)": "[$10^9\,\$$]",
        "Number of samples with negative final fund": "",
        "Resulting probability for food security": "",
        "Resulting probability for solvency": "",
        "Resulting probability for food security for VSS": "",
        "Resulting probability for solvency for VSS": "",
        "Value of stochastic solution": "[$10^9\,\$$]",
        "Validation value (deviation of total penalty costs)": "",
        "Seed (for yield generation)": "",
        "Filename for full results": ""}    
    
    convert =  {"Input probability food security": float,
         "Input probability solvency": float,
         "Number of crops": int,
         "Number of clusters": int,
         "Used clusters": "list of ints",
         "Yield projection": str,
         "Simulation start": int,
         "Population scenario": str,
         "Risk level covered": float,
         "Tax rate": float,
         "Share of income that is guaranteed": float,
         "Initial fund size": float,
         "Sample size": int,
         "Sample size for validation": int,
         "Number of covered years": int,
         "Average food demand": float,
         "Import (given as model input)": float,
         "Necessary add. import (excluding solvency constraint, including theoretical export)": float,
         "Necessary add. import (including solvency constraint, including theoretical export)": float,
         "Total necessary import when including solvency constraint": float,
         "Expected income (to calculate guaranteed income)": "list of floats",
         "Penalty for food shortage": float,
         "Penalty for insolvency": float,
         "Max. possible probability for food security (area optimized for food security)": float,
         "Max. possible probability for solvency (area optimized for food security)": float,
         "Max. possible probability for food security (area optimized for solvency)": float,
         "Max. possible probability for solvency (area optimized for solvency)": float,         
         "Necessary debt (excluding food security constraint)": float,
         "Necessary debt (including food security constraint)": float,
         "Probability for a catastrophic year": float,
         "Share of samples with no catastrophe": float,
         "Share of years/clusters with unprofitable rice yields": float,
         "Share of years/clusters with unprofitable maize yields": float,
         "Share of West Africa's population that is living in currently considered region (2015)": \
             float,
         "On average cultivated area per cluster": "list of floats",
        "Average yearly total cultivated area": float,
         "Average food demand penalty (over years and samples)": float,
         "Average solvency penalty (over samples)": float,
         "Average cultivation costs per cluster (over years and samples)": "list of floats",
         "Average total cultivation costs": float,
         "Expected total costs": float,
         "Average food shortcomings (over all years and samples with shortcomings)": float,
         "Number of occurrences per cluster where farmers make losses": "list of ints",
         "Average income per cluster in final run (over years and samples)": "list of floats",
         "Average government payouts per cluster (over samples)": "list of floats",
         "Average final fund (over all samples with negative final fund)": float,
         "Number of samples with negative final fund": int,
         "Resulting probability for food security": float,
         "Resulting probability for solvency": float,
         "Resulting probability for food security for VSS": float,
         "Resulting probability for solvency for VSS": float,
         "Value of stochastic solution": float,
         "Validation value (deviation of total penalty costs)": float,
         "Seed (for yield generation)": int,
         "Filename for full results": str}
        
    colnames = ['Input probability food security', 
        'Input probability solvency', 
        'Number of crops',
        'Number of clusters', 
        'Used clusters', 
        'Yield projection', 
        'Simulation start', 
        'Population scenario', 
        'Risk level covered', 
        'Tax rate', 
        'Share of income that is guaranteed', 
        'Initial fund size', 
        'Sample size', 
        'Sample size for validation', 
        'Number of covered years', 
        'Average food demand', 
        'Import (given as model input)', 
        'Necessary add. import (excluding solvency constraint, including theoretical export)', 
        'Necessary add. import (including solvency constraint, including theoretical export)', 
        'Total necessary import when including solvency constraint', 
        'Expected income (to calculate guaranteed income)', 
        'Penalty for food shortage', 
        'Penalty for insolvency', 
        'Max. possible probability for food security (area optimized for food security)', 
        'Max. possible probability for solvency (area optimized for food security)', 
        'Max. possible probability for food security (area optimized for solvency)', 
        'Max. possible probability for solvency (area optimized for solvency)',
        'Necessary debt (excluding food security constraint)', 
        'Necessary debt (including food security constraint)', 
        'Probability for a catastrophic year', 
        'Share of samples with no catastrophe', 
        'Share of years/clusters with unprofitable rice yields', 
        'Share of years/clusters with unprofitable maize yields', 
        'Share of West Africa\'s population that is living in currently considered region (2015)', 
        'On average cultivated area per cluster', 
        'Average yearly total cultivated area',
        'Average food demand penalty (over years and samples)', 
        'Average solvency penalty (over samples)', 
        'Average cultivation costs per cluster (over years and samples)', 
        'Average total cultivation costs',
        'Expected total costs', 
        'Average food shortcomings (over all years and samples with shortcomings)', 
        'Number of occurrences per cluster where farmers make losses', 
        'Average income per cluster in final run (over years and samples)', 
        'Average government payouts per cluster (over samples)', 
        'Average final fund (over all samples with negative final fund)',
        'Number of samples with negative final fund',
        'Resulting probability for food security', 
        'Resulting probability for solvency', 
        'Resulting probability for food security for VSS', 
        'Resulting probability for solvency for VSS', 
        'Value of stochastic solution', 
        'Validation value (deviation of total penalty costs)',
        'Seed (for yield generation)',
        'Filename for full results']
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "wb") as fp:
        pickle.dump(units, fp)
        
    with open("ModelOutput/Pandas/ColumnNames.txt", "wb") as fp:
        pickle.dump(colnames, fp)
        
    with open("ModelOutput/Pandas/ColumnTypes.txt", "wb") as fp:
        pickle.dump(convert, fp)
        
    return(None)

