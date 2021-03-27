# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:05:23 2021

@author: Debbora Leip
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
    crop_alloc :  np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'yearly_fixed_costs', 'fd_penalty', 'avg_fd_penalty', 
        'sol_penalty', 'shortcomings', 'exp_shortcomings', 'expected_incomes', 
        'profits', 'num_years_with_losses', 'payouts', 'final_fund', 'probF', 
        'probS', 'necessary_import', 'necessary_debt')
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
    fn_fullresults : str
        Filename used to save the full model results.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.
    file : str
        filename of panda csv to which the information is to be added.
    Returns
    -------
    None.

    """
    
    pop_per_cluster = np.outer(population_information["total_pop_scen"], population_information["pop_cluster_ratio2015"])
    pop_of_area = population_information["population"]
    food_shortage_capita = meta_sol["shortcomings"]/pop_of_area
    
    wh_neg_fund = np.where(meta_sol["final_fund"] < 0)[0]
    ff_debt = -meta_sol["final_fund"][wh_neg_fund]
    ter_years = args["terminal_years"][wh_neg_fund].astype(int)
    total_guar_inc = np.sum(args["guaranteed_income"], axis = 1)
    
    ff_debt_all = -meta_sol["final_fund"]
    ff_debt_all[ff_debt_all < 0] = 0
    ter_years_all = args["terminal_years"].astype(int)
    
    shortcomings_capita = meta_sol["shortcomings"]/(pop_of_area/1e9)
    
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
                           "Average necessary add. import excluding solvency constraint (over samples and then years)": AddInfo_CalcParameters["import_onlyF"],
                           "Average necessary add. import (over samples and then years)": meta_sol["avg_nec_import"],
                           "Average total necessary import (over samples and then years)": args["import"] + meta_sol["avg_nec_import"],
                           "Average necessary add. import per capita (over samples and then years)": np.nanmean(food_shortage_capita[food_shortage_capita>0])*1e9,
                           "Expected income (to calculate guaranteed income)": list(AddInfo_CalcParameters["expected_incomes"]),
                           "Penalty for food shortage": args["rhoF"],
                           "Penalty for insolvency": args["rhoS"],
                           "Max. possible probability for food security (excluding solvency constraint)": AddInfo_CalcParameters["probF_onlyF"],
                           "Max. possible probability for solvency (excluding food security constraint)": AddInfo_CalcParameters["probS_onlyS"],
                           "Average necessary debt (excluding food security constraint)": AddInfo_CalcParameters["debt_onlyS"],
                           "Average necessary debt": meta_sol["avg_nec_debt"],
                           "Average necessary debt (over all samples with negative final fund)": np.nanmean(ff_debt),
                           "Averge necessary debt per capita (over all samples with negative final fund)": np.nanmean(ff_debt / (pop_of_area[ter_years]/1e9)),
                           "Averge necessary debt per capita (over all samples)": np.nanmean(ff_debt_all / (pop_of_area[ter_years_all]/1e9)),
                           "Average final fund (over all samples)": np.nanmean(meta_sol["final_fund"]),
                           "Number of samples with negative final fund": np.nansum(meta_sol["final_fund"] < 0),
                           "Probability for a catastrophic year": yield_information["prob_cat_year"],
                           "Share of samples with no catastrophe": yield_information["share_no_cat"],
                           "Share of years/clusters with unprofitable rice yields": yield_information["share_rice_np"],
                           "Share of years/clusters with unprofitable maize yields": yield_information["share_maize_np"],
                           "Share of West Africa's population that is living in total considered region (2015)": \
                               np.sum(population_information["pop_cluster_ratio2015"]),
                           "Share of West Africa's population that is living in considered clusters (2015)": \
                               list(population_information["pop_cluster_ratio2015"]),
                           "On average cultivated area per cluster": list(np.nanmean(crop_alloc, axis = (0,1))),
                           "Average yearly total cultivated area": np.nanmean(np.nansum(crop_alloc, axis = (1,2))),
                           "Average food demand penalty (over samples and then years)": np.nanmean(np.nanmean(meta_sol["fd_penalty"], axis = 0)),
                           "Average solvency penalty (over samples)": np.mean(meta_sol["sol_penalty"]),
                           "Average cultivation costs per cluster (over samples and then years)": list(np.nanmean(np.nanmean(meta_sol["yearly_fixed_costs"], axis = 0), axis = 0)),
                           "Average total cultivation costs": np.nanmean(np.nansum(meta_sol["yearly_fixed_costs"], axis = (1,2))),
                           "Expected total costs": meta_sol["exp_tot_costs"],
                           "Number of occurrences per cluster where farmers make losses": list(meta_sol["num_years_with_losses"]),
         # name                  "Average income per cluster in final run (over samples and then years)": list(np.nanmean(np.nanmean(meta_sol["profits"], axis = 0), axis = 0))),
         # new                  "Average income per cluster in final run scaled with capita (over samples and then years)": list(np.nanmean(np.nanmean(meta_sol["profits"], axis = 0)/(pop_per_cluster/1e9), axis = 0)),
                           "Average government payouts per cluster (over samples)": list(np.nanmean(np.nansum(meta_sol["payouts"], axis = 1), axis = 0)),
                           "Resulting probability for food security": meta_sol["probF"],
                           "Resulting probability for solvency": meta_sol["probS"],
                           "Resulting probability for food security for VSS": meta_sol_vss["probF"],
                           "Resulting probability for solvency for VSS": meta_sol_vss["probS"],
                           "Value of stochastic solution": VSS_value,
                           "VSS as share of total costs (sto. solution)": VSS_value/meta_sol["exp_tot_costs"],
                           "VSS as share of total costs (det. solution)": VSS_value/meta_sol_vss["exp_tot_costs"],
                           "Validation value (deviation of total penalty costs)": validation_values["deviation_penalties"],
                           "Seed (for yield generation)": settings["seed"],
                           "Filename for full results": fn_fullresults}
            
        if np.isnan(dict_for_pandas["Average food shortcomings (over all years and samples with shortcomings)"]):
            dict_for_pandas["Average food shortcomings (over all years and samples with shortcomings)"] = 0
        if np.isnan(dict_for_pandas["Average food shortcomings per capita (over all years and samples with shortcomings)"]):
            dict_for_pandas["Average food shortcomings per capita (over all years and samples with shortcomings)"] = 0
            
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

    Parameters
    ----------
    file : str, optional
        Name for file where to save the new panda csv.
        The default is "current_panda".

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
    """
    Load the panda csv into python.

    Parameters
    ----------
    file : str, optional
        Name of file with the csv that is to be loaded.
        The default is "current_panda".

    Returns
    -------
    panda : panda dataframe
        The panda object with information on model runs.

    """
 
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
    """
    Returns a list with the current panda variables

    Returns
    -------
    colnames : list
        All variables that are currently available in the panda csvs.

    """
    
    with open("ModelOutput/Pandas/ColumnNames.txt", "rb") as fp:
        colnames = pickle.load(fp)
            
    return(colnames)
    
def SetUpPandaDicts():
    """
    Sets up the dicts that are needed to deal with the panda csvs.
    Dictonaries are defined here and saved so they can be loaded from other
    functions. If additional variables are included in the write_to_pandas
    function, they need to be added to all three dictionaries here as well!!

    Returns
    ------
    None.

    """
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
        "Average necessary add. import excluding solvency constraint (over samples and then years)": "[$10^{12}\,$kcal]",
        "Average necessary add. import (over samples and then years)":"[$10^{12}\,$kcal]",
        "Average total necessary import (over samples and then years)": "[$10^{12}\,$kcal]",
        "Average necessary add. import per capita (over samples and then years)": "[$10^{3}\,$kcal]",
        "Expected income (to calculate guaranteed income)": "[$10^9\,\$$]",
        "Penalty for food shortage": "[$\$/10^3\,$kcal]",
        "Penalty for insolvency": "[$\$/\$$]",
        "Max. possible probability for food security (excluding solvency constraint)": "",
        "Max. possible probability for solvency (excluding food security constraint)": "",
        "Average necessary debt (excluding food security constraint)": "[$10^9\,\$$]",
        "Average necessary debt": "[$10^9\,\$$]",
        "Average necessary debt (over all samples with negative final fund)": "[$10^9\,\$$]",
        "Averge necessary debt per capita (over all samples with negative final fund)": "[$\$$]",
        "Averge necessary debt per capita (over all samples)": "[$\$$]",
        "Average final fund (over all samples)": "[$10^9\,\$$]",
        "Number of samples with negative final fund": "",
        "Probability for a catastrophic year": "",
        "Share of samples with no catastrophe": "",
        "Share of years/clusters with unprofitable rice yields": "",
        "Share of years/clusters with unprofitable maize yields": "",
        "Share of West Africa's population that is living in total considered region (2015)": "",
        "Share of West Africa's population that is living in considered clusters (2015)": "",
        "On average cultivated area per cluster": "[$10^9\,ha$]",
        "Average yearly total cultivated area": "[$10^9\,ha$]",
        "Average food demand penalty (over samples and then years)": "[$10^9\,\$$]",
        "Average solvency penalty (over samples)": "[$10^9\,\$$]",
        "Average cultivation costs per cluster (over samples and then years)": "[$10^9\,\$$]",
        "Average total cultivation costs": "[$10^9\,\$$]",
        "Expected total costs": "[$10^9\,\$$]",
        "Number of occurrences per cluster where farmers make losses": "",
        "Average income per cluster in final run (over samples and then years)": "[$10^9\,\$$]",
        "Average income per cluster in final run scaled with capita (over samples and then years)": "[$\$$]",
        "Average government payouts per cluster (over samples)": "[$10^9\,\$$]",
        "Resulting probability for food security": "",
        "Resulting probability for solvency": "",
        "Resulting probability for food security for VSS": "",
        "Resulting probability for solvency for VSS": "",
        "Value of stochastic solution": "[$10^9\,\$$]",
        "VSS as share of total costs (sto. solution)": "",
        "VSS as share of total costs (det. solution)": "",
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
         "Average necessary add. import excluding solvency constraint (over samples and then years)": float,
         "Average necessary add. import (over samples and then years)": float,
         "Average total necessary import (over samples and then years)": float,
         "Average necessary add. import per capita (over samples and then years)": float,
         "Expected income (to calculate guaranteed income)": "list of floats",
         "Penalty for food shortage": float,
         "Penalty for insolvency": float,
         "Max. possible probability for food security (excluding solvency constraint)": float,
         "Max. possible probability for solvency (excluding food security constraint)": float,
         "Average necessary debt (excluding food security constraint)": float,
         "Average necessary debt": float, 
         "Average necessary debt (over all samples with negative final fund)": float,
         "Averge necessary debt per capita (over all samples with negative final fund)": float,
         "Averge necessary debt per capita (over all samples)": float,
         "Average final fund (over all samples)": float,
         "Number of samples with negative final fund": int,
         "Probability for a catastrophic year": float,
         "Share of samples with no catastrophe": float,
         "Share of years/clusters with unprofitable rice yields": float,
         "Share of years/clusters with unprofitable maize yields": float,
         "Share of West Africa's population that is living in total considered region (2015)": float,
         "Share of West Africa's population that is living in considered clusters (2015)": "list of floats",
         "On average cultivated area per cluster": "list of floats",
         "Average yearly total cultivated area": float,
         "Average food demand penalty (over samples and then years)": float,
         "Average solvency penalty (over samples)": float,
         "Average cultivation costs per cluster (over samples and then years)": "list of floats",
         "Average total cultivation costs": float,
         "Expected total costs": float,
         "Number of occurrences per cluster where farmers make losses": "list of ints",
         "Average income per cluster in final run (over samples and then years)": "list of ints",
         "Average income per cluster in final run scaled with capita (over samples and then years)": "list of ints",
         "Average government payouts per cluster (over samples)": "list of floats",
         "Resulting probability for food security": float,
         "Resulting probability for solvency": float,
         "Resulting probability for food security for VSS": float,
         "Resulting probability for solvency for VSS": float,
         "Value of stochastic solution": float,
         "VSS as share of total costs (sto. solution)": float,
         "VSS as share of total costs (det. solution)": float,
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
        'Average necessary add. import excluding solvency constraint (over samples and then years)', 
        'Average necessary add. import (over samples and then years)', 
        'Average total necessary import (over samples and then years)', 
        'Average necessary add. import per capita (over samples and then years)',
        'Expected income (to calculate guaranteed income)', 
        'Penalty for food shortage', 
        'Penalty for insolvency', 
        'Max. possible probability for food security (excluding solvency constraint)',
        'Max. possible probability for solvency (excluding food security constraint)',
        'Average necessary debt (excluding food security constraint)',
        'Average necessary debt',
        'Average necessary debt (over all samples with negative final fund)',
        'Averge necessary debt per capita (over all samples with negative final fund)',
        'Averge necessary debt per capita (over all samples)',
        'Average final fund (over all samples)',
        'Number of samples with negative final fund',
        'Probability for a catastrophic year', 
        'Share of samples with no catastrophe', 
        'Share of years/clusters with unprofitable rice yields', 
        'Share of years/clusters with unprofitable maize yields', 
        'Share of West Africa\'s population that is living in total considered region (2015)', 
        'Share of West Africa\'s population that is living in considered clusters (2015)',
        'On average cultivated area per cluster', 
        'Average yearly total cultivated area',
        'Average food demand penalty (over samples and then years)', 
        'Average solvency penalty (over samples)', 
        'Average cultivation costs per cluster (over samples and then years)', 
        'Average total cultivation costs',
        'Expected total costs', 
        'Number of occurrences per cluster where farmers make losses', 
        'Average income per cluster in final run (over samples and then years)',
        'Average income per cluster in final run scaled with capita (over samples and then years)',
        'Average government payouts per cluster (over samples)', 
        'Resulting probability for food security', 
        'Resulting probability for solvency', 
        'Resulting probability for food security for VSS', 
        'Resulting probability for solvency for VSS', 
        'Value of stochastic solution', 
        'VSS as share of total costs (sto. solution)',
        'VSS as share of total costs (det. solution)',
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

