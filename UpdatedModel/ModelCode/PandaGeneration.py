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

from ModelCode.Auxiliary import _printing

# %% ########## FUNCTIONS TO CREATE AND FILL THE RESULTS PANDA OBJECT ###########

def _WriteToPandas(settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, crop_alloc, \
                    meta_sol, meta_sol_vss, VSS_value, validation_values, \
                    fn_fullresults, console_output = None, logs_on = None,
                    file = "current_panda"):
    """
    Adds information on the model run to the given pandas csv.
    
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
        optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'yearly_fixed_costs', 'fd_penalty', 'avg_fd_penalty', 
        'sol_penalty', 'shortcomings', 'exp_shortcomings', 'expected_incomes', 
        'profits', 'num_years_with_losses', 'payouts', 'final_fund', 'probF', 
        'probS', 'avg_nec_import', 'avg_nec_debt')
    meta_sol_vss : dict
        additional information on the deterministic solution 
    VSS_value : float
        VSS calculated as the difference between total costs using 
        deterministic solution for crop allocation and stochastic solution
        for crop allocation       
    validation_values : dict
        total costs and penalty costs for the resulted crop areas but a higher 
        sample size of crop yields for validation ("sample_size", 
        "total_costs", "total_costs_val", "fd_penalty", "fd_penalty_val", 
        "sol_penalty", "sol_penalty_val", "total_penalties", 
        "total_penalties_val", "deviation_penalties")
    fn_fullresults : str
        Filename used to save the full model results.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        The default is defined in ModelCode/GeneralSettings.
    file : str
        filename of panda csv to which the information is to be added.
    Returns
    -------
    None.

    """
    
    _printing("\nAdding results to pandas", console_output = console_output, logs_on = logs_on)
    
    # some pre-calculations
    pop_per_cluster = np.outer(population_information["total_pop_scen"], population_information["pop_cluster_ratio2015"])
    pop_of_area = population_information["population"]
    food_shortage_capita = meta_sol["shortcomings"]/pop_of_area
    # meta_sol["shortcimings"] reports shortcomings as positive values, and 
    # surplus as zero
    food_shortage_capita_only_shortage = food_shortage_capita.copy()
    food_shortage_capita_only_shortage[food_shortage_capita_only_shortage == 0] = np.nan
    # to avoid getting nan as average: set years that never have shortcomings back to zero
    food_shortage_capita_only_shortage[:,np.sum(np.isnan(food_shortage_capita_only_shortage), axis = 0) == settings["N"]] = 0
    
    
    wh_neg_fund = np.where(meta_sol["final_fund"] < 0)[0]
    ff_debt = -meta_sol["final_fund"][wh_neg_fund]
    ter_years = args["terminal_years"][wh_neg_fund].astype(int)
    
    ff_debt_all = -meta_sol["final_fund"].copy()
    ff_debt_all[ff_debt_all < 0] = 0
    ter_years_all = args["terminal_years"].astype(int)
    
    yearly_fixed_costs = np.nansum(meta_sol["yearly_fixed_costs"], axis = 2)
    yearly_fixed_costs[yearly_fixed_costs == 0] = np.nan
    cultivation_costs = np.sum(np.nanmean(yearly_fixed_costs, axis = 0))
    
    # setting up dictionary of parameters to add to the panda object as new row:
        
    # 1 settings
    panda = {"Penalty method":                     settings["PenMet"],
             "Input probability food security":    settings["probF"],
             "Input probability solvency":         settings["probS"],
             "Number of crops":                    settings["num_crops"],
             "Number of clusters":                 settings["k"],
             "Used clusters":                      settings["k_using"],
             "Yield projection":                   settings["yield_projection"],
             "Simulation start":                   settings["sim_start"],
             "Population scenario":                settings["pop_scenario"],
             "Risk level covered":                 settings["risk"],
             "Tax rate":                           settings["tax"],
             "Share of income that is guaranteed": settings["perc_guaranteed"],
             "Initial fund size":                  settings["ini_fund"],
             "Sample size":                        settings["N"],
             "Sample size for validation":         settings["validation_size"],
             "Number of covered years":            settings["T"]}
    
    # 2 resulting penalties and probabilities
    panda["Penalty for food shortage"]               = args["rhoF"]
    panda["Penalty for insolvency"]                  = args["rhoS"]
    panda["Resulting probability for food security"] = meta_sol["probF"]
    panda["Resulting probability for solvency"]      = meta_sol["probS"]
    panda["Max. possible probability for food security (excluding solvency constraint)"] \
        = AddInfo_CalcParameters["probF_onlyF"]
    panda["Max. possible probability for solvency (excluding food security constraint)"] \
        = AddInfo_CalcParameters["probS_onlyS"]
    
    # 3 yield information
    panda["Probability for a catastrophic year"]                    = yield_information["prob_cat_year"]
    panda["Share of samples with no catastrophe"]                   = yield_information["share_no_cat"]
    panda["Share of years/clusters with unprofitable rice yields"]  = yield_information["share_rice_np"]
    panda["Share of years/clusters with unprofitable maize yields"] = yield_information["share_maize_np"]
    
    # 4 population information
    panda["Share of West Africa's population that is living in total considered region (2015)"] \
            = np.sum(population_information["pop_cluster_ratio2015"])
    panda["Share of West Africa's population that is living in considered clusters (2015)"] \
            = list(population_information["pop_cluster_ratio2015"])
         
    # 5 crop areas
    panda["On average cultivated area per cluster"] = list(np.nanmean(crop_alloc, axis = (0,1)))
    panda["Average yearly total cultivated area"]   = np.nanmean(np.nansum(crop_alloc, axis = (1,2)))
    panda["Total cultivation costs"]                = cultivation_costs
        
    # 6 food demand and needed import
    panda["Import (given as model input)"] = args["import"]
    panda["Average food demand"]           = np.mean(args["demand"])
    panda["Food demand per capita"]        = args["demand"][0]/population_information["population"][0] 
    
    panda["Average total necessary import (over samples and then years)"] \
        = args["import"] + meta_sol["avg_nec_import"]
    panda["Average necessary add. import (over samples and then years)"] \
        = meta_sol["avg_nec_import"] # includes also caes that don't need import as zero
    panda["Average necessary add. import excluding solvency constraint (over samples and then years)"] \
        = AddInfo_CalcParameters["import_onlyF"]
    panda["Average necessary add. import per capita (over samples and then years)"] \
        = np.nanmean(np.nanmean(food_shortage_capita, axis = 0))*1e9
    panda["Average necessary add. import per capita (over samples and then years, only cases that need import)"] \
        = np.nanmean(np.nanmean(food_shortage_capita_only_shortage, axis = 0))*1e9 
        # TODO: Befroe I only had "Average necessary add. import per capita
        # (over samples and then years)", which was actually directly taking the
        # average over samples and years in one step, and which excluded cases
        # that don't need import (instead of including them as zero).
        # Another option would be to take the average over the surplus or 
        # food shortage 
        
    # 7 expected income, final fund and needed debt
    panda["Expected income (to calculate guaranteed income)"] \
        = list(AddInfo_CalcParameters["expected_incomes"])
    panda["Number of occurrences per cluster where farmers make losses"] \
        = list(meta_sol["num_years_with_losses"])
    panda["Number of samples with negative final fund"] \
        = np.nansum(meta_sol["final_fund"] < 0)
    panda["Average final fund (over all samples)"] \
        = np.nanmean(meta_sol["final_fund"])
        # TODO maybe don't include cases that don't have a catastrohpe in average?
    
    panda["Average income per cluster in final run (over samples and then years)"] \
        = list(np.nanmean(np.nanmean(meta_sol["profits"], axis = 0), axis = 0))  # profits include both actual profits and losses
    panda["Average income per cluster in final run scaled with capita (over samples and then years)"] \
        = list(np.nanmean(np.nanmean(meta_sol["profits"], axis = 0)/(pop_per_cluster/1e9), axis = 0))
    panda["Aggregated average government payouts per cluster (over samples)"] \
        = list(np.nansum(np.nanmean(meta_sol["payouts"], axis = 0), axis = 0))
        
    panda["Average necessary debt (excluding food security constraint)"] \
        = AddInfo_CalcParameters["debt_onlyS"]
    panda["Average necessary debt"] \
        = meta_sol["avg_nec_debt"] # includes cases that don't need debt as zero
    panda["Average necessary debt (over all samples with negative final fund)"] \
        = np.nanmean(ff_debt)
    panda["Average necessary debt per capita (over all samples with negative final fund)"] \
        = np.nanmean(ff_debt / (pop_of_area[ter_years]/1e9))
    panda["Average necessary debt per capita (over all samples)"] \
        = np.nanmean(ff_debt_all / (pop_of_area[ter_years_all]/1e9)) # negative debt set to zero
    
    # 8 different cost items in objective function
    panda["Average food demand penalty (over samples and then years)"] \
        = np.nanmean(np.nanmean(meta_sol["fd_penalty"], axis = 0))
    panda["Average solvency penalty (over samples)"] \
        = np.mean(meta_sol["sol_penalty"])
    panda["Average total cultivation costs"] \
        = np.nanmean(np.nansum(meta_sol["yearly_fixed_costs"], axis = (1,2)))
    panda["Expected total costs"] \
        = meta_sol["exp_tot_costs"] # this includes varying cultivation costs (depending on catastrophic year)
        
    # 9 VSS
    panda["Value of stochastic solution"]                    = VSS_value
    panda["VSS as share of total costs (sto. solution)"]     = VSS_value/meta_sol["exp_tot_costs"]
    panda["VSS as share of total costs (det. solution)"]     = VSS_value/meta_sol_vss["exp_tot_costs"]
    panda["Resulting probability for food security for VSS"] = meta_sol_vss["probF"]
    panda["Resulting probability for solvency for VSS"]      = meta_sol_vss["probS"]
        
      
    # 10 technincal variables   
    panda["Validation value (deviation of total penalty costs)"] = validation_values["deviation_penalties"]
    panda["Seed (for yield generation)"]                         = settings["seed"]
    panda["Filename for full results"]                           = fn_fullresults
       
       
            
    # if np.isnan(panda["Average necessary add. import per capita (over samples and then years)"]):
    #     panda["Average necessary add. import per capita (over samples and then years)"] = 0
    # if np.isnan(dict_for_pandas["Average food shortcomings per capita (over all years and samples with shortcomings)"]):
    #     dict_for_pandas["Average food shortcomings per capita (over all years and samples with shortcomings)"] = 0
     
    # load panda object
    if not os.path.exists("ModelOutput/Pandas/" + file + ".csv"):
        CreateEmptyPanda(file)
    
    # add new row
    current_panda = pd.read_csv("ModelOutput/Pandas/" + file + ".csv")
    current_panda = current_panda.append(panda, ignore_index = True)
    
    # try to sve updated panda object as csv
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

def CreateEmptyPanda(file = "current_panda"):
    """
    Creating a new empty pandas object with the correct columns.    

    Parameters
    ----------
    file : str, optional
        Name of file to save the new panda csv to.
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
    Load the csv into a panda object.

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
 
    def _ConvertListsInts(arg):
        arg = arg.strip("][").split(", ")
        res = []
        for j in range(0, len(arg)):
            res.append(int(arg[j]))
        return(res)
    
    def _ConvertListsFloats(arg):
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
            dict_convert[key] = _ConvertListsFloats
        elif dict_convert[key] == "list of ints":
            dict_convert[key] = _ConvertListsInts
    
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
    
def _SetUpPandaDicts():
    """
    Sets up and saves the dicts that are needed to deal with the panda csvs.
    Dictonaries are defined here and saved so they can be loaded from other
    functions. 
    !! If additional variables are included in the write_to_pandas function,
    they need to be added to all three dictionaries here as well!!

    Returns
    ------
    None.

    """
    units = {"Penalty method": "",
        "Input probability food security": "",
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
        
        "Penalty for food shortage": "[$\$/10^3\,$kcal]",
        "Penalty for insolvency": "[$\$/\$$]",
        "Resulting probability for food security": "",
        "Resulting probability for solvency": "",
        "Max. possible probability for food security (excluding solvency constraint)": "",
        "Max. possible probability for solvency (excluding food security constraint)": "",
        
        "Probability for a catastrophic year": "",
        "Share of samples with no catastrophe": "",
        "Share of years/clusters with unprofitable rice yields": "",
        "Share of years/clusters with unprofitable maize yields": "",
        
        "Share of West Africa's population that is living in total considered region (2015)": "",
        "Share of West Africa's population that is living in considered clusters (2015)": "",
        
        "On average cultivated area per cluster": "[$10^9\,ha$]",
        "Average yearly total cultivated area": "[$10^9\,ha$]",
        "Total cultivation costs": "[$10^9\,\$$]",
        
        "Import (given as model input)": "[$10^{12}\,$kcal]",
        "Average food demand": "[$10^{12}\,$kcal]",
        "Food demand per capita" : "[$10^{12}\,$kcal]",
        "Average total necessary import (over samples and then years)": "[$10^{12}\,$kcal]",
        "Average necessary add. import (over samples and then years)":"[$10^{12}\,$kcal]",
        "Average necessary add. import excluding solvency constraint (over samples and then years)": "[$10^{12}\,$kcal]",
        "Average necessary add. import per capita (over samples and then years)": "[$10^{3}\,$kcal]",
        "Average necessary add. import per capita (over samples and then years, only cases that need import)": "[$10^{3}\,$kcal]",
        
        "Expected income (to calculate guaranteed income)": "[$10^9\,\$$]",
        "Number of occurrences per cluster where farmers make losses": "",
        "Number of samples with negative final fund": "",
        "Average final fund (over all samples)": "[$10^9\,\$$]",
        "Average income per cluster in final run (over samples and then years)": "[$10^9\,\$$]",
        "Average income per cluster in final run scaled with capita (over samples and then years)": "[$\$$]",
        "Aggregated average government payouts per cluster (over samples)": "[$10^9\,\$$]",
        "Average necessary debt (excluding food security constraint)": "[$10^9\,\$$]",
        "Average necessary debt": "[$10^9\,\$$]",
        "Average necessary debt (over all samples with negative final fund)": "[$10^9\,\$$]",
        "Average necessary debt per capita (over all samples with negative final fund)": "[$\$$]",
        "Average necessary debt per capita (over all samples)": "[$\$$]",
        
        "Average food demand penalty (over samples and then years)": "[$10^9\,\$$]",
        "Average solvency penalty (over samples)": "[$10^9\,\$$]",
        "Average total cultivation costs": "[$10^9\,\$$]",
        "Expected total costs": "[$10^9\,\$$]",
        
        "Value of stochastic solution": "[$10^9\,\$$]",
        "VSS as share of total costs (sto. solution)": "",
        "VSS as share of total costs (det. solution)": "",
        "Resulting probability for food security for VSS": "",
        "Resulting probability for solvency for VSS": "",
        
        "Validation value (deviation of total penalty costs)": "",
        "Seed (for yield generation)": "",
        "Filename for full results": ""}   
    
    convert =  {"Penalty method": str,
         "Input probability food security": float,
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
         
         "Penalty for food shortage": float,
         "Penalty for insolvency": float,
         "Resulting probability for food security": float,
         "Resulting probability for solvency": float,
         "Max. possible probability for food security (excluding solvency constraint)": float,
         "Max. possible probability for solvency (excluding food security constraint)": float,
         
         "Probability for a catastrophic year": float,
         "Share of samples with no catastrophe": float,
         "Share of years/clusters with unprofitable rice yields": float,
         "Share of years/clusters with unprofitable maize yields": float,
        
         "Share of West Africa's population that is living in total considered region (2015)": float,
         "Share of West Africa's population that is living in considered clusters (2015)": "list of floats",
        
         "On average cultivated area per cluster": "list of floats",
         "Average yearly total cultivated area": float,
         "Total cultivation costs": float,
         
         "Import (given as model input)": float,
         "Average food demand": float,
         "Food demand per capita" : float,
         "Average total necessary import (over samples and then years)": float,
         "Average necessary add. import (over samples and then years)": float,
         "Average necessary add. import excluding solvency constraint (over samples and then years)": float,
         "Average necessary add. import per capita (over samples and then years)": float,
         "Average necessary add. import per capita (over samples and then years, only cases that need import)": float,
        
         "Expected income (to calculate guaranteed income)": "list of floats",
         "Number of occurrences per cluster where farmers make losses": "list of ints",
         "Number of samples with negative final fund": "list of ints",
         "Average final fund (over all samples)": float,
         "Average income per cluster in final run (over samples and then years)": "list of floats",
         "Average income per cluster in final run scaled with capita (over samples and then years)": "list of floats",
         "Aggregated average government payouts per cluster (over samples)": "list of floats",
         "Average necessary debt (excluding food security constraint)": float,
         "Average necessary debt": float,
         "Average necessary debt (over all samples with negative final fund)": float,
         "Average necessary debt per capita (over all samples with negative final fund)": float,
         "Average necessary debt per capita (over all samples)": float,  
         
         "Average food demand penalty (over samples and then years)": float,
         "Average solvency penalty (over samples)": float,
         "Average total cultivation costs": float,
         "Expected total costs": float,
        
         "Value of stochastic solution": float,
         "VSS as share of total costs (sto. solution)": float,
         "VSS as share of total costs (det. solution)": float,
         "Resulting probability for food security for VSS": float,
         "Resulting probability for solvency for VSS": float,
        
         "Validation value (deviation of total penalty costs)": float,
         "Seed (for yield generation)": int,
         "Filename for full results": str}
        
    colnames = ["Penalty method",
        "Input probability food security",
        "Input probability solvency",
        "Number of crops",
        "Number of clusters",
        "Used clusters",
        "Yield projection",
        "Simulation start",
        "Population scenario",
        "Risk level covered",
        "Tax rate",
        "Share of income that is guaranteed",
        "Initial fund size",
        "Sample size",
        "Sample size for validation",
        "Number of covered years",
        
        "Penalty for food shortage",
        "Penalty for insolvency",
        "Resulting probability for food security",
        "Resulting probability for solvency",
        "Max. possible probability for food security (excluding solvency constraint)",
        "Max. possible probability for solvency (excluding food security constraint)",
        
        "Probability for a catastrophic year",
        "Share of samples with no catastrophe",
        "Share of years/clusters with unprofitable rice yields",
        "Share of years/clusters with unprofitable maize yields",
        
        "Share of West Africa's population that is living in total considered region (2015)",
        "Share of West Africa's population that is living in considered clusters (2015)",
        
        "On average cultivated area per cluster",
        "Average yearly total cultivated area",
        "Total cultivation costs",
        
        "Import (given as model input)",
        "Average food demand",
        "Food demand per capita",
        "Average total necessary import (over samples and then years)",
        "Average necessary add. import (over samples and then years)",
        "Average necessary add. import excluding solvency constraint (over samples and then years)",
        "Average necessary add. import per capita (over samples and then years)",
        "Average necessary add. import per capita (over samples and then years, only cases that need import)",
        
        "Expected income (to calculate guaranteed income)",
        "Number of occurrences per cluster where farmers make losses",
        "Number of samples with negative final fund",
        "Average final fund (over all samples)",
        "Average income per cluster in final run (over samples and then years)",
        "Average income per cluster in final run scaled with capita (over samples and then years)",
        "Aggregated average government payouts per cluster (over samples)",
        "Average necessary debt (excluding food security constraint)",
        "Average necessary debt",
        "Average necessary debt (over all samples with negative final fund)",
        "Average necessary debt per capita (over all samples with negative final fund)",
        "Average necessary debt per capita (over all samples)",
        
        "Average food demand penalty (over samples and then years)",
        "Average solvency penalty (over samples)",
        "Average total cultivation costs",
        "Expected total costs",
        
        "Value of stochastic solution",
        "VSS as share of total costs (sto. solution)",
        "VSS as share of total costs (det. solution)",
        "Resulting probability for food security for VSS",
        "Resulting probability for solvency for VSS",
        
        "Validation value (deviation of total penalty costs)",
        "Seed (for yield generation)",
        "Filename for full results"]
    
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "wb") as fp:
        pickle.dump(units, fp)
        
    with open("ModelOutput/Pandas/ColumnNames.txt", "wb") as fp:
        pickle.dump(colnames, fp)
        
    with open("ModelOutput/Pandas/ColumnTypes.txt", "wb") as fp:
        pickle.dump(convert, fp)
        
    return(None)

