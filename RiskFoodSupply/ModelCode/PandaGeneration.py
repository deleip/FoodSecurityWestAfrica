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


def _WriteToPandas(settings, args, yield_information, population_information, \
                status, all_durations, exp_incomes, crop_alloc, \
                meta_sol, crop_allocF, meta_solF, crop_allocS, meta_solS, 
                penalty_methods, crop_alloc_vss, meta_sol_vss, VSS_value, \
                validation_values, fn_fullresults, console_output = None, \
                logs_on = None, file = "current_panda"):
    """
    Adds information on the model run to the given pandas csv.
    
    !! If additional variables are included in the _WriteToPandas function,
    they need to be added to all three dictionaries in _SetUpPandaDicts as well !!
    
    Parameters
    ---------- 
    settings : dict
        The model input settings that were given by user. 
    args : dict
        Dictionary of arguments needed as direct model input.
    yield_information : dict
        Information on the yield distributions for the considered clusters.
    population_information : dict
        Information on the population in the considered area.
    status : int
        status of solver (optimal: 2)
    all_durations :  dict
        Information on the duration of different steps of the model framework.
    exp_incomes : dict
        Expected income (on which guaranteed income is based) for the stochastic
        and the deterministic setting.
    crop_alloc :  np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output ('exp_tot_costs', 
        'fix_costs', 'yearly_fixed_costs', 'fd_penalty', 'avg_fd_penalty', 
        'sol_penalty', 'shortcomings', 'exp_shortcomings', 'avg_profits_preTax', 
        'avg_profits_afterTax', 'food_supply', 'profits_preTax', 
        'profits_afterTax', 'num_years_with_losses', 'payouts', 'final_fund', 
        'probF', 'probS', 'avg_nec_import', 'avg_nec_debt', 'guaranteed_income')
    crop_allocF : np.array
        optimal crop allocation for scenario with only food security objective
    meta_solF : dict
        additional information on model output for scenario with only food
        security objective
    crop_allocS : np.array
        optimal crop allocation for scenario with only solvency objective
    meta_solS : dict
        additional information on model output for scenario with only solvency
        objective
    penalty_methods : dict
        Which methods were used to find the penalty for food shortages and
        insolvency.
    crop_alloc_vss : np.array
        deterministic solution for optimal crop areas    
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
        filename of panda csv to which the information is to be added. The
        default is "current_panda".
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
    ff_debt_neg = -meta_sol["final_fund"][wh_neg_fund]
    ter_years_neg = args["terminal_years"][wh_neg_fund].astype(int)
    
    wh_catastrophe = np.where(args["terminal_years"] != -1)[0]
    ff_debt_cat = -meta_sol["final_fund"][wh_catastrophe]
    ff_debt_cat[ff_debt_cat < 0] = 0
    ter_years_cat = args["terminal_years"][wh_catastrophe].astype(int)
    
    ff_debt_all = -meta_sol["final_fund"].copy()
    ff_debt_all[ff_debt_all < 0] = 0
    ter_years_all = args["terminal_years"].astype(int)
    
    # finding cultivation costs by taking a sample w/o catastrophic yields
    sample_no_cat = np.where(args["terminal_years"] == -1)[0][0]
    cultivation_costs = np.sum(meta_sol["yearly_fixed_costs"][sample_no_cat, :, :])
    cultivation_costs_det = np.sum(meta_sol_vss["yearly_fixed_costs"][sample_no_cat, :, :])
    
    # setting up dictionary of parameters to add to the panda object as new row:
        
    # 1 settings
    panda = {"Penalty method":                     settings["PenMet"],
             "Input probability food security":    settings["probF"],
             "Input probability solvency":         settings["probS"],
             "Including solvency constraint":      settings["solv_const"],
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
    if settings["validation_size"] is None:
        panda["Sample size for validation"] = 0
    
    # 2 resulting penalties and probabilities
    panda["Penalty for food shortage"]               = args["rhoF"]
    panda["Penalty for insolvency"]                  = args["rhoS"]
    panda["Resulting probability for food security"] = meta_sol["probF"] * 100
    panda["Resulting probability for solvency"]      = meta_sol["probS"] * 100
    if meta_solF is not None:
        panda["Max. possible probability for food security (excluding solvency constraint)"] \
            = meta_solF["probF"] 
    else:
        panda["Max. possible probability for food security (excluding solvency constraint)"] \
            = np.nan
    if meta_solS is not None:
        panda["Max. possible probability for solvency (excluding food security constraint)"] \
            = meta_solS["probS"]
    else:
        panda["Max. possible probability for solvency (excluding food security constraint)"] \
            = np.nan
        
    
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
    panda["Available arable area"]  = np.sum(args["max_areas"])
    panda["On average cultivated area per cluster"]  = list(np.nanmean(crop_alloc, axis = (0,1)))
    panda["Average yearly total cultivated area"]    = np.nanmean(np.nansum(crop_alloc, axis = (1,2)))
    panda["Total cultivation costs (sto. solution)"] = cultivation_costs
        
    # 6 food demand and needed import
    panda["Import (given as model input)"] = args["import"]
    panda["Average food demand"]           = np.mean(args["demand"])
    panda["Food demand per capita"]        = args["demand"][0]/population_information["population"][0] 
    
    panda["Average aggregate food shortage (without taking into account imports)"] \
        = args["import"] + meta_sol["avg_nec_import"]
    panda["Average aggregate food shortage"] \
        = meta_sol["avg_nec_import"] # includes also caes that don't need import as zero
    if meta_solF is not None:
        panda["Average aggregate food shortage excluding solvency constraint"] \
            = meta_solF["avg_nec_import"]
    else:
        panda["Average aggregate food shortage excluding solvency constraint"] \
            = np.nan
    panda["Average aggregate food shortage per capita"] \
        = np.nanmean(np.nanmean(food_shortage_capita, axis = 0))*1e9
    panda["Average aggregate food shortage per capita (including only samples that have shortage)"] \
        = np.nanmean(np.nanmean(food_shortage_capita_only_shortage, axis = 0))*1e9 
        
    # 7 a priori expected income, resulting average income
    panda["Expected income (stochastic setting)"] \
        = list(exp_incomes["sto. setting"])
    panda["Expected income (deterministic setting)"] \
        = list(exp_incomes["det. setting"])
        
    panda["Number of occurrences per cluster where farmers make losses"] \
        = list(meta_sol["num_years_with_losses"])
    panda["Average profits (pre tax) per cluster in final run (over samples and then years)"] \
        = list(np.nanmean(np.nanmean(meta_sol["profits_preTax"], axis = 0), axis = 0))  # profits include both actual profits and losses
    panda["Average profits (after tax) per cluster in final run (over samples and then years)"] \
        = list(np.nanmean(np.nanmean(meta_sol["profits_afterTax"], axis = 0), axis = 0))  # profits include both actual profits and losses
    panda["Average profits (pre tax) per cluster in final run scaled with capita (over samples and then years)"] \
        = list(np.nanmean(np.nanmean(meta_sol["profits_preTax"], axis = 0)/(pop_per_cluster/1e9), axis = 0))
    panda["Average profits (after tax) per cluster in final run scaled with capita (over samples and then years)"] \
        = list(np.nanmean(np.nanmean(meta_sol["profits_afterTax"], axis = 0)/(pop_per_cluster/1e9), axis = 0))
    panda["Aggregated average government payouts per cluster (over samples)"] \
        = list(np.nansum(np.nanmean(meta_sol["payouts"], axis = 0), axis = 0))
        
    # 8 final fund and needed debt
    panda["Number of samples with negative final fund"] \
        = np.nansum(meta_sol["final_fund"] < 0)
    panda["Average final fund (over all samples)"] \
        = np.nanmean(meta_sol["final_fund"])
    panda["Average final fund (over samples with catastrophe)"] \
        = np.nanmean(meta_sol["final_fund"][args["terminal_years"] != -1])
    
    
    if meta_solS is not None:
        panda["Average aggregate debt after payout (excluding food security constraint)"] \
            = meta_solS["avg_nec_debt"]
    else:
        panda["Average aggregate debt after payout (excluding food security constraint)"] \
            = np.nan
    panda["Average aggregate debt after payout"] \
        = meta_sol["avg_nec_debt"] # includes cases that don't need debt as zero
    panda["Average aggregate debt after payout (including only samples with negative final fund)"] \
        = np.nanmean(ff_debt_neg)
    panda["Average aggregate debt after payout (including only samples with catastrophe)"] \
        = np.nanmean(ff_debt_cat)
    panda["Average aggregate debt after payout per capita (including only samples with catastrophe)"] \
        = np.nanmean(ff_debt_cat / (pop_of_area[ter_years_cat]/1e9))
    panda["Average aggregate debt after payout per capita (including only samples with negative final fund)"] \
        = np.nanmean(ff_debt_neg / (pop_of_area[ter_years_neg]/1e9))
    panda["Average aggregate debt after payout per capita"] \
        = np.nanmean(ff_debt_all / (pop_of_area[ter_years_all]/1e9)) # negative debt set to zero
    
    # 9 different cost items in objective function
    panda["Average food demand penalty (over samples and then years)"] \
        = np.nanmean(np.nanmean(meta_sol["fd_penalty"], axis = 0))
    panda["Average total food demand penalty (over samples)"] \
        =np.nanmean(np.nansum(meta_sol["fd_penalty"], axis = 1))
    panda["Average solvency penalty (over samples)"] \
        = np.mean(meta_sol["sol_penalty"])
    panda["Average total cultivation costs"] \
        = np.nanmean(np.nansum(meta_sol["yearly_fixed_costs"], axis = (1,2)))
    panda["Expected total costs"] \
        = meta_sol["exp_tot_costs"] # this includes varying cultivation costs (depending on catastrophic year)
        
    # 10 VSS
    panda["Value of stochastic solution"]                    = VSS_value      # diff of total costs using det. solution and using 
    panda["Total cultivation costs (det. solution)"]         = cultivation_costs_det
    if meta_sol["exp_tot_costs"] == 0:
        panda["VSS as share of total costs (sto. solution)"] = 0
    else:
        panda["VSS as share of total costs (sto. solution)"] = VSS_value/meta_sol["exp_tot_costs"]
    panda["VSS as share of total costs (det. solution)"]     = VSS_value/meta_sol_vss["exp_tot_costs"]
    panda["VSS in terms of avg. nec. debt"] \
            = meta_sol_vss["avg_nec_debt"] - meta_sol["avg_nec_debt"]
    if meta_sol_vss["avg_nec_debt"] == 0:
        panda["VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution"] = 0
    else:
        panda["VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution"] \
                = (meta_sol_vss["avg_nec_debt"] - meta_sol["avg_nec_debt"])/meta_sol_vss["avg_nec_debt"]
    panda["VSS in terms of avg. nec. debt as share of avg. nec. debt of sto. solution"] \
            =(meta_sol_vss["avg_nec_debt"] - meta_sol["avg_nec_debt"])/meta_sol["avg_nec_debt"]
    panda["VSS in terms of avg. nec. import"] \
            = meta_sol_vss["avg_nec_import"] - meta_sol["avg_nec_import"]
    panda["VSS in terms of avg. nec. import as share of avg. nec. import of det. solution"] \
            = (meta_sol_vss["avg_nec_import"] - meta_sol["avg_nec_import"])/meta_sol_vss["avg_nec_import"]
    if meta_sol["avg_nec_import"] == 0:
        panda["VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution"] \
            = 0
    else:
        panda["VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution"] \
            = (meta_sol_vss["avg_nec_import"] - meta_sol["avg_nec_import"])/meta_sol["avg_nec_import"]
    
    panda["Resulting probability for food security for VSS"] = meta_sol_vss["probF"]
    panda["Resulting probability for solvency for VSS"]      = meta_sol_vss["probS"]
        
      
    # 11 technincal variables   
    if len(validation_values) > 0:
        panda["Validation value (deviation of total penalty costs)"] = validation_values["deviation_penalties"]
    else:
        panda["Validation value (deviation of total penalty costs)"] = 0
    panda["Seed (for yield generation)"]   = settings["seed"]
    panda["Filename for full results"]     = fn_fullresults
    panda["Method for finding rhoF"]       = penalty_methods["methodF"]
    panda["Method for finding rhoS"]       = penalty_methods["methodS"]
    panda["Accuracy for demanded probF"]   = settings["accuracyF_demandedProb"]
    panda["Accuracy for demanded probS"]   = settings["accuracyS_demandedProb"]
    panda["Accuracy for maximum probF"]    = settings["accuracyF_maxProb"]
    panda["Accuracy for maximum probS"]    = settings["accuracyS_maxProb"]
    panda["Accuracy for rhoF"]             = settings["accuracyF_rho"]
    panda["Accuracy for rhoS"]             = settings["accuracyS_rho"]
    panda["Accuracy for necessary help"]   = settings["accuracy_help"]
       
       
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
    
    def _ConvertFloat(arg):
        if arg == "":
            arg = np.NaN
        arg = float(arg)
        return(arg)
    
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
        elif dict_convert[key] == float:
            dict_convert[key] = _ConvertFloat
            
    # sometimes after changes laoding the file doesn't work anymore if something
    # is missinterpreted - than this code chunk helps figuring out where exactly 
    # someting is going wrong:
        
    # print(dict_convert.keys(), flush = True)
    # for key in dict_convert.keys():
    #     print(key, flush = True)
    #     dict_tmp = {key: dict_convert[key]}
    #     panda = pd.read_csv("ModelOutput/Pandas/" + file + ".csv", converters = dict_tmp)
    
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
    !! If additional variables are included in the _WriteToPandas function,
    they need to be added to all three dictionaries here as well !!

    Returns
    ------
    None.

    """
    units = {"Penalty method": "",
        "Input probability food security": "",
        "Input probability solvency": "",
        "Including solvency constraint": "",
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
        "Resulting probability for food security": "%",
        "Resulting probability for solvency": "%",
        "Max. possible probability for food security (excluding solvency constraint)": "",
        "Max. possible probability for solvency (excluding food security constraint)": "",
        
        "Probability for a catastrophic year": "",
        "Share of samples with no catastrophe": "",
        "Share of years/clusters with unprofitable rice yields": "",
        "Share of years/clusters with unprofitable maize yields": "",
        
        "Share of West Africa's population that is living in total considered region (2015)": "",
        "Share of West Africa's population that is living in considered clusters (2015)": "",
        
        "Available arable area": "[$10^9\,$ha]",
        "On average cultivated area per cluster": "[$10^9\,$ha]",
        "Average yearly total cultivated area": "[$10^9\,$ha]",
        "Total cultivation costs (sto. solution)": "[$10^9\,\$$]",
        
        "Import (given as model input)": "[$10^{12}\,$kcal]",
        "Average food demand": "[$10^{12}\,$kcal]",
        "Food demand per capita" : "[$10^{12}\,$kcal]",
        "Average aggregate food shortage (without taking into account imports)": "[$10^{12}\,$kcal]",
        "Average aggregate food shortage": "[$10^{12}\,$kcal]",
        "Average aggregate food shortage excluding solvency constraint": "[$10^{12}\,$kcal]",
        "Average aggregate food shortage per capita": "[$10^{3}\,$kcal]",
        "Average aggregate food shortage per capita (including only samples that have shortage)": "[$10^{3}\,$kcal]",

        "Expected income (stochastic setting)": "[$10^9\,\$$]",
        "Expected income (deterministic setting)": "[$10^9\,\$$]",
        "Number of occurrences per cluster where farmers make losses": "",
        "Average profits (pre tax) per cluster in final run (over samples and then years)": "[$10^9\,\$$]",
        "Average profits (after tax) per cluster in final run (over samples and then years)": "[$10^9\,\$$]",
        "Average profits (pre tax) per cluster in final run scaled with capita (over samples and then years)": "[$\$$]",
        "Average profits (after tax) per cluster in final run scaled with capita (over samples and then years)": "[$\$$]",
        "Aggregated average government payouts per cluster (over samples)": "[$10^9\,\$$]",
        
        "Number of samples with negative final fund": "",
        "Average final fund (over all samples)": "[$10^9\,\$$]",
        "Average final fund (over samples with catastrophe)": "[$10^9\,\$$]",
        "Average aggregate debt after payout (excluding food security constraint)": "[$10^9\,\$$]",
        "Average aggregate debt after payout": "[$10^9\,\$$]",
        "Average aggregate debt after payout (including only samples with negative final fund)": "[$10^9\,\$$]",
        "Average aggregate debt after payout (including only samples with catastrophe)": "[$10^9\,\$$]",
        "Average aggregate debt after payout per capita (including only samples with catastrophe)": "[$10^9\,\$$]",
        "Average aggregate debt after payout per capita (including only samples with negative final fund)": "[$\$$]",
        "Average aggregate debt after payout per capita": "[$\$$]",
        
        "Average food demand penalty (over samples and then years)": "[$10^9\,\$$]",
        "Average total food demand penalty (over samples)": "[$10^9\,\$$]",
        "Average solvency penalty (over samples)": "[$10^9\,\$$]",
        "Average total cultivation costs": "[$10^9\,\$$]",
        "Expected total costs": "[$10^9\,\$$]",
        
        "Value of stochastic solution": "[$10^9\,\$$]",
        "Total cultivation costs (det. solution)": "[$10^9\,\$$]",
        "VSS as share of total costs (sto. solution)": "",
        "VSS as share of total costs (det. solution)": "",
        "VSS in terms of avg. nec. debt": "[$10^9\,\$$]",
        "VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution": "",
        "VSS in terms of avg. nec. debt as share of avg. nec. debt of sto. solution": "",
        "VSS in terms of avg. nec. import": "[$10^{12}\,$kcal]",
        "VSS in terms of avg. nec. import as share of avg. nec. import of det. solution": "",
        "VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution": "",        
        "Resulting probability for food security for VSS": "",
        "Resulting probability for solvency for VSS": "",
        
        "Validation value (deviation of total penalty costs)": "",
        "Seed (for yield generation)": "",
        "Filename for full results": "",
        "Method for finding rhoF": "",
        "Method for finding rhoS": "",
        "Accuracy for demanded probF": "",
        "Accuracy for demanded probS": "",
        "Accuracy for maximum probF": "",
        "Accuracy for maximum probS": "",
        "Accuracy for rhoF": "",
        "Accuracy for rhoS": "",
        "Accuracy for necessary help": ""}
    
    
    convert =  {"Penalty method": str,
         "Input probability food security": float,
         "Input probability solvency": float,
         "Including solvency constraint": str,
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
        
         "Available arable area": float,
         "On average cultivated area per cluster": "list of floats",
         "Average yearly total cultivated area": float,
         "Total cultivation costs (sto. solution)": float,
         
         "Import (given as model input)": float,
         "Average food demand": float,
         "Food demand per capita" : float,
         "Average aggregate food shortage (without taking into account imports)": float,
         "Average aggregate food shortage": float,
         "Average aggregate food shortage excluding solvency constraint": float,
         "Average aggregate food shortage per capita": float,
         "Average aggregate food shortage per capita (including only samples that have shortage)": float,   
  
         "Expected income (stochastic setting)": "list of floats",
         "Expected income (deterministic setting)": "list of floats",
         "Number of occurrences per cluster where farmers make losses": "list of ints",
         "Average profits (pre tax) per cluster in final run (over samples and then years)": "list of floats",
         "Average profits (after tax) per cluster in final run (over samples and then years)": "list of floats",
         "Average profits (pre tax) per cluster in final run scaled with capita (over samples and then years)": "list of floats",
         "Average profits (after tax) per cluster in final run scaled with capita (over samples and then years)": "list of floats",
         "Aggregated average government payouts per cluster (over samples)": "list of floats",
         
         "Number of samples with negative final fund": "list of ints",
         "Average final fund (over all samples)": float,
         "Average final fund (over samples with catastrophe)": float,
         "Average aggregate debt after payout (excluding food security constraint)": float,
         "Average aggregate debt after payout": float,
         "Average aggregate debt after payout (including only samples with negative final fund)": float,
         "Average aggregate debt after payout (including only samples with catastrophe)": float,
         "Average aggregate debt after payout per capita (including only samples with catastrophe)": float, 
         "Average aggregate debt after payout per capita (including only samples with negative final fund)": float,
         "Average aggregate debt after payout per capita": float,  
         
         "Average food demand penalty (over samples and then years)": float,
         "Average total food demand penalty (over samples)": float,
         "Average solvency penalty (over samples)": float,
         "Average total cultivation costs": float,
         "Expected total costs": float,
        
         "Value of stochastic solution": float,
         "Total cultivation costs (det. solution)": float, 
         "VSS as share of total costs (sto. solution)": float,
         "VSS as share of total costs (det. solution)": float,
         "VSS in terms of avg. nec. debt": float,
         "VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution": float,
         "VSS in terms of avg. nec. debt as share of avg. nec. debt of sto. solution": float,
         "VSS in terms of avg. nec. import": float,
         "VSS in terms of avg. nec. import as share of avg. nec. import of det. solution": float,
         "VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution": float,   
         "Resulting probability for food security for VSS": float,
         "Resulting probability for solvency for VSS": float,
        
         "Validation value (deviation of total penalty costs)": float,
         "Seed (for yield generation)": int,
         "Filename for full results": str,
         "Method for finding rhoF": str,
         "Method for finding rhoS": str,
         "Accuracy for demanded probF": float,
         "Accuracy for demanded probS": float,
         "Accuracy for maximum probF": float,
         "Accuracy for maximum probS": float,
         "Accuracy for rhoF": float,
         "Accuracy for rhoS": float,
         "Accuracy for necessary help": float}
        
    colnames = ["Penalty method",
        "Input probability food security",
        "Input probability solvency",
        "Including solvency constraint",
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
        
        "Available arable area",
        "On average cultivated area per cluster",
        "Average yearly total cultivated area",
        "Total cultivation costs (sto. solution)",
        
        "Import (given as model input)",
        "Average food demand",
        "Food demand per capita",
        "Average aggregate food shortage (without taking into account imports)",
        "Average aggregate food shortage",
        "Average aggregate food shortage excluding solvency constraint",
        "Average aggregate food shortage per capita",
        "Average aggregate food shortage per capita (including only samples that have shortage)",
         
        "Expected income (stochastic setting)",
        "Expected income (deterministic setting)",
        "Number of occurrences per cluster where farmers make losses",
        "Average profits (pre tax) per cluster in final run (over samples and then years)",
        "Average profits (after tax) per cluster in final run (over samples and then years)",
        "Average profits (pre tax) per cluster in final run scaled with capita (over samples and then years)",
        "Average profits (after tax) per cluster in final run scaled with capita (over samples and then years)",
        "Aggregated average government payouts per cluster (over samples)",
        
        "Number of samples with negative final fund",
        "Average final fund (over all samples)",
        "Average final fund (over samples with catastrophe)",
        "Average aggregate debt after payout (excluding food security constraint)",
        "Average aggregate debt after payout",
        "Average aggregate debt after payout (including only samples with negative final fund)",
        "Average aggregate debt after payout (including only samples with catastrophe)",
        "Average aggregate debt after payout per capita (including only samples with catastrophe)",
        "Average aggregate debt after payout per capita (including only samples with negative final fund)",
        "Average aggregate debt after payout per capita",
        
        "Average food demand penalty (over samples and then years)",
        "Average total food demand penalty (over samples)",
        "Average solvency penalty (over samples)",
        "Average total cultivation costs",
        "Expected total costs",
        
        "Value of stochastic solution",
        "Total cultivation costs (det. solution)",
        "VSS as share of total costs (sto. solution)",
        "VSS as share of total costs (det. solution)",
        "VSS in terms of avg. nec. debt",
        "VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution",
        "VSS in terms of avg. nec. debt as share of avg. nec. debt of sto. solution",
        "VSS in terms of avg. nec. import",
        "VSS in terms of avg. nec. import as share of avg. nec. import of det. solution",
        "VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution",  
        "Resulting probability for food security for VSS",
        "Resulting probability for solvency for VSS",
        
        "Validation value (deviation of total penalty costs)",
        "Seed (for yield generation)",
        "Filename for full results",
        "Method for finding rhoF",
        "Method for finding rhoS",
        "Accuracy for demanded probF",
        "Accuracy for demanded probS",
        "Accuracy for maximum probF",
        "Accuracy for maximum probS",
        "Accuracy for rhoF",
        "Accuracy for rhoS",
        "Accuracy for necessary help"]
    
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "wb") as fp:
        pickle.dump(units, fp)
        
    with open("ModelOutput/Pandas/ColumnNames.txt", "wb") as fp:
        pickle.dump(colnames, fp)
        
    with open("ModelOutput/Pandas/ColumnTypes.txt", "wb") as fp:
        pickle.dump(convert, fp)
        
    return(None)
