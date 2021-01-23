# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:05:23 2021

@author: leip
"""
import numpy as np
import pandas as pd
import os
import sys
import pickle
import matplotlib.pyplot as plt

from ModelCode.Auxiliary import printing

# %% ############## FUNCTIONS DEALING WITH THE RESULTS PANDA CSV ##############

def write_to_pandas(settings, args, AddInfo_CalcParameters, yield_information, \
                    population_information, crop_alloc, \
                    meta_sol, meta_sol_vss, VSS_value, validation_values, \
                    console_output):
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
                           "Import (excluding solvency constraint)": args["import"],
                           "Import (excluding solvency constraint, including theoretical export)": AddInfo_CalcParameters["import"],
                           "Additional import needed when including solvency constraint": meta_sol["add_needed_import"],
                           "Expected income (to calculate guaranteed income)": list(AddInfo_CalcParameters["expected_incomes"]),
                           "Penalty for food shortage": args["rhoF"],
                           "Penalty for insolvency": args["rhoS"],
                           "Necessary debt (excluding food security constraint)": AddInfo_CalcParameters["necessary_debt"],
                           "Necessary debt (including food security constraint)": meta_sol["necessary_debt"],
                           "Probability for a catastrophic year": yield_information["prob_cat_year"],
                           "Share of samples with no catastrophe": yield_information["share_no_cat"],
                           "Share of years/clusters with unprofitable rice yields": yield_information["share_rice_np"],
                           "Share of years/clusters with unprofitable maize yields": yield_information["share_maize_np"],
                           "Share of West Africa's population that is living in currently considered region (2015)": \
                               population_information["pop_area_ratio2015"],
                           "On average cultivated area per cluster": list(np.nanmean(crop_alloc, axis = (0,1))),
                           "Average food demand penalty (over years and samples)": np.nanmean(meta_sol["fd_penalty"]),
                           "Average solvency penalty (over samples)": np.mean(meta_sol["sol_penalty"]),
                           "Average cultivation costs per cluster (over years and samples)": list(np.nanmean(meta_sol["yearly_fixed_costs"], axis = (0,1))),
                           "Expected total costs": meta_sol["exp_tot_costs"],
                           "Average food shortcomings (over years and samples)": np.nanmean(meta_sol["shortcomings"]),
                           "Number of occurrences per cluster where farmers make losses": list(meta_sol["num_years_with_losses"]),
                           "Average income per cluster in final run (over years and samples)": list(np.nanmean(meta_sol["profits"], axis = (0,1))),
                           "Average government payouts per cluster (over samples)": list(np.nanmean(np.nansum(meta_sol["payouts"], axis = 1), axis = 0)),
                           "Resulting probability for food security": meta_sol["probF"],
                           "Resulting probability for solvency": meta_sol["probS"],
                           "Resulting probability for food security for VSS": meta_sol_vss["probF"],
                           "Resulting probability for solvency for VSS": meta_sol_vss["probS"],
                           "Value of stochastic solution": VSS_value,
                           "Validation value (deviation of total penalty costs)": validation_values["deviation_penalties"]}
        
        current_panda = pd.read_csv("ModelOutput/Pandas/current_panda.csv")
        current_panda = current_panda.append(dict_for_pandas, ignore_index = True)
        current_panda.to_csv("ModelOutput/Pandas/current_panda.csv", index = False)
        
    return(None)

def SetUpNewPandas(name_old_pandas):
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

def CreateEmptyPanda():
    """
    Creating a new empty pandas object with the correct columns.

    Returns
    -------
    None.

    """
    
    with open("ModelOutput/Pandas/ColumnNames.txt", "rb") as fp:
        colnames = pickle.load(fp)
    
    new_panda = pd.DataFrame(columns = colnames)
    new_panda.to_csv("ModelOutput/Pandas/current_panda.csv", index = False)

    return(new_panda)

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
    
    
    with open("ModelOutput/Pandas/ColumnTypes.txt", "rb") as fp:
        dict_convert = pickle.load(fp)
        
    for key in dict_convert.keys():
        if dict_convert[key] == "list of floats":
            dict_convert[key] = __ConvertListsFloats
        elif dict_convert[key] == "list of ints":
            dict_convert[key] = __ConvertListsInts
        
    panda = pd.read_csv("ModelOutput/Pandas/current_panda.csv", converters = dict_convert)
    
    return(panda)


def ReadFromPandaSingleClusterGroup(file = "current_panda", 
                 output_var = None,
                 probF = 0.99,
                 probS = 0.95, 
                 rhoF = None,
                 rhoS = None,
                 k = 9,     
                 k_using = [3],
                 yield_projection = "fixed",   
                 sim_start = 2017,
                 pop_scenario = "fixed",
                 risk = 0.05,       
                 tax = 0.01,       
                 perc_guaranteed = 0.9,
                 ini_fund = 0,            
                 N = None, 
                 validation_size = None,
                 T = 25,
                 seed = 201120):
    
    if output_var is None:
        sys.exit("Please probide an output variable.")
    
    panda = OpenPanda(file = file)
    
    if type(output_var) is str:
        output_var = [output_var]
        
    
    output_var_fct = output_var.copy()
    output_var_fct.insert(0, "Used clusters")
    tmp = output_var_fct.copy()
    tmp.append("Sample size")
    tmp.append("Sample size for validation")
    sub_panda = panda[tmp]\
                    [list((panda.loc[:, "Input probability food security"] == probF) & \
                     (panda.loc[:, "Input probability solvency"] == probS) & \
                     (panda.loc[:, "Number of clusters"] == k) & \
                     (panda.loc[:, "Used clusters"] == k_using) & \
                     (panda.loc[:, "Yield projection"] == yield_projection) & \
                     (panda.loc[:, "Simulation start"] == sim_start) & \
                     (panda.loc[:, "Population scenario"] == pop_scenario) & \
                     (panda.loc[:, "Risk level covered"] == risk) & \
                     (panda.loc[:, "Tax rate"] == tax) & \
                     (panda.loc[:, "Share of income that is guaranteed"] == perc_guaranteed) & \
                     (panda.loc[:, "Initial fund size"] == ini_fund) & \
                     (panda.loc[:, "Number of covered years"] == T))]
                  
    # no results for these settings
    if sub_panda.empty:
        sys.exit("Requested data is not available.")
        
    # finding right sample size
    if N is not None:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size"] == N]
        # nor results for right sample siize
        if sub_panda.empty:
            sys.exit("Reyuested data is not available.")
        return(sub_panda)
        
    # results for highest sample size for these settings
    sub_panda = sub_panda[sub_panda["Sample size"] == max(sub_panda["Sample size"])]
    # if multiple runs for highest sample size, find highest validation sample size
    if len(sub_panda) == 1:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size for validation"] == \
                                          max(sub_panda["Sample size for validation"])]
    else:
        sub_panda = sub_panda[output_var_fct]
                
    return(sub_panda)
    
def ReadFromPanda(file = "current_panda", 
                 output_var = None,
                 k_using = [3],
                 **kwargs):
    
    if output_var is None:
        sys.exit("Please probide an output variable.")
    elif type(output_var) is str:
        output_var = [output_var]

        
    # prepare cluster groups
    if type(k_using) is tuple:
       k_using = [str(list(k_using))]
    elif (type(k_using) is list) and (type(k_using[0]) is not int):
        k_using = [str(list(k_using_tmp)) for k_using_tmp in k_using]
    elif type(k_using) is int:
        k_using = [str([k_using])]
    else:
        k_using = [str(k_using)]
    
    sub_panda = pd.DataFrame()
    for k_using_tmp in k_using:
        sub_panda = sub_panda.append(ReadFromPandaSingleClusterGroup(file = file, \
                                                        output_var = output_var, \
                                                        k_using = k_using_tmp, \
                                                        **kwargs))
        
    return(sub_panda)
            

def __ExtractResPanda(sub_panda, out_type, output_var, size):

    output_var_fct = output_var.copy()
    
    if out_type == "agg":
        output_var_fct.insert(0, "Group size")
        res = pd.DataFrame(columns = output_var_fct, index = [size])
        res.iloc[0,0] = size
        res.iloc[0,1:] = sub_panda[output_var].sum()
        res = res.add_suffix(" - Aggregated over all groups")
        res.rename(columns = {"Group size - Aggregated over all groups": \
                              "Group size"}, inplace = True)
        return(res)
    
    if out_type == "median":
        colnames = ["Group size"]
        for var in output_var_fct:
            colnames.append(var + " - Minimum")
            colnames.append(var + " - Median")
            colnames.append(var + " - Maximum")
        res = pd.DataFrame(columns = colnames, index = [size])
        res.iloc[0,0] = size
        for idx, var in enumerate(output_var_fct):
            res.iloc[0, idx*3 + 1] = sub_panda[var].min()
            res.iloc[0, idx*3 + 2] = sub_panda[var].median()
            res.iloc[0, idx*3 + 3] = sub_panda[var].max()
        return(res)

def PandaToPlot_GetResults(file = "current_panda", 
                           output_var = None,
                           out_type = "agg", # or median
                           grouping_aim = "Dissimilar",
                           adjacent = False,
                           **kwargs):
    
    add = ""
    if adjacent:
        add = "Adj"
       
    res = pd.DataFrame()
    
    for size in [1,2,3,5]:
        with open("InputData/Clusters/ClusterGroups/GroupingSize" \
                      + str(size) + grouping_aim + add + ".txt", "rb") as fp:
                BestGrouping = pickle.load(fp)
    
        panda_tmp = ReadFromPanda(file = file, \
                                  output_var = output_var, \
                                  k_using = BestGrouping, \
                                  **kwargs)
            
        res = res.append(__ExtractResPanda(panda_tmp, out_type, output_var, size))
            
    return(res)

def PlotPandaMedian(panda_file = "current_panda", 
                    output_var = None,
                    grouping_aim = "Dissimilar",
                    adjacent = False,
                    figsize = None,
                    subplots = True,
                    plt_file = None,
                    **kwargs):
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "rb") as fp:
        units = pickle.load(fp)
    
    res = PandaToPlot_GetResults(panda_file, output_var, "median", grouping_aim, adjacent, **kwargs)
    
    if output_var is str:
        output_var = [output_var]
    
    if subplots:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.35)
        rows = int(np.floor(np.sqrt(len(output_var))))
        cols = int(np.ceil(np.sqrt(len(output_var))))
    
    for idx, var in enumerate(output_var):
        if subplots:
            fig.add_subplot(rows, cols, idx + 1)
            plt.suptitle("Development depending on colaboration of clusters", \
                  fontsize = 24)
        else:
            fig = plt.figure(figsize = figsize)
            plt.title("Development depending on colaboration of clusters", \
                  fontsize = 24, pad = 15)
        plt.scatter([1, 2, 3, 4], res[var + " - Maximum"], marker = "^", label = "Maximum")
        plt.scatter([1, 2, 3, 4], res[var + " - Median"], marker = "X", label = "Median")
        plt.scatter([1, 2, 3, 4], res[var + " - Minimum"], label = "Minimum")
        plt.xticks([1, 2, 3, 4, 5], [9, 5, 3, 2, 1], fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("Number of different cluster groups", fontsize = 20)
        plt.ylabel(var + " " + units[var], fontsize = 20)
        plt.legend(fontsize = 20)
        
    if plt_file is not None:
        fig.savefig("Figures/PandaPlots/" + plt_file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    return(None)
    

def OverViewCurrentPandaVariables():
    
    with open("ModelOutput/Pandas/ColumnNames.txt", "rb") as fp:
        colnames = pickle.load(fp)
            
    return(colnames)
    
def SetUpPandaDicts():
    units = {"Input probability food security": "",
        "Input probability solvency": "",
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
        "Average food demand": "[$10^{12}\,kcal$]",
        "Import (excluding solvency constraint)": "[$10^{12}\,kcal$]",
        "Import (excluding solvency constraint, including theoretical export)": "[$10^{12}\,kcal$]",
        "Additional import needed when including solvency constraint": "[$10^{12}\,kcal$]",
        "Expected income (to calculate guaranteed income)": "[$10^9\,\$$]",
        "Penalty for food shortage": "[$\$/10^3\,kcal$]",
        "Penalty for insolvency": "[$\$/\$$]",
        "Necessary debt (excluding food security constraint)": "[$10^9\,\$$]",
        "Necessary debt (including food security constraint)": "[$10^9\,\$$]",
        "Probability for a catastrophic year": "",
        "Share of samples with no catastrophe": "",
        "Share of years/clusters with unprofitable rice yields": "",
        "Share of years/clusters with unprofitable maize yields": "",
        "Share of West Africa's population that is living in currently considered region (2015)": "",
        "On average cultivated area per cluster": "[$10^9\,ha$]",
        "Average food demand penalty (over years and samples)": "[$10^9\,\$$]",
        "Average solvency penalty (over samples)": "[$10^9\,\$$]",
        "Average cultivation costs per cluster (over years and samples)": "[$10^9\,\$$]",
        "Expected total costs": "[$10^9\,\$$]",
        "Average food shortcomings (over years and samples)": "[$10^{12}\,kcal$]",
        "Number of occurrences per cluster where farmers make losses": "",
        "Average income per cluster in final run (over years and samples)": "[$10^9\,\$$]",
        "Average government payouts per cluster (over samples)": "[$10^9\,\$$]",
        "Resulting probability for food security": "",
        "Resulting probability for solvency": "",
        "Resulting probability for food security for VSS": "",
        "Resulting probability for solvency for VSS": "",
        "Value of stochastic solution": "[$10^9\,\$$]",
        "Validation value (deviation of total penalty costs)": ""}    
    
    convert =  {"Input probability food security": float,
         "Input probability solvency": float,
         "Number of clusters": int,
         "Used clusters": str,
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
         "Import (excluding solvency constraint)": float,
         "Import (excluding solvency constraint, including theoretical export)": float,
         "Additional import needed when including solvency constraint": float,
         "Expected income (to calculate guaranteed income)": "list of floats",
         "Penalty for food shortage": float,
         "Penalty for insolvency": float,
         "Necessary debt (excluding food security constraint)": float,
         "Necessary debt (including food security constraint)": float,
         "Probability for a catastrophic year": float,
         "Share of samples with no catastrophe": float,
         "Share of years/clusters with unprofitable rice yields": float,
         "Share of years/clusters with unprofitable maize yields": float,
         "Share of West Africa's population that is living in currently considered region (2015)": \
             float,
         "On average cultivated area per cluster": "list of floats",
         "Average food demand penalty (over years and samples)": float,
         "Average solvency penalty (over samples)": float,
         "Average cultivation costs per cluster (over years and samples)": "list of floats",
         "Expected total costs": float,
         "Average food shortcomings (over years and samples)": float,
         "Number of occurrences per cluster where farmers make losses": "list of ints",
         "Average income per cluster in final run (over years and samples)": "list of floats",
         "Average government payouts per cluster (over samples)": "list of floats",
         "Resulting probability for food security": float,
         "Resulting probability for solvency": float,
         "Resulting probability for food security for VSS": float,
         "Resulting probability for solvency for VSS": float,
         "Value of stochastic solution": float,
         "Validation value (deviation of total penalty costs)": float}
        
    colnames = ['Input probability food security', 
        'Input probability solvency', 
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
        'Import (excluding solvency constraint)', 
        'Import (excluding solvency constraint, including theoretical export)', 
        'Additional import needed when including solvency constraint', 
        'Expected income (to calculate guaranteed income)', 
        'Penalty for food shortage', 
        'Penalty for insolvency', 
        'Necessary debt (excluding food security constraint)', 
        'Necessary debt (including food security constraint)', 
        'Probability for a catastrophic year', 
        'Share of samples with no catastrophe', 
        'Share of years/clusters with unprofitable rice yields', 
        'Share of years/clusters with unprofitable maize yields', 
        'Share of West Africa\'s population that is living in currently considered region (2015)', 
        'On average cultivated area per cluster', 
        'Average food demand penalty (over years and samples)', 
        'Average solvency penalty (over samples)', 
        'Average cultivation costs per cluster (over years and samples)', 
        'Expected total costs', 
        'Average food shortcomings (over years and samples)', 
        'Number of occurrences per cluster where farmers make losses', 
        'Average income per cluster in final run (over years and samples)', 
        'Average government payouts per cluster (over samples)', 
        'Resulting probability for food security', 
        'Resulting probability for solvency', 
        'Resulting probability for food security for VSS', 
        'Resulting probability for solvency for VSS', 
        'Value of stochastic solution', 
        'Validation value (deviation of total penalty costs)']
    
    with open("ModelOutput/Pandas/ColumnUnits.txt", "wb") as fp:
        pickle.dump(units, fp)
        
    with open("ModelOutput/Pandas/ColumnNames.txt", "wb") as fp:
        pickle.dump(colnames, fp)
        
    with open("ModelOutput/Pandas/ColumnTypes.txt", "wb") as fp:
        pickle.dump(convert, fp)
        
    return(None)