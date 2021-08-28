"""
Created on Sat Jan 23 16:15:20 2021

@author: Debbora Leip
"""
import pandas as pd
import os
import sys

from ModelCode.CompleteModelCall import LoadModelResults
from ModelCode.PandaGeneration import OpenPanda
from ModelCode.PandaGeneration import CreateEmptyPanda
from ModelCode.PandaGeneration import _WriteToPandas
from ModelCode.PandaGeneration import _SetUpPandaDicts
from ModelCode.Auxiliary import _GetDefaults

# %% ####### FUNCTIONS UPDATING AND ACCESSING THE RESULTS PANDA OBJECT #########

def UpdatePandaWithAddInfo(OldFile = "current_panda", console_output = None):
    """
    If additional variables were included in write_to_pandas and the 
    dictionaries in SetUpPandaDicts were extended accordingly, this function
    rewrites the panda object (including the new variables) by loading the full
    model results (using "Filename for full results") and applying the new
    write_to_pandas row for row.

    Parameters
    ----------
    OldFile : str, optional
        Name of the file containing the panda csv that needs to be updated.
        The default is "current_panda".
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.

    Returns
    -------
    None.

    """
    
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
        
    # load the panda that should be updated
    oldPanda = OpenPanda(file = OldFile)
    
    # updating the panda dicts
    os.remove("ModelOutput/Pandas/ColumnUnits.txt")
    os.remove("ModelOutput/Pandas/ColumnNames.txt")
    os.remove("ModelOutput/Pandas/ColumnTypes.txt")
    _SetUpPandaDicts()
    
    # temporarily move the current_panda to tmp to be able to work on a
    # current_panda
    CreateEmptyPanda(OldFile + "_updating")
    
    # going through the object row for row
    for i in range(0, len(oldPanda)):
        if console_output:
            sys.stdout.write("\r     Updating row " + str(i + 1) + " of " + str(len(oldPanda)))
            
        filename = oldPanda.at[i,'Filename for full results']
        
        # loading full results
        settings, args, yield_information, population_information, \
        status, all_durations, exp_incomes, crop_alloc, meta_sol, \
        crop_allocF, meta_solF, crop_allocS, meta_solS, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values = \
            LoadModelResults(filename)
            
        # applying updated write_to_pandas
        _WriteToPandas(settings, args, yield_information, population_information, \
                       status, all_durations, exp_incomes, crop_alloc, meta_sol, \
                       crop_allocF, meta_solF, crop_allocS, meta_solS, \
                       crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, \
                       filename, console_output = False, logs_on = False, file = OldFile + "_updating")

    # remove old panda file
    os.remove("ModelOutput/Pandas/" + OldFile + ".csv")
    
    # rename _update file to OldFile
    os.rename("ModelOutput/Pandas/" + OldFile + "_updating.csv", "ModelOutput/Pandas/" + OldFile + ".csv")

    return(None)

def _ReadFromPandaSingleClusterGroup(file = "current_panda", 
                                    output_var = None,
                                    probF = "default",
                                    probS = "default", 
                                    rhoF = "default",
                                    rhoS = "default",
                                    k = "default",     
                                    k_using = "default",
                                    num_crops = "default",
                                    yield_projection = "default",   
                                    sim_start = "default",
                                    pop_scenario = "default",
                                    risk = "default",       
                                    tax = "default",       
                                    perc_guaranteed = "default",
                                    ini_fund = "default",    
                                    food_import = "default",
                                    N = None, 
                                    validation_size = None,
                                    T = "default",
                                    seed = "default"):
    """
    Function returning a specific line (depending on the settings) of the 
    given panda csv for specific output variables. If N is not specified, it 
    uses the model run for the settings that used the highest sample size.

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : str or list of str
        A subset of the columns of the panda csv which should be returned.
    probF : float or None, optional
        demanded probability of keeping the food demand constraint. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    probS : float or None, optional
        demanded probability of keeping the solvency constraint. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    rhoF : float or None, optional
        The penalty for food shortages. The default is defined in
        ModelCode/DefaultModelSettings.py.
    rhoS : float or None, optional
        The penalty for insolvency. The default is defined in
        ModelCode/DefaultModelSettings.py.
    k : int, optional
        Number of clusters in which the area is devided. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    k_using : list of int i\in{1,...,k}, optional
        Specifies the clusters considered in the model. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    num_crops : int, optional
        Number of crops used in the model.
        The default is defined in ModelCode/DefaultModelSettings.py..
    yield_projection : "fixed" or "trend", optional
        Specifies the yield projection used in the model. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    sim_start : int, optional
        The first year of the simulation.
        The default is defined in ModelCode/DefaultModelSettings.py.
    pop_scenario : str, optional
        Specifies the population scenario used in the model.
        The default is defined in ModelCode/DefaultModelSettings.py.
    risk : int, optional
        The risk level that is covered by the government. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    tax : float, optional
        Tax rate to be paied on farmers profits.
        The default is defined in ModelCode/DefaultModelSettings.py.
    perc_guaranteed : float, optional
        The percentage that determines how high the guaranteed income is 
        depending on the expected income of farmers in a scenario excluding
        the government. The default is defined in ModelCode/DefaultModelSettings.py.
    ini_fund : float
        Initial fund size. The default is defined in ModelCode/DefaultModelSettings.py.   
    food_import : float, optional
        Amount of food that is imported (and therefore substracted from the
        food demand). The default is defined in ModelCode/DefaultModelSettings.py.
    N : int, optional
        Number of yield samples used to approximate the expected value
        in the original objective function. The default is defined in
        ModelCode/DefaultModelSettings.py.
    validation_size : None or int, optional
        The sample size used for validation. The default is defined in 
        ModelCode/DefaultModelSettings.py.
    T : int, optional
        Number of years to cover in the simulation. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    seed : int, optional
        Seed used for yield generation. The default is defined in 
        ModelCode/DefaultModelSettings.py.
        
    Returns
    -------
    sub_panda : panda dataframe
        subset of the panda dataframe according to the settings and the 
        specified output variables.

    """
        
    if output_var is None:
        sys.exit("Please provide an output variable.")
        
    # fill up missing settings with defaults
    PenMet, probF, probS, rhoF, rhoS, k, k_using, \
    num_crops, yield_projection, sim_start, pop_scenario, \
    risk, N, validation_size, T, seed, tax, perc_guaranteed, \
    ini_fund, food_import = _GetDefaults(None, probF, probS, rhoF, rhoS, k, k_using,
                num_crops, yield_projection, sim_start, pop_scenario,
                risk, N, validation_size, T, seed, tax, perc_guaranteed,
                ini_fund, food_import)
    
    
    # open data frame
    panda = OpenPanda(file = file)
    
    # either settings sepcify the probabilites or the penalties
    if (probF is not None) and (probS is not None):
        panda = panda[:][list((panda.loc[:, "Input probability food security"] == probF) & \
                     (panda.loc[:, "Input probability solvency"] == probS))]
    elif (rhoF is not None) and (rhoS is not None):
        panda = panda[:][list((panda.loc[:, "Penalty for food shortage"] == rhoF) & \
                     (panda.loc[:, "Penalty for insolvency"] == rhoS))]
        
    # cannot compare with list over full column -> as string
    panda["Used clusters"] = panda["Used clusters"].apply(str)

    # make sure the output_variables are given as list
    if type(output_var) is str:
        output_var = [output_var]
    
    # subset the data frame according to the settings and the output_variables,
    # keeping sample size and sample size for validation ...
    output_var_fct = output_var.copy()
    output_var_fct.insert(0, "Used clusters")
    tmp = output_var_fct.copy()
    tmp.append("Sample size")
    tmp.append("Sample size for validation")
    sub_panda = panda[tmp]\
                    [list((panda.loc[:, "Number of clusters"] == k) & \
                     (panda.loc[:, "Used clusters"] == str(k_using)) & \
                     (panda.loc[:, "Number of crops"] == num_crops) & \
                     (panda.loc[:, "Yield projection"] == yield_projection) & \
                     (panda.loc[:, "Simulation start"] == sim_start) & \
                     (panda.loc[:, "Population scenario"] == pop_scenario) & \
                     (panda.loc[:, "Risk level covered"] == risk) & \
                     (panda.loc[:, "Tax rate"] == tax) & \
                     (panda.loc[:, "Share of income that is guaranteed"] == perc_guaranteed) & \
                     (panda.loc[:, "Initial fund size"] == ini_fund) & \
                     (panda.loc[:, "Number of covered years"] == T) & \
                     (panda.loc[:, "Import (given as model input)"] == food_import))]
                  
    # no results for these settings?
    if sub_panda.empty:
        sys.exit("Requested data is not available.")
        
    # ... subset sample size here if given ...
    if N is not None:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size"] == N]
        # no results for right sample size?
        if sub_panda.empty:
            sys.exit("Requested data is not available for this sample size.")
        return(sub_panda)
        
    # ... else use the results for highest sample size for these settings
    sub_panda = sub_panda[sub_panda["Sample size"] == max(sub_panda["Sample size"])]
    # if multiple runs for highest sample size, find highest validation sample size
    if (len(sub_panda) > 1) and (validation_size is None):
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size for validation"] == \
                                          max(sub_panda["Sample size for validation"])]
    elif validation_size is not None:
        sub_panda = sub_panda[output_var_fct][sub_panda["Sample size for validation"] == \
                                                              validation_size]
        if sub_panda.empty:
            sys.exit("Requested data is not available.")
    else:
        sub_panda = sub_panda[output_var_fct]
                       
    # turn used clusters back from strings to lists
    def _ConvertListsInts(arg):
        arg = arg.strip("][").split(", ")
        res = []
        for j in range(0, len(arg)):
            res.append(int(arg[j]))
        return(res)

    sub_panda["Used clusters"] = sub_panda["Used clusters"].apply(_ConvertListsInts)
        
    return(sub_panda)
    
def ReadFromPanda(file = "current_panda", 
                  output_var = None,
                  k_using = [3],
                  **kwargs):
    """
    Function returning a subset of the given panda dataframe, according to the
    cluster groups, settings, and output variables specified

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    output_var : str or list of str
        A subset of the columns of the panda csv which should be returned.
    k_using : tuple of ints, list of tuples of ints, int, list of ints, optional
        Either one group of clusters or a set of different cluster groups for
        which results shall be returned. The default is [3].
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to _ReadFromPandaSingleClusterGroup.

    Returns
    -------
    subpanda : panda dataframe
        subset of the given panda dataframe, according to the cluster groups, 
        settings, and output variables specified

    """
    
    # check given output_variables
    if output_var is None:
        sys.exit("Please probide an output variable.")
    elif type(output_var) is str:
        output_var = [output_var]
        
    # prepare cluster groups
    if type(k_using) is tuple:
       k_using = [sorted(list(k_using))]
    elif (type(k_using) is list) and (type(k_using[0]) is not int):
        k_using = [sorted(list(k_using_tmp)) for k_using_tmp in k_using]
    elif type(k_using) is int:
        k_using = [[k_using]]
    else:
        k_using = [sorted(k_using)]
    
    # set up panda cluster group per cluster grop
    sub_panda = pd.DataFrame()
    for k_using_tmp in k_using:
        sub_panda = sub_panda.append(_ReadFromPandaSingleClusterGroup(file = file, \
                                                        output_var = output_var, \
                                                        k_using = k_using_tmp, \
                                                        **kwargs))
        
    return(sub_panda)
            
def LoadFullResults(file = "current_panda", **kwargs):
    """
    Function returning the full model results for specific settings, taking
    the run with the highest sample size if N is not specified

    Parameters
    ----------
    file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    **kwargs : 
        Settings specifiying for which model run results shall be returned, 
        passed to ReadFromPanda
        
    Returns
    -------
    Everything returned when running the model (FoodSecurityProblem).

    """
    
    fn = ReadFromPanda(file = file,
                       output_var = "Filename for full results",
                       **kwargs)
    
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values = \
                LoadModelResults(fn["Filename for full results"].iloc[0])
                
    return(settings, args, yield_information, population_information, \
           status, all_durations, exp_incomes, crop_alloc, meta_sol, \
           crop_allocF, meta_solF, crop_allocS, meta_solS, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values)