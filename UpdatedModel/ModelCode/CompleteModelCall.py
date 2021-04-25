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
import matplotlib.pyplot as plt
import math

from ModelCode.SetFolderStructure import CheckFolderStructure
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import GetFilename
from ModelCode.PandaGeneration import write_to_pandas
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
                        save = True, plotTitle = None, close_plots = None, 
                        panda_file = "current_panda", **kwargs):
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
        with that title and saved to Figures/CropAllocs/numclusters/.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    panda_file : str
        Name of the csv file used to append the results of the model run for
        plotting and easy accessibility. If None, results are not saved to 
        panda csv. Default is "current_panda"
    **kwargs
        settings for the model, passed to DefaultSettingsExcept()
        
    Returns
    -------
    settings : dict
        The model input settings that were given by user. 
    args : dict
        Dictionary of arguments needed as direct model input.
    AddInfo_CalcParameters : dict
        Additional information from calculating expected income and penalties
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
        'fix_costs', 'yearly_fixed_costs', 'fd_penalty', 'avg_fd_penalty', 
        'sol_penalty', 'shortcomings', 'exp_shortcomings', 'expected_incomes', 
        'profits', 'num_years_with_losses', 'payouts', 'final_fund', 'probF', 
        'probS', 'avg_nec_import', 'avg_nec_debt')
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
    fn : str
        all settings combined to a single file name to save/load results and 
        for log file
    """    
    
    # set up folder structure (if not already done)
    CheckFolderStructure()
        
    # defining settings
    settings = DefaultSettingsExcept(**kwargs)
    
    # get filename for model results
    fn = GetFilename(settings)
    
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
                                              plotTitle = plotTitle,
                                              panda_file = panda_file)
        except KeyboardInterrupt:
            print(colored("\nThe model run was interupted by the user.", "red"), flush = True)
            import ModelCode.GeneralSettings as GS
            if logs_on or (logs_on is None and GS.logs_on):                
                log = open("ModelLogs/" + GS.fn_log + ".txt", "a")
                log.write("\n\nThe model run was interupted by the user.")
                log.close()
                del GS.fn_log
        
    # if it does, it is loaded from output file
    else:            
        printing("Loading results", console_output = console_output, logs_on = False)
        
        settings, args, AddInfo_CalcParameters, yield_information, \
        population_information, status, all_durations, crop_alloc, meta_sol, \
        crop_alloc_vss, meta_sol_vss, VSS_value, validation_values = \
            LoadModelResults(fn)
        
    # if a plottitle is provided, crop allocations over time are plotted
    if plotTitle is not None:        
        PlotCropAlloc(crop_alloc = crop_alloc, k = settings["k"], k_using = settings["k_using"], 
                      max_areas = args["max_areas"], close_plots = close_plots,
                      title = plotTitle, file = fn)
    
    return(settings, args, AddInfo_CalcParameters, yield_information, \
           population_information, status, all_durations, crop_alloc, meta_sol, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn)          

def OptimizeModel(settings, panda_file, console_output = None, logs_on = None, \
                  save = True, plotTitle = None):
    """
    Function combines setting up and solving the model, calculating additional
    information, and saving the results.

    Parameters
    ----------
    settings : dict
        All input settings for the model framework.
    panda_file : str or None
        Name of the csv file used to append the results of the model run for
        plotting and easy accessibility. If None, results are not saved to 
        panda csv. Default is "current_panda"
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log file.
        If None, the default as defined in ModelCode/GeneralSettings is used.
    save : boolean, optional
        whether the results should be saved (does not refer to the panda results).
        The default is True.
    plotTitle : str or None
        If not None, a plot of the resulting crop allocations will be made 
        with that title and saved to Figures/CropAllocs.

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
        'fix_costs', 'yearly_fixed_costs', 'fd_penalty', 'avg_fd_penalty', 
        'sol_penalty', 'shortcomings', 'exp_shortcomings', 'expected_incomes', 
        'profits', 'num_years_with_losses', 'payouts', 'final_fund', 'probF', 
        'probS', 'necessary_import', 'necessary_debt')
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
    """
    
    # timing
    all_start  = tm.time()
    all_durations = {}
    
    # file name used for log file and saving results
    fn = GetFilename(settings)
    
    # initialize log file
    if logs_on is None:
        from ModelCode.GeneralSettings import logs_on
    
    if logs_on:
        import ModelCode.GeneralSettings as GS
        GS.fn_log = fn
        
        if os.path.exists("ModelLogs/" + GS.fn_log  + ".txt"):
            i = 1
            while os.path.exists("ModelLogs/" + GS.fn_log  + "_" + str(i) + ".txt"):
                i += 1
            GS.fn_log = GS.fn_log + "_" + str(i)
            
        log = open("ModelLogs/" + GS.fn_log + ".txt", "a")
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
    printing("\nGetting parameters", console_output = console_output, logs_on = logs_on)
    args, yield_information, population_information = \
                    SetParameters(settings, exp_incomes)
    
    # get the right penalties
    penalties_start  = tm.time()
    if settings["PenMet"] == "prob":
        rhoF, rhoS, probF_onlyF, probS_onlyS, import_onlyF, debt_onlyS = \
            GetPenalties(settings, args, console_output = console_output, logs_on = logs_on)
            
        args["rhoF"] = rhoF
        args["rhoS"] = rhoS
        
        AddInfo_CalcParameters["probF_onlyF"] = probF_onlyF
        AddInfo_CalcParameters["probS_onlyS"] = probS_onlyS
        AddInfo_CalcParameters["import_onlyF"] = import_onlyF
        AddInfo_CalcParameters["debt_onlyS"] = debt_onlyS
    else:
        args["rhoF"] = settings["rhoF"]
        args["rhoS"] = settings["rhoS"]
        
        AddInfo_CalcParameters["probF_onlyF"] = None
        AddInfo_CalcParameters["probS_onlyS"] = None
        AddInfo_CalcParameters["import_onlyF"] = None
        AddInfo_CalcParameters["debt_onlyS"] = None
        
    penalties_end  = tm.time()
    all_durations["GetPenalties"] = penalties_end - penalties_start
    
        
    # run the optimizer
    status, crop_alloc, meta_sol, prob, durations = \
        SolveReducedcLinearProblemGurobiPy(args, \
                                           console_output = console_output, \
                                           logs_on = logs_on)
    all_durations["MainModelRun"] = durations[2]
        
    printing("\nResulting probabilities:\n" + \
            "     probF: " + str(np.round(meta_sol["probF"]*100, 2)) + "%\n" + \
            "     probS: " + str(np.round(meta_sol["probS"]*100, 2)) + "%", 
            console_output = console_output,
            logs_on = logs_on)
        
    # VSS
    vss_start  = tm.time()
    printing("\nCalculating VSS", console_output = console_output, logs_on = logs_on)
    crop_alloc_vss, meta_sol_vss = VSS(settings, exp_incomes, args)
    VSS_value = meta_sol_vss["exp_tot_costs"] - meta_sol["exp_tot_costs"]
    vss_end  = tm.time()
    all_durations["VSS"] = vss_end - vss_start
    
    # out of sample validation
    validation_start  = tm.time()
    if settings["validation_size"] is not None:
        printing("\nOut of sample validation", console_output = console_output, logs_on = logs_on)
        validation_values = OutOfSampleVal(crop_alloc, settings, exp_incomes, args["rhoF"], \
                              args["rhoS"], meta_sol, args["probS"], console_output)
    validation_end  = tm.time()
    all_durations["Validation"] = validation_end - validation_start

    # add results to pandas overview
    fn = GetFilename(settings)
    if panda_file is not None:
        write_to_pandas(settings, args, AddInfo_CalcParameters, yield_information, \
                        population_information, crop_alloc, \
                        meta_sol, meta_sol_vss, VSS_value, validation_values, \
                        fn, console_output, panda_file)     
            
    # timing
    all_end  = tm.time()   
    full_time = all_end - all_start
    printing("\nTotal time: " + str(np.round(full_time, 2)) + "s", 
             console_output = console_output, logs_on = logs_on)
    all_durations["TotalTime"] = full_time
       
    
    # saving results
    if save:
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
     
    # remove the global variable fn_log
    if logs_on:
        del GS.fn_log
        
    return(settings, args, AddInfo_CalcParameters, yield_information, \
           population_information, status, all_durations, crop_alloc, meta_sol, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values)          


def LoadModelResults(filename):
    """
    Loads results of a saved model run.

    Parameters
    ----------
    filename : str
        Filenae from which to load model results.

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
        'fix_costs', 'yearly_fixed_costs', 'fd_penalty', 'avg_fd_penalty', 
        'sol_penalty', 'shortcomings', 'exp_shortcomings', 'expected_incomes', 
        'profits', 'num_years_with_losses', 'payouts', 'final_fund', 'probF', 
        'probS', 'necessary_import', 'necessary_debt')
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

    """
    # load results
    with open("ModelOutput/SavedRuns/" + filename + ".txt", "rb") as fp:
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
            
    # calculate meta_sols
    meta_sol = GetMetaInformation(crop_alloc, args, args["rhoF"], args["rhoS"])
    meta_sol_vss =  GetMetaInformation(crop_alloc_vss, args, args["rhoF"], args["rhoS"])
    
    return(settings, args, AddInfo_CalcParameters, yield_information, \
           population_information, status, all_durations, crop_alloc, meta_sol, \
           crop_alloc_vss, meta_sol_vss, VSS_value, validation_values) 
        
        
def PlotCropAlloc(crop_alloc, k, k_using, max_areas, cols = None, cols_b = None, \
                  figsize = None, close_plots = None, title = None, file = None, sim_start = 2017):
    """
    Plots crop area allocations over the years for all given clusters.

    Parameters
    ----------
    crop_alloc : np.array of size (T*num_crops*len(k_using),)
        Gives allocation of area to each crop in each cluster.
    k : int, optional
        Number of clusters in which the area is to be devided. 
    k_using :  "all" or a list of int
        Specifies which of the clusters are to be considered in the model. 
        The default is "all".
    max_areas : np.array of size (len(k_using),) 
        Upper limit of area available for agricultural cultivation in each
        cluster
    cols : list, optional
        List of colors to use for plotting. If None, default values will
        be used. The default is None.
    cols_b : list, optional
        Lighter shades of the colors to use for plotting. If None, default
        values will be used. The default is None.
    figsize : tuple, optional
        The figure size. The default is defined at the top of the document.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    title : str
        Basis of title for for the plot (will be completed with information on
        the clusters). The default is None.
    file : str
        Filename to save resulting figure. If None, figure will not be saved.
        The default is None.
    sim_start : int
        The first year of the simulation. The default is 2017.

    Returns
    -------
    None.

    """
    # get plotting settings
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
    
    if close_plots is None:
        from ModelCode.GeneralSettings import close_plots
        
    # defining colors
    if cols is None:            
        cols = ["royalblue", "darkred", "grey", "gold", \
                "limegreen", "darkturquoise", "darkorchid", "seagreen", 
                "indigo"]
    if cols_b is None:
        cols_b = ["dodgerblue", "red", "darkgrey", "khaki", \
              "lime", "cyan", "orchid", "lightseagreen", "mediumpurple"]     
            
    # set up title        
    if title is None:
        title = ""
    else:
        title = " - " + title
        
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    [T, J, K] = crop_alloc.shape
    years = range(sim_start, sim_start + T)
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.95,
                    wspace=0.15, hspace=0.35)
    
    # overview of all crop area allocations in one plot
    if K > 1:
        ax = fig.add_subplot(1,2,1)
        l0, = plt.plot(years, np.sum(crop_alloc, axis = (1,2)), color = "k", \
                 lw = 2, alpha = 0.7, ls = "-.")
    else:
        ax = fig.add_subplot(1,1,1)
    for cl in range(0, K):
        l1, = plt.plot(years, np.repeat(max_areas[cl], len(years)), \
                 color = cols_b[k_using[cl]-1], lw = 5, alpha = 0.4)
        l2, = plt.plot(years, crop_alloc[:,0,cl], color = cols[k_using[cl]-1], \
                 lw = 1.8, linestyle = "--")
        l3, = plt.plot(years, crop_alloc[:,1,cl], color = cols[k_using[cl]-1], \
                 lw = 1.8, label = "Cluster " + str(k_using[cl]))
    plt.xlim(years[0] - 0.5, years[-1] + 0.5)
    # plt.ylim(-0.05 * np.max(max_areas), 1.1 * np.max(max_areas))
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.yaxis.offsetText.set_fontsize(24)
    ax.xaxis.offsetText.set_fontsize(24)
    plt.xlabel("Years", fontsize = 26)
    plt.ylabel(r"Crop area in [$10^6$ ha]", fontsize = 26)
    if K > 2:
        plt.suptitle(str(k_using) + title, fontsize = 30)
    elif K == 1:
        plt.title("Cluster " + str(k_using[0]) + title, \
                  fontsize = 26, pad = 10)
    elif K == 2:
        plt.title("Cluster " + str(k_using[0]) + " and " + str(k_using[1]) + title, \
                 fontsize = 26, pad = 10)
    if K == 1:
        legend1 = plt.legend([l1, l2, l3], ["Maximum available area", \
                                            "Area Rice", "Area Maize"], \
                             fontsize = 20, loc = 2)
    else:
        legend1 = plt.legend([l0, l1, l2, l3], ["Total cultivated area", \
                                                "Maximum available area", \
                                                "Area Rice", "Area Maize"], \
                             fontsize = 20, loc = 2)
        plt.legend(loc='lower left', ncol=K, borderaxespad=0.6, fontsize = 16)
    plt.gca().add_artist(legend1)
    
    # crop area allocations in separate subplots per cluster
    if K > 2:
        rows = math.ceil(K/2)
        whichplots = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
        for cl in range(0, K):
            ax = fig.add_subplot(rows, 4, whichplots[cl])
            plt.plot(years, np.repeat(max_areas[cl], len(years)), \
                     color = cols_b[k_using[cl]-1], lw = 5, alpha = 0.4)
            plt.plot(years, crop_alloc[:,0,cl], color = cols[k_using[cl]-1], \
                     lw = 1.8, linestyle = "--")
            plt.plot(years, crop_alloc[:,1,cl], color = cols[k_using[cl]-1], \
                     lw = 1.8, label = "Cluster " + str(k_using[cl]))
            plt.ylim(-0.05 * np.max(max_areas), 1.1 * np.max(max_areas))
            plt.xlim(years[0] - 0.5, years[-1] + 0.5)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.yaxis.offsetText.set_fontsize(16)
            ax.xaxis.offsetText.set_fontsize(16)
    
    
    if file is not None:
        if not os.path.isdir("Figures/CropAllocs/" + str(K) + "clusters"):
            os.mkdir("Figures/CropAllocs/" + str(K) + "clusters") 
        fig.savefig("Figures/CropAllocs/" + str(K) + "clusters/CropAlloc_" + file + ".jpg", bbox_inches = "tight", pad_inches = 1)
    
    if close_plots:
        plt.close()
        
    return(None)