#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:17:52 2020

@author: debbora
"""

from os import chdir 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

chdir('/home/debbora/IIASA/FinalVersion')
#chdir("H:\FinalVersion")

import Functions_StochasticOptimization as OF


figsize = (24, 13.5)            
cols = ["royalblue", "darkred", "darkgrey", "gold", \
        "limegreen", "darkturquoise", "darkorchid"]
cols_b = ["dodgerblue", "red", "gainsboro", "yellow", \
          "lime", "cyan", "orchid"]
years = range(2017, 2042)
T_max = 25
yield_year = 2017

# %% ################## STOCHASTIC OPTIMIZATION SINGLE YEAR ###################

# %% Parameters
#
#k = 1
#num_realisations = 10000
#year = 2016
#seed = 310520
#
## load clutsering
#with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
#                    str(k) + "opt_PearsonDist_spei03.txt", "rb") as fp:  
#    clusters = pickle.load(fp)
#    costs = pickle.load(fp)
#    medoids = pickle.load(fp) 
#cluster = clusters[-1]
#
## load information on yield distributions, using historic yield data from GDHY  
#with open("IntermediateResults/LinearRegression/GDHY/DetrYieldAvg_k" + \
#                          str(k) + ".txt", "rb") as fp:   
#     yields_avg = pickle.load(fp)
#     avg_pred = pickle.load(fp)
#     residuals = pickle.load(fp)
#     residual_means = pickle.load(fp)
#     residual_stds = pickle.load(fp)
#     fstat = pickle.load(fp)
#     constants = pickle.load(fp)
#     slopes = pickle.load(fp)
#     crops = pickle.load(fp)
#     
#num_crops = len(crops)
#
## calculate area proportions of the different clusters
#area_proportions = np.zeros(k)
#for cl in range(0, k):
#    area_proportions[cl] = np.sum(cluster == (cl + 1))/ \
#                                    (np.sum(~np.isnan(cluster)))
#    
## cultivation costs for different crops in different clusters
#costs = np.ones([num_crops, k])
## energetic value of crops
#crop_cal = np.array([1,1])
## food demand
#demand = 100
## total available agricultural area
#maxarea = 400
## available agricultural area per cluster
#max_areas = maxarea * area_proportions
## initial search radius for cobyla
#rhobeg_cobyla = 10
## food demand penalties to use
#rhoFs = [50]
## initial guess for crop allocations
#x_ini = np.tile(max_areas.flatten()/(num_crops + 1), num_crops)
#
## Linear Constraint: total allocated crop area within each cluster can't be 
##                    more than max agricultural area
#iden = np.identity(k)
#const = np.tile(iden, num_crops)
#
#def const_cobyla1(x, max_areas, const):
#    return((max_areas - const.dot(x)))
#def const_cobyla2(x, max_areas, const):
#    return(x) 
#constraints = [const_cobyla1, const_cobyla2]
#
#
#    
## %% Running Optimizer
#np.random.seed(seed)   
#ylds = OF.YieldRealisationsSingleYear(slopes, constants, residual_stds, \
#                                      year, num_realisations, k, num_crops)
#
#for rhoF in rhoFs:
#    print("Food demand penalty " + str(rhoF) + ": ")
#    t_b = tm.time()
#    crop_alloc = OF.OptimizeSingleYear(x_ini, constraints, 
#                      num_realisations, max_areas, const,
#                      costs, rhoF, demand, crop_cal,
#                      num_crops, k, seed, rhobeg_cobyla, ylds)
#    t_e = tm.time()
#    print("Time of total optimization", t_e-t_b, "\n")
#    
#    all_settings = {"k": k, 
#                    "costs": costs, 
#                    "num_realisations": num_realisations,
#                    "year": year, 
#                    "seed": seed, 
#                    "crop_cal": crop_cal, 
#                    "demand": demand, 
#                    "maxarea": maxarea, 
#                    "rhobeg_cobyla": rhobeg_cobyla, 
#                    "rhoF": rhoF, 
#                    "x_ini": x_ini}
#    
#    info_solver = {"TotalTime": t_e-t_b, 
#                   "ObjFct_Calls": OF.ObjectiveFunctionSingleYear.Calls}
#    
#    with open("StoOptSingleYear/StoOpt_k" +str(k) + \
#                                  "_rhoF" + str(rhoF) + ".txt", "wb") as fp: 
#            pickle.dump(crop_alloc, fp) 
#            pickle.dump(all_settings, fp)
#            pickle.dump(info_solver, fp)


###############################################################################
# %% ################## STOCHASTIC OPTIMIZATION WITH TIME #####################
###############################################################################

# %% #################### 0. EXPLANATION OF SETTINGS ##########################

# These are the default setting, which will be set by the function call 
# settings, names = OF.DefaultSettingsExcept()
# settings will then be a dictionary of the following default settings, while 
# names includes two settingnames as meta info for solving results.
# If the function OF.DefaultSettingsExcept is called with any arguments, these 
# will be used instead of the default values for the respectvie parameters.

# choose number of clusters
k = 1
# how many clusters have to be catastrophic to have a catastrophic year?
num_cl_cat = 1 
# should yield distributions be "fixed" over time or follow the linear "trend"?
yield_projection = "fixed"
# which year should the simulation start in?
yield_year = 2017  # will use values of 2016 for fixed versions
# should the stilised values be used, or the more realistic values?
stilised = False
# if using realistic values: which UN_WPP population scenario should be uses?
# 'Medium', 'High', 'Low', 'ConstantFertility', 'InstantReplacement', 
# 'ZeroMigration', 'ConstantMortality', 'NoChange', 'Momentum' from UN 
# population prospects or 'fixed' which uses population value of yield_year - 1 
# for all years. All scenarios have the same estimates up to (including) 2019,
# scenariospecific predictions start from 2020
pop_scenario = "fixed"
# what should the risk level be? (one in how-many-years event)
risk = 20
# how many realizations should be used
N_c = 3500
# what should the maximum number of years the simulation runs be?
T_max = 25
# what seed should be used for the yield realizations?
seed = 150620   
# tax on profits
tax = 0.03
# how much of expected income should be guaranteed by government? (in realstic
# version. in stilised it is a fixed value anyway)
perc_guaranteed = 0.75      
expected_income = 0 # will be calculated from rhoF = 0.9, rhoS = 0 case 

# crops are rice and maize in that order (i.e. rice is crop 0, maize is crop 1)


    
# %% ######################  1. ANALYZING SAMPLE SIZE #########################
    
# %% a) find reasonable penalties for the default setting, which to use in the 
#       N_c analysis:

probF = 0.8
probS = 0.9

np.warnings.filterwarnings('ignore')
crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
            OF.OptimizeFoodSecurityProblem( probF, probS, rhoFini = 1e-3, \
                                           rhoSini = 287.5, N_c = 10000)
         
with open("StoOptMultipleYears/N_c/penalties_for_NcAnalysis.txt", "wb") as fp: 
    pickle.dump(["settings", "penalties", "probs", \
                 "crop_alloc", "meta_sol"], fp)
    pickle.dump(settings, fp)
    pickle.dump([rhoF, rhoS], fp)   # 0.00025
    pickle.dump([probF, probS], fp)   # 287.5
    pickle.dump(crop_alloc, fp)
    pickle.dump(meta_sol, fp)        
    
# %% b) for different values of sample size N_c run the model for  50 different 
#    seeds each 
         
with open("StoOptMultipleYears/N_c/penalties_for_NcAnalysis.txt", "rb") as fp: 
    meta = pickle.load(fp)
    settings = pickle.load(fp)
    [rhoF, rhoS] = pickle.load(fp)
    [probF, probS] = pickle.load(fp)
    crop_alloc = pickle.load(fp)
    meta_sol = pickle.load(fp)
    
# using the expected income that was calculated above, as it would else be
# recalculated for each N_c
expected_income = settings["expected_incomes"]    
    
Ncs = [10, 25, 50, 75, 100, \
       150, 200, 300, 500, 750, \
       1000, 1500, 2000, 2500, 3500, \
       5000, 6000, 7500, 10000, 15000, \
       30000]

np.warnings.filterwarnings('ignore')    
for n in Ncs:
    print("\n" + str(n) + ": ")
    exp_total_costs = []
    crop_allocs = []
    for i in range(0, 50):
        settings = OF.DefaultSettingsExcept(N_c = n, seed = i, \
                                            exp_income = expected_income)
        x_ini, const, args, meta_cobyla, other = OF.SetParameters(settings) 
        crop_alloc, meta_sol, duration = \
                    OF.OptimizeMultipleYears(x_ini, const, args,
                                             meta_cobyla, rhoF, rhoS)  
        exp_total_costs.append(meta_sol["exp_tot_costs"])
        crop_allocs.append(crop_alloc)
        print(str(i) + ", duration: " + str(duration))
             
    with open("StoOptMultipleYears/N_c/N_c" + str(n) + ".txt", "wb") as fp: 
        pickle.dump(["exp_total_costs", "n", "seeds", "crop_allocs"], fp)
        pickle.dump(exp_total_costs, fp)
        pickle.dump(n, fp)   
        pickle.dump(range(0, 50), fp)    
        pickle.dump(crop_allocs, fp)


# %% c) Visualize the results of runs for different sample size
        
Ncs = [10, 25, 50, 75, 100, \
       150, 200, 300, 500, 750, \
       1000, 1500, 2000, 3500, 5000, \
       6000, 7500, 10000, 15000, 30000]
Ncs = np.array(Ncs)    

# c.1) Relative standard deviation of the total costs (minimum of the objective
#      function)
std_normalized = []
fig = plt.figure(figsize = (20.8, 11.7))
title = ["A", "B"]
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                    wspace=0.3, hspace=0.3)
ax = fig.add_subplot(1,2,1)
for n in Ncs:
    with open("StoOptMultipleYears/N_c/N_c" + str(n) + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        exp_total_costs = pickle.load(fp)
        n = pickle.load(fp)
        seeds = pickle.load(fp)
    std_normalized.append(np.std(exp_total_costs)/np.mean(exp_total_costs))
plt.scatter(Ncs, std_normalized, s = 60, color = cols[0])
plt.plot([-500,30500], [0.01,0.01], color = cols[1], linewidth = 3)
plt.ylabel("RSD of minimum costs", fontsize = 25)
plt.xlabel(r"Sample size $N$", fontsize = 25)
plt.ylim([-0.003, 0.25])
plt.xlim([-500, 30500])
plt.title(title[0], fontsize = 28)
ax.xaxis.set_tick_params(labelsize = 20)
ax.yaxis.set_tick_params(labelsize = 20)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.xaxis.offsetText.set_fontsize(20)

# find lowest sample size with RSD below 1%
std_normalized = np.array(std_normalized)
Ncs_accurate = Ncs[std_normalized < 0.01]
np.min(Ncs_accurate)            # 3500


# c.2) Relative standard deviation of the crop allocations (argmin of the 
#      objective function)
crop0 = np.zeros([50, T_max])
crop1 = np.zeros([50, T_max])
std_normalized_crop0 = np.zeros([len(Ncs), T_max])
std_normalized_crop1 = np.zeros([len(Ncs), T_max])
ax = fig.add_subplot(1,2,2)
for idx, n in enumerate(Ncs):
    with open("StoOptMultipleYears/N_c/N_c" + str(n) + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        exp_total_costs = pickle.load(fp)
        n = pickle.load(fp)
        seeds = pickle.load(fp)
        crop_allocs = pickle.load(fp)
    for t in range(0, T_max):
        for i in range(0, len(seeds)):
            crop0[i, t] = crop_allocs[i][t,0,0]
            crop1[i, t] = crop_allocs[i][t,1,0]
        std_normalized_crop0[idx, t] = np.std(crop0[:, t])/np.mean(crop0[:, t])
        std_normalized_crop1[idx, t] = np.std(crop1[:, t])/np.mean(crop1[:, t])
plt.scatter(Ncs, np.mean(std_normalized_crop1, axis =1), \
            s = 60, color = cols[0])
plt.plot([-500,30500], [0.01,0.01], color = cols[1], linewidth = 3)
plt.ylabel("Average RSD of crop allocations", fontsize = 25)
plt.xlabel(r"Sample size $N$", fontsize = 25)
plt.ylim([-0.003, 0.25])
plt.xlim([-500, 30500])
plt.title(title[1], fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.xaxis.offsetText.set_fontsize(20)

# find lowest sample size with RSD below 1%
Ncs_accurate = Ncs[np.mean(std_normalized_crop1, axis =1)<0.01]
np.min(Ncs_accurate)            # 750
plt.show()

fig.savefig("Figures/StoOpt/N_c/AccuracySampleSize.png", \
            bbox_inches = "tight", pad_inches = 0.5)  


# %% c.3) How does STD for crop allocation for the cosen sample size N_c = 3500
#         change over the period of time?

with open("StoOptMultipleYears/N_c/N_c3500.txt", "rb") as fp: 
    meta = pickle.load(fp)
    exp_total_costs = pickle.load(fp)
    n = pickle.load(fp)
    seeds = pickle.load(fp)
    crop_allocs = pickle.load(fp)
    
crop_allocs = np.array(crop_allocs)
crop_allocs_means = np.mean(crop_allocs, axis = 0)
crop_allocs_stds = np.std(crop_allocs, axis = 0)

labels = ["Rice", "Maize"]
fig = plt.figure(figsize = figsize)
ax = fig.add_subplot(1, 1, 1)
for i in [0, 1]:
    plt.plot(range(yield_year, yield_year + T_max), \
             crop_allocs_means[:, i, 0], color = cols[i], \
             label = labels[i], linewidth = 2)
    plt.fill_between(range(yield_year, yield_year + T_max), \
                     crop_allocs_means[:, i, 0] + crop_allocs_stds[:, i, 0] , \
                     crop_allocs_means[:, i, 0] - crop_allocs_stds[:, i, 0] , \
                     color = cols[i], alpha = 0.5)

plt.ylabel("Crop allocation as given by model", fontsize = 18)
plt.xlabel("Year", fontsize = 18)
plt.legend(fontsize = 18, fancybox = True)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
plt.show()

fig.savefig("Figures/StoOpt/N_c/STD_CropAlloc_Nc3500.png", \
            bbox_inches = "tight", pad_inches = 0.5)  


###############################################################################

# %% ##############  2. ANALYZING RELATION BETWEEN RHO AND ALPHA ##############

# %% a.1) Run default settings for different rhoF with rhoS = 0

np.warnings.filterwarnings('ignore')
settings  = OF.DefaultSettingsExcept()
x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)

rhoS = 0
rhoFs = np.flip(np.array([4e-2, 3e-2, 2e-2, 1e-2, 75e-4, \
                          5e-3, 3e-3, 2e-3, 1e-3, 75e-5, \
                          5e-4, 4e-4, 3e-4, 2e-4, 15e-5, \
                          1e-4, 8e-5, 7e-5, 6.8e-5, 6.5e-5, \
                          6.2e-5, 6e-5, 5.8e-5, 5.5e-5, 5e-5, \
                          1e-5, 1e-6, 1e-7]))

for rhoF in rhoFs:
    crop_alloc, meta_sol, duration = OF.OptimizeMultipleYears(x_ini, \
                                        const, args, meta_cobyla, rhoF, rhoS) 
    print("rhoF: " + str(rhoF) + ", duration: " + str(duration))
    with open("StoOptMultipleYears/Penalties/" + \
              "rhoF" + str(rhoF) + "rhoS" + str(rhoS) + ".txt", "wb") as fp: 
        pickle.dump(["crop_alloc", "meta_sol", "rhoF", "rhoS", "settings"], fp)
        pickle.dump(crop_alloc, fp)
        pickle.dump(meta_sol, fp)   
        pickle.dump(rhoF, fp)    
        pickle.dump(rhoS, fp)
        pickle.dump(settings, fp)        
    print(meta_sol["prob_food_security"])
    
# Overview of resulting probabilities for reaching food demand
for idx, rhoF in enumerate(rhoFs):
    with open("StoOptMultipleYears/Penalties/" + \
              "rhoF" + str(rhoF) + "rhoS" + str(rhoS) + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    print(str(rhoF) + ": " + str(meta_sol["prob_food_security"]))


# %% a.2) Run default settings for different rhoS with rhoF = 0

np.warnings.filterwarnings('ignore')

settings  = OF.DefaultSettingsExcept()
x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)

rhoF = 0
rhoSs = np.array([30, 45, 50, 75, 100, \
                  150, 200, 250, 300, 400, \
                  500, 1000, 2000, 3000])
    
for rhoS in rhoSs:
    crop_alloc, meta_sol, duration = OF.OptimizeMultipleYears(x_ini, \
                                        const, args, meta_cobyla, rhoF, rhoS) 
    print("rhoS: " + str(rhoS) + ", duration: " + str(duration))
    with open("StoOptMultipleYears/Penalties/" + \
              "rhoF" + str(rhoF) + "rhoS" + str(rhoS) + ".txt", "wb") as fp: 
        pickle.dump(["crop_alloc", "meta_sol", "rhoF", "rhoS", "settings"], fp)
        pickle.dump(crop_alloc, fp)
        pickle.dump(meta_sol, fp)   
        pickle.dump(rhoF, fp)    
        pickle.dump(rhoS, fp)
        pickle.dump(settings, fp)        
    print(meta_sol["prob_staying_solvent"])


# Overview of resulting probabilities for staying solvent
for idx, rhoS in enumerate(rhoSs):
    with open("StoOptMultipleYears/Penalties/" + \
              "rhoF" + str(rhoF) + "rhoS" + str(rhoS) + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    print(str(rhoS) + ": " + str(meta_sol["prob_staying_solvent"]))    
    
    
# repeat with higher sample size
np.warnings.filterwarnings('ignore')

settings  = OF.DefaultSettingsExcept(N_c = 10000)
x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)

rhoF = 0
rhoSs = np.array([30, 45, 50, 75, 100, \
                  150, 200, 250, 300, 400, \
                  500, 1000, 2000, 3000])

#settings  = OF.DefaultSettingsExcept(N_c = 100000)
#rhoSs = [30]
    
for rhoS in rhoSs:
    crop_alloc, meta_sol, duration = OF.OptimizeMultipleYears(x_ini, \
                                        const, args, meta_cobyla, rhoF, rhoS) 
    print("rhoS: " + str(rhoS) + ", duration: " + str(duration))
    with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                          "rhoS" + str(rhoS) + "_Nc10000.txt", "wb") as fp: 
#                          "rhoS" + str(rhoS) + "_Nc100000.txt", "wb") as fp: 
        pickle.dump(["crop_alloc", "meta_sol", "rhoF", "rhoS", "settings"], fp)
        pickle.dump(crop_alloc, fp)
        pickle.dump(meta_sol, fp)   
        pickle.dump(rhoF, fp)    
        pickle.dump(rhoS, fp)
        pickle.dump(settings, fp)        
    print(meta_sol["prob_staying_solvent"])
    
    
# %% a.3) Run default settings for  both penalties != 0

np.warnings.filterwarnings('ignore')

settings  = OF.DefaultSettingsExcept()
x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)

rhoFs = [75e-4, 3e-3, \
         1e-3, 5e-4, 3e-4, \
         2e-4, 1e-4, 8e-5, 6.9e-5]
rhoSs = [30, 50, 75, \
         100, 200, \
         300, 500, 1000, 2000]

for rhoF in rhoFs:
    for rhoS in rhoSs:
        crop_alloc, meta_sol, duration = OF.OptimizeMultipleYears(x_ini, \
                                     const, args, meta_cobyla, rhoF, rhoS) 
        print("rhoS: " + str(rhoS) + ", rhoF: " + str(rhoF) + \
                      ", duration: " + str(duration))
        with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                                  "rhoS" + str(rhoS) + ".txt", "wb") as fp: 
            pickle.dump(["crop_alloc", "meta_sol", "rhoF", \
                                                 "rhoS", "settings"], fp)
            pickle.dump(crop_alloc, fp)
            pickle.dump(meta_sol, fp)   
            pickle.dump(rhoF, fp)    
            pickle.dump(rhoS, fp)
            pickle.dump(settings, fp)        
        print("FS: " + str(meta_sol["prob_food_security"]))
        print("SOL: " + str(meta_sol["prob_staying_solvent"]))
        
        
# Overview of resulting probabilities for food security and staying solvent
for rhoF in rhoFs:
    for rhoS in rhoSs:
        with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                                  "rhoS" + str(rhoS) + ".txt", "rb") as fp: 
            meta = pickle.load(fp)
            crop_alloc = pickle.load(fp)
            meta_sol = pickle.load(fp)
            print("rhoS: " + "{:.1e}".format(rhoS)  + \
              ", rhoF " + "{:.1e}".format(rhoF) + ": " + "alphaS " + \
              "{:.1e}".format(np.round(meta_sol["prob_staying_solvent"], 2)) \
              + ", alphaF " + \
              "{:.1e}".format(np.round(meta_sol["prob_food_security"], 2)))  
            
            
# repeat with higher sample size
np.warnings.filterwarnings('ignore')

settings  = OF.DefaultSettingsExcept(N_c = 10000)
x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)

rhoFs = [1e-3]
rhoSs = [0, 30, 50, 75, 100, 200, \
         300, 500, 1000, 2000]    
       
rhoSs = [0]
for rhoF in rhoFs:
    for rhoS in rhoSs:
        crop_alloc, meta_sol, duration = OF.OptimizeMultipleYears(x_ini, \
                                         const, args, meta_cobyla, rhoF, rhoS) 
        print("rhoS: " + str(rhoS) + ", rhoF: " + str(rhoF) + \
                                      ", duration: " + str(duration))
        with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                          "rhoS" + str(rhoS) + "_Nc10000.txt", "wb") as fp: 
            pickle.dump(["crop_alloc", "meta_sol", "rhoF", \
                         "rhoS", "settings"], fp)
            pickle.dump(crop_alloc, fp)
            pickle.dump(meta_sol, fp)   
            pickle.dump(rhoF, fp)    
            pickle.dump(rhoS, fp)
            pickle.dump(settings, fp)        
        print("FS: " + str(meta_sol["prob_food_security"]))
        print("SOL: " + str(meta_sol["prob_staying_solvent"]))
        
        
    
# %% b.1)  Compare crop allocations when either penalty is 0
    
settings  = OF.DefaultSettingsExcept()
x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)

fig1 = plt.figure(figsize = figsize)
fig1.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3)
ax = fig1.add_subplot(1, 2, 1)

    
rhoFs = np.flip(np.array([4e-2, 3e-2, 2e-2, 1e-2, 75e-4, \
                          5e-3, 3e-3, 2e-3, 1e-3, 75e-5, \
                          5e-4, 4e-4, 3e-4, 2e-4, 15e-5, \
                          1e-4, 8e-5, 7e-5, 6.8e-5, 6.5e-5, \
                          6.2e-5, 6e-5, 5.8e-5, 5.5e-5, 5e-5, \
                          1e-5, 1e-6, 1e-7]))
which_rhoF = np.flip(np.array([False, False, False, False, True, \
                               False, True, False, True, False, \
                               True, False, True, True, False, \
                               True, True, False, True, False, \
                               False, False, False, False, False, \
                               False, False, False]))


rhoSs = np.array([30, 45, 50, 75, 100, \
                  150, 200, 250, 300, 400, \
                  500, 1000, 2000, 3000])
which_rhoS = np.array([True, False, True, True, True, \
                        False, True, False, True, False, \
                        True, True, True, False])   
    
# - plot crop allocations over time for all penalties (rhoS = 0)
years = range(settings["yield_year"], \
              settings["yield_year"] + settings["T_max"])   

rhoS = 0
crop_allocs1 = np.empty([len(rhoFs), settings["num_crops"]])
prob_fd1 = np.empty(len(rhoFs))
prob_sol1 = np.empty(len(rhoFs))
prob_fd_yearly1 = np.empty([len(rhoFs), settings["T_max"]])
avg_fd_costs = np.empty(len(rhoFs))
fix_costs1 = np.empty(len(rhoFs))
avg_fd_short = np.empty(len(rhoFs))
tot_costs1 = np.empty(len(rhoFs))
cmap = plt.cm.get_cmap('Spectral')
for idx, rhoF in enumerate(rhoFs): 
    with open("StoOptMultipleYears/Penalties/" + \
              "rhoF" + str(rhoF) + "rhoS" + str(rhoS) + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    S = meta_sol["S"]
    S[S>0] = 1
    tot_costs1[idx] = meta_sol["exp_tot_costs"]
    avg_fd_costs[idx] = np.nansum(np.nanmean(meta_sol["fd_penalty"], axis = 0))
    fix_costs1[idx] = np.nanmean(meta_sol["fix_costs"])
    avg_fd_short[idx] = np.nanmean(np.nansum(meta_sol["S"], axis = 1))
    prob_fd_yearly1[idx, :] = np.nanmean(S, axis = 0)
    prob_fd1[idx] = meta_sol["prob_food_security"]
    prob_sol1[idx] = meta_sol["prob_staying_solvent"]
    crop_allocs1[idx, :] = np.mean(crop_alloc, axis = 0)[:,0]
    if which_rhoF[idx]:
        plt.plot(years, crop_alloc[:,0,0], \
          color = cmap(np.sum(which_rhoF[:idx])/np.sum(which_rhoF)), \
          linestyle = "--", linewidth = 2)
        plt.plot(years, crop_alloc[:,1,0], \
          color = cmap(np.sum(which_rhoF[:idx])/np.sum(which_rhoF)), \
          label = "{:.1e}".format(rhoF), linewidth = 2)
plt.legend(title = r"Penalty $\rho_F$ in [\$/kcal]:", loc = 3, \
           bbox_to_anchor=(0.015, 0.07), fontsize = 18, \
           fancybox = True, title_fontsize = "20", ncol = 2)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
plt.xlabel("Year", fontsize = 26)
plt.ylabel("Crop area in [ha]", fontsize = 26)
plt.show()
plt.title("A", fontsize = 28)
plt.ylim([-5e6, 7e7])

ax = fig1.add_subplot(1, 2, 2)

# - plot crop allocations over time for all penalties (rhoF = 0)
rhoF = 0    
crop_allocs2 = np.empty([len(rhoSs), settings["num_crops"]])
prob_sol2 = np.empty(len(rhoSs))
prob_fd2 = np.empty(len(rhoSs))
prob_fd_yearly2 = np.empty([len(rhoSs), settings["T_max"]])
avg_sol_costs = np.empty(len(rhoSs))
avg_sol_short = np.empty(len(rhoSs))
fix_costs2 = np.empty(len(rhoSs))
tot_costs2 = np.empty(len(rhoSs))
cmap = plt.cm.get_cmap('Spectral')
for idx, rhoS in enumerate(rhoSs):
    if rhoS == 30:
        with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                          "rhoS" + str(rhoS) + "_Nc100000.txt", "rb") as fp: 
            meta = pickle.load(fp)
            crop_alloc = pickle.load(fp)
            meta_sol = pickle.load(fp)        
    else:
        with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                           "rhoS" + str(rhoS) + "_Nc10000.txt", "rb") as fp: 
            meta = pickle.load(fp)
            crop_alloc = pickle.load(fp)
            meta_sol = pickle.load(fp)    
    fix_costs2[idx] = np.nanmean(meta_sol["fix_costs"])
    avg_sol_costs[idx] = np.nanmean(meta_sol["sol_penalty"])
    tot_costs2[idx] = meta_sol["exp_tot_costs"]
    tmp = meta_sol["final_fund"]
    tmp[tmp>0] == 0
    avg_sol_short[idx] = np.nanmean(tmp)
    S = meta_sol["S"]
    S[S>0] = 1
    prob_fd_yearly2[idx, :] = np.nanmean(S, axis = 0)
    prob_sol2[idx] = meta_sol["prob_staying_solvent"]
    prob_fd2[idx] = meta_sol["prob_food_security"]
    crop_allocs2[idx, :] = np.mean(crop_alloc, axis = 0)[:,0]
    if which_rhoS[idx]:
        plt.plot(years, crop_alloc[:,0,0], \
           color = cmap(np.sum(which_rhoS[:idx])/np.sum(which_rhoS)), \
           linestyle = "--", linewidth = 2)
        plt.plot(years, crop_alloc[:,1,0], \
           color = cmap(np.sum(which_rhoS[:idx])/np.sum(which_rhoS)), \
           label = "{:.1e}".format(rhoS), linewidth = 2)
plt.legend(title = r"Penalty $\rho_S$ in [\$/\$]:", loc = 3, \
           bbox_to_anchor=(0.015, 0.07), fontsize = 18, \
           fancybox = True, title_fontsize = "20", ncol = 2)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
plt.xlabel("Year", fontsize = 26)
plt.ylabel("Crop area in [ha]", fontsize = 26)
plt.show()
plt.title("B", fontsize = 28)
plt.ylim([-5e6, 7e7])

fig1.savefig("Figures/StoOpt/Penalties/CropAllocations_SinglePenalty.png", \
            bbox_inches = "tight", pad_inches = 0.5)  

# plotdifferent shares making up the total expected costs
plt.figure()
plt.plot(prob_fd1[3:], avg_fd_costs[3:], label = "avg fd costs")
plt.plot(prob_fd1[3:], fix_costs1[3:], label = "fix costs fd")
plt.plot(prob_fd1[3:], tot_costs1[3:], label = "total costs fd")
plt.plot(prob_sol2[1:], avg_sol_costs[1:], label = "avg sol costs")
plt.plot(prob_sol2[1:], fix_costs2[1:], label = "fix costs sol")
plt.plot(prob_sol2[1:], tot_costs2[1:], label = "total costs sol")
plt.legend()
plt.show()

# %% b.2)  Compare overall probabilities for either penalty = 0

fig = plt.figure(figsize = figsize)  
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3)


rhoFs = np.flip(np.array([4e-2, 3e-2, 2e-2, 1e-2, 75e-4, \
                          5e-3, 3e-3, 2e-3, 1e-3, 75e-5, \
                          5e-4, 4e-4, 3e-4, 2e-4, 15e-5, \
                          1e-4, 8e-5, 7e-5, 6.8e-5, 6.5e-5, \
                          6.2e-5, 6e-5, 5.8e-5, 5.5e-5, 5e-5, \
                          1e-5, 1e-6, 1e-7]))
which_rhoF = np.flip(np.array([True, True, True, True, True, \
                               True, True, True, True, True, \
                               True, True, True, True, True, \
                               True, True, True, True, True, \
                               True, True, True, True, True, \
                               True, True, True]))


rhoSs = np.array([30, 45, 50, 75, 100, \
                  150, 200, 250, 300, 400, \
                  500, 1000, 2000, 3000])
which_rhoS = np.array([False, True, True, True, True, \
                       True, True, True, True, True, \
                       True, True, True, True])    
    
# - for rhoS = 0
ax = fig.add_subplot(1, 2, 2)
plt.scatter(rhoFs[which_rhoF], prob_fd1[which_rhoF], color = cols[0], \
            label = r"$\alpha_F$", s = 50)
plt.scatter(rhoFs[which_rhoF], prob_sol1[which_rhoF], color = cols[1], \
            label = r"$\alpha_S$", s = 50)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
plt.xlabel(r"Penalty $\rho_F$ in [\$/kcal]", fontsize = 26)
plt.ylim([-0.02, 1.05])
plt.xlim([-0.001, 0.0425])
plt.ylabel("Probabilities", fontsize = 26)
plt.legend(title = "Probability:", fontsize = 18, loc = 4, \
               fancybox = True, title_fontsize = "20", ncol = 2)
plt.title("B", fontsize = 28)

# - for rhoS = 0
ax = fig.add_subplot(1, 2, 1)
plt.scatter(rhoSs[which_rhoS], prob_fd2[which_rhoS], color = cols[0], \
            label = r"$\alpha_F$", s = 50)
plt.scatter(rhoSs[which_rhoS], prob_sol2[which_rhoS], color = cols[1], \
            label = r"$\alpha_S$", s = 50)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
plt.xlabel(r"Penalty $\rho_S$ in [\$/\$]", fontsize = 26)
plt.ylabel("Probabilities", fontsize = 26)
plt.ylim([-0.02, 1.05])
plt.xlim([-50, 3100])
plt.legend(title = "Probability:", fontsize = 18, loc = 4, \
               fancybox = True, title_fontsize = "20", ncol = 2)
plt.title("A", fontsize = 28)

fig.savefig("Figures/StoOpt/Penalties/OverallProbabilities_SinglePenalty.png",\
            bbox_inches = "tight", pad_inches = 0.5)  

# %% b.3)  Yearly food securtiy probabilities for either penalty = 0

rhoFs = np.flip(np.array([4e-2, 3e-2, 2e-2, 1e-2, 75e-4, \
                          5e-3, 3e-3, 2e-3, 1e-3, 75e-5, \
                          5e-4, 4e-4, 3e-4, 2e-4, 15e-5, \
                          1e-4, 8e-5, 7e-5, 6.8e-5, 6.5e-5, \
                          6.2e-5, 6e-5, 5.8e-5, 5.5e-5, 5e-5, \
                          1e-5, 1e-6, 1e-7]))
which_rhoF = np.flip(np.array([False, False, False, False, True, \
                               False, True, False, True, False, \
                               True, False, True, True, False, \
                               True, True, False, True, False, \
                               False, False, False, False, False, \
                               False, False, False]))

rhoSs = np.array([30, 45, 50, 75, 100, \
                  150, 200, 250, 300, 400, \
                  500, 1000, 2000, 3000])
which_rhoS = np.array([True, False, True, True, True, \
                        False, True, False, True, False, \
                        True, True, True, False])   
    

fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3)

# - plot resulting probabilities of food security per year for rhoF != 0 and
#   rhoS = 0
ax = fig.add_subplot(1, 2, 1)
for i in range(0, len(rhoFs)):
    if which_rhoF[i]:
        plt.plot(years, 1 - prob_fd_yearly1[i,:], \
                 color = cmap(np.sum(which_rhoF[:i])/np.sum(which_rhoF)),
                 label = "{:.1e}".format(rhoFs[i]), linewidth = 2)
plt.legend(title = r"Penalty $\rho_F$ in [\$/kcal]:", loc = 4, \
           bbox_to_anchor=(0.99, 0.02), fontsize = 18, \
           fancybox = True, title_fontsize = "20", ncol = 2)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
plt.ylim([-0.02,1.01])
plt.xlabel("Years", fontsize = 26)
plt.ylabel(r"Yearly probability $\alpha_F$", fontsize = 26)
plt.title("A", fontsize = 28)

# - plot probability of food security over the different years when rhoF = 0
#   but rhoS != 0
ax = fig.add_subplot(1, 2, 2)
for i in range(0, len(rhoSs)):
    if which_rhoS[i]:
        plt.plot(years, 1 - prob_fd_yearly2[i,:], \
                 color = cmap(np.sum(which_rhoS[:i])/np.sum(which_rhoS)),
                 label = "{:.1e}".format(rhoSs[i]), linewidth = 2)
plt.legend(title = r"Penalty $\rho_S$ in [\$/\$]:", loc = 4, \
           bbox_to_anchor=(0.99, 0.02), fontsize = 18, \
           fancybox = True, title_fontsize = "20", ncol = 2)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
plt.ylim([-0.02,1.02])
plt.xlabel("Years", fontsize = 26)
plt.ylabel(r"Yearly probability $\alpha_F$", fontsize = 26)
plt.title("B", fontsize = 28)

fig.savefig("Figures/StoOpt/Penalties/YearlyFoodSecurity_SinglePenalty.png", \
            bbox_inches = "tight", pad_inches = 0.5)  


# %% b.5) Visualize of interaction of penalties != 0

rhoFs = np.flip([75e-4, 3e-3, \
         1e-3, 5e-4, 3e-4, \
         2e-4, 1e-4, 8e-5, 6.9e-5, 0])
rhoSs = [0, 30, 50, 75, \
         100, 200, \
         300, 500, 1000, 2000]

probsF = np.zeros([len(rhoFs), len(rhoSs)])
probsS = np.zeros([len(rhoFs), len(rhoSs)])

cmap = plt.cm.get_cmap('Spectral')

# - prepare data
for idxF, rhoF in enumerate(rhoFs):
    if rhoF == 0:
        rhoF = int(0)
    for idxS, rhoS in enumerate(rhoSs):
        if (rhoS == 0) and (rhoF == 0):
            probsF[idxS, idxF] = 0
            probsS[idxS, idxF] = 0
        else:
            with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                                      "rhoS" + str(rhoS) + ".txt", "rb") as fp: 
                meta = pickle.load(fp)
                crop_alloc = pickle.load(fp)
                meta_sol = pickle.load(fp)
            probsF[idxS, idxF] = meta_sol["prob_food_security"]
            probsS[idxS, idxF] = meta_sol["prob_staying_solvent"]

# plot probF for changing rhoS and different fixed rhoF, and probS for changing
# rhoF and different fixed rhoS
fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3)
ax = fig.add_subplot(1, 2, 1)
for idx, rhoF in enumerate(rhoFs):
    plt.plot(rhoSs, probsF[:, idx], \
             color = cmap(idx/len(rhoFs)), \
             label = "{:.1e}".format(rhoF), linewidth = 2)
    plt.legend(title = r"Penalty $\rho_F$ in [\$/kcal]:", fontsize = 18, \
               fancybox = True, ncol = 2, title_fontsize = "20")
    plt.ylabel(r"Probability $\alpha_F$", fontsize = 26)
    plt.xlabel(r"Penalty $\rho_S$ in [\$/\$]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.ylim([-0.01, 1.01])
    plt.title("A", fontsize = 28)
ax = fig.add_subplot(1, 2, 2)
for idx, rhoS in enumerate(rhoSs):
    plt.plot(rhoFs, probsS[idx, :], \
             color = cmap(idx/len(rhoSs)), \
             label = "{:.0e}".format(rhoS), linewidth = 2)
    plt.legend(title = r"Penalty $\rho_S$ in [\$/\$]:", fontsize = 18, \
               fancybox = True, title_fontsize = "20", ncol = 2)
    plt.ylabel(r"Probability $\alpha_S$", fontsize = 26)
    plt.xlabel(r"Penalty $\rho_F$ in [\$/kcal]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.ylim([-0.01, 1.01])
    plt.title("B", fontsize = 28)
    
fig.savefig("Figures/StoOpt/Penalties/InteractionPenalties_Other.png", \
            bbox_inches = "tight", pad_inches = 0.5)  
    
# plot probF for changing rhoF and different fixed rhoS, and probS for changing
# rhoS and different fixed rhoF
fig = plt.figure(figsize = figsize)  
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3) 
ax = fig.add_subplot(1, 2, 2)
for idx, rhoS in enumerate(rhoSs):
    plt.plot(rhoFs, probsF[idx, :], \
             color = cmap(idx/len(rhoSs)), \
             label = "{:.0e}".format(rhoS), linewidth = 2)
    plt.legend(title = r"Penalty $\rho_S$ in [\$/\$]:", fontsize = 18, \
               fancybox = True, ncol = 2, title_fontsize = "20")
    plt.ylabel(r"Probability $\alpha_F$", fontsize = 26)
    plt.xlabel(r"Penlty $\rho_F$ in [\$/kcal]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.ylim([-0.01, 1.01])
    plt.title("B", fontsize = 28)
ax = fig.add_subplot(1, 2, 1)    
for idx, rhoF in enumerate(rhoFs):
    plt.plot(rhoSs, probsS[:, idx],\
             color = cmap(idx/len(rhoFs)), \
             label = "{:.1e}".format(rhoF), linewidth = 2)
    plt.legend(title = r"Penalty $\rho_F$ in [\$/kcal]:", fontsize = 18, \
               fancybox = True, title_fontsize = "20", ncol = 2)
    plt.ylabel(r"Probability $\alpha_S$", fontsize = 26)
    plt.xlabel(r"Penalty $\rho_S$ in [\$/\$]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.ylim([-0.01, 1.01])
    plt.title("A", fontsize = 28)
    
fig.savefig("Figures/StoOpt/Penalties/InteractionPenalties_Same.png", \
            bbox_inches = "tight", pad_inches = 0.5) 
    
# %% b.6) Visualize of crop allocation over time for both penalties != 0
  
# fixed rhoF for different rhoS
rhoF = 1e-3 
rhoSs = [0, 30, 50, 75, \
         100, 200, \
         300, 500, 1000, 2000]

fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3) 
ax = fig.add_subplot(1, 2, 2)
for idx, rhoS in enumerate(rhoSs):
    with open("StoOptMultipleYears/Penalties/rhoF" + str(rhoF) + \
                          "rhoS" + str(rhoS) + "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    plt.plot(years, crop_alloc[:, 0, 0], \
             color = cmap(idx/len(rhoSs)), \
             linestyle = "--", linewidth = 2)
    plt.plot(years, crop_alloc[:, 1, 0], \
             color = cmap(idx/len(rhoSs)), \
             label = "{:.1e}".format(rhoS), linewidth = 2)
    plt.xlabel("Year", fontsize = 26)
    plt.ylabel("Crop area in [ha]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.legend(title = r"Penalty $\rho_S$ in [\$/\$]:", fontsize = 18, \
               fancybox = True, loc = 3, bbox_to_anchor=(0.015, 0.07), \
               title_fontsize = "20", ncol = 2)
    plt.ylim([-5e6, 7e7])
    ax.yaxis.offsetText.set_fontsize(20)
    plt.title("B", fontsize = 28)
    
    
    
# fixed rhoS for different rhoF
rhoFs = np.flip([75e-4, 3e-3, \
         1e-3, 5e-4, 3e-4, \
         2e-4, 1e-4, 8e-5, 6.9e-5, 0])
rhoS = 200

ax = fig.add_subplot(1, 2, 1)
for idx, rhoF in enumerate(rhoFs):
    if rhoF == 0:
        rhoF = int(0)
    with open("StoOptMultipleYears/Penalties/" + \
              "rhoF" + str(rhoF) + "rhoS" + str(rhoS) + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    plt.plot(years, crop_alloc[:, 0, 0], \
             color = cmap(idx/len(rhoFs)), \
             linestyle = "--", linewidth = 2)
    plt.plot(years, crop_alloc[:, 1, 0], \
             color = cmap(idx/len(rhoFs)), \
             label = "{:.1e}".format(rhoF), linewidth = 2)
    plt.xlabel("Year", fontsize = 26)
    plt.ylabel("Crop area in [ha]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.legend(title = r"Penalty $\rho_F$ in [\$/kcal]:", fontsize = 18, \
               fancybox = True, loc = 3, bbox_to_anchor=(0.015, 0.07), \
               title_fontsize = "20", ncol = 2)
    plt.ylim([-5e6, 7e7])
    ax.yaxis.offsetText.set_fontsize(20)
    plt.title("A", fontsize = 28)
   
fig.savefig("Figures/StoOpt/Penalties/CropAllocations_BothPenalties.png", \
            bbox_inches = "tight", pad_inches = 0.5) 

# %% Overview of costs and share that penalties have in total expected costs
#    for one example of penalties

with open("StoOptMultipleYears/Penalties/" + \
              "rhoF0.0002rhoS500.txt", "rb") as fp: 
    meta = pickle.load(fp)
    crop_alloc = pickle.load(fp)
    meta_sol = pickle.load(fp)
                
avg_fd_costs_both = np.nansum(np.nanmean(meta_sol["fd_penalty"], axis = 0))
avg_sol_costs_both = np.nanmean(meta_sol["sol_penalty"])
fix_costs_both = np.nanmean(meta_sol["fix_costs"])
tot_costs_both = meta_sol["exp_tot_costs"]

print("{:.1e}".format(avg_fd_costs_both))
print("{:.1e}".format(avg_sol_costs_both))
print("{:.1e}".format(fix_costs_both))
print("{:.1e}".format(tot_costs_both))

print(avg_fd_costs_both/tot_costs_both)
print(avg_sol_costs_both/tot_costs_both)

###############################################################################

# %% ######### 3. ANALYZING DIFFERENT YIELD AND DEMAND SCENARIOS ##############

# %% a) Running model for different scenarios

np.warnings.filterwarnings('ignore')

yield_scenarios = ["fixed", "trend"]
demand_scenarios = ["fixed", "Low", "Medium", "High"]

probFs = [0.7, 0.85, 0.95]
probSs = [0.7, 0.85, 0.95]
for y_s in yield_scenarios:
    for d_s in demand_scenarios:
        for probF in probFs:
            for probS in probSs:
                np.warnings.filterwarnings('ignore')
                crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                    OF.OptimizeFoodSecurityProblem(probF, probS, \
                                rhoFini = 1e-3, rhoSini = 100, \
                                yield_projection = y_s, \
                                pop_scenario = d_s) 
                with open("StoOptMultipleYears/Scenarios/" + \
                          y_s + "Ylds" + d_s + "Pop" + \
                          "_probF" + str(probF) + "probS" + str(probS) + \
                          ".txt", "wb") as fp: 
                    pickle.dump(["settings", "penalties", "probs", \
                                 "crop_alloc", "meta_sol"], fp)
                    pickle.dump(settings, fp)
                    pickle.dump([rhoF, rhoS], fp)   
                    pickle.dump([probF, probS], fp) 
                    pickle.dump(crop_alloc, fp)
                    pickle.dump(meta_sol, fp)         

# repeat with higher sample size
np.warnings.filterwarnings('ignore')

yield_scenarios = ["fixed"]
demand_scenarios = ["fixed"]

probFs = [0.95]
probSs = [0.85]

for y_s in yield_scenarios:
    for d_s in demand_scenarios:
        for probF in probFs:
            for probS in probSs:
                np.warnings.filterwarnings('ignore')
                crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                    OF.OptimizeFoodSecurityProblem(probF, probS, \
                                rhoFini = 1e-3, rhoSini = 100, \
                                yield_projection = y_s, \
                                pop_scenario = d_s, N_c = 10000) 
                with open("StoOptMultipleYears/Scenarios/" + \
                          y_s + "Ylds" + d_s + "Pop" + \
                          "_probF" + str(probF) + "probS" + str(probS) + \
                          "_Nc10000.txt", "wb") as fp: 
                    pickle.dump(["settings", "penalties", "probs", \
                                 "crop_alloc", "meta_sol"], fp)
                    pickle.dump(settings, fp)
                    pickle.dump([rhoF, rhoS], fp)   
                    pickle.dump([probF, probS], fp) 
                    pickle.dump(crop_alloc, fp)
                    pickle.dump(meta_sol, fp)    
     
# %% b) Visualize
                    
col = ["darkred", "royalblue", "gold", "darkgrey"]   
  
probF = 0.95
probS = 0.85

yield_scenarios = ["fixed", "trend"]
demand_scenarios = ["fixed", "Low", "Medium", "High"]

fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3) 

titles = ["A", "B"]
for idx, y_s in enumerate(yield_scenarios):
    ax = fig.add_subplot(1, 2, idx + 1)
    for idx1, d_s in enumerate(demand_scenarios):
        if (d_s == "fixed") and (y_s == "fixed"):
            with open("StoOptMultipleYears/Scenarios/" + \
                      y_s + "Ylds" + d_s + "Pop" + \
                      "_probF" + str(probF) + "probS" + str(probS) + \
                      "_Nc10000.txt", "rb") as fp: 
                meta = pickle.load(fp)
                settings = pickle.load(fp)
                [rhoF, rhoS] = pickle.load(fp)
                [probF, probS] = pickle.load(fp)
                crop_alloc = pickle.load(fp)
                meta_sol = pickle.load(fp)
        else:
            with open("StoOptMultipleYears/Scenarios/" + \
                      y_s + "Ylds" + d_s + "Pop" + \
                      "_probF" + str(probF) + "probS" + str(probS) + \
                      ".txt", "rb") as fp: 
                meta = pickle.load(fp)
                settings = pickle.load(fp)
                [rhoF, rhoS] = pickle.load(fp)
                [probF, probS] = pickle.load(fp)
                crop_alloc = pickle.load(fp)
                meta_sol = pickle.load(fp)
        if d_s == "fixed":
            d_s = "Fixed"
        plt.plot(years, crop_alloc[:,1,0], label = d_s, \
                     color = col[idx1], \
                     linewidth = 2)
        plt.plot(years, crop_alloc[:,0,0], \
                     color = col[idx1], \
                     linestyle = "--", linewidth = 2)
    plt.xlabel("Year", fontsize = 26)
    plt.ylabel("Crop area in [ha]", fontsize = 26)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.legend(title = "Population:", fontsize = 18, \
               fancybox = True, loc = 3, bbox_to_anchor=(0.015, 0.07), \
               title_fontsize = "20")
    plt.ylim([-5e6, 7e7])
    plt.title(titles[idx], fontsize = 28)
    ax.yaxis.offsetText.set_fontsize(20)
    print(meta_sol["prob_staying_solvent"])
    print(meta_sol["prob_food_security"])
    print(rhoF)
    print(rhoS)
    
         
fig.savefig("Figures/StoOpt/Scenarios/CropAllocations.png", \
            bbox_inches = "tight", pad_inches = 0.5)     
               
###############################################################################

# %% #################### 3. ANALYZING OTHER PARAMETERS #######################

# other parameters are: risk, tax, perc_guaranteed

# %% a.1) Run model for different risks

risks = [10, 40]
probFs = [0.95]
probSs = [0.85]

np.warnings.filterwarnings('ignore')

for risk in risks:
    for probF in probFs:
        for probS in probSs:
            print("Risk: " + str(risk) + ", probF: " + \
                  str(probF)+ ", probS: " + str(probS))
            crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                    OF.OptimizeFoodSecurityProblem(probF, probS, \
                                       rhoFini = 1e-3, rhoSini = 100, \
                                       risk = risk, N_c = 10000) 
            with open("StoOptMultipleYears/OtherParameters/" + \
                      "risk" + str(risk) + \
                      "_probF" + str(probF) + "probS" + str(probS) + \
                      "_Nc10000.txt", "wb") as fp: 
                pickle.dump(["settings", "penalties", "probs", \
                             "crop_alloc", "meta_sol"], fp)
                pickle.dump(settings, fp)
                pickle.dump([rhoF, rhoS], fp)   
                pickle.dump([probF, probS], fp) 
                pickle.dump(crop_alloc, fp)
                pickle.dump(meta_sol, fp)  

# %% a.2) Run model for different taxes
                
taxes = [0.01, 0.05]
probFs = [0.95]
probSs = [0.85]

np.warnings.filterwarnings('ignore')
    
for tax in taxes:
    for probF in probFs:
        for probS in probSs:
            print("Tax: " + str(tax) + ", probF: " + \
                  str(probF)+ ", probS: " + str(probS))
            crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                    OF.OptimizeFoodSecurityProblem(probF, probS, \
                                       rhoFini = 1e-3, rhoSini = 100, \
                                       tax = tax, N_c = 10000)
            with open("StoOptMultipleYears/OtherParameters/" + \
                      "tax" + str(tax) + \
                      "_probF" + str(probF) + "probS" + str(probS) + \
                      "_Nc10000.txt", "wb") as fp: 
                pickle.dump(["settings", "penalties", "probs", \
                             "crop_alloc", "meta_sol"], fp)
                pickle.dump(settings, fp)
                pickle.dump([rhoF, rhoS], fp)   
                pickle.dump([probF, probS], fp) 
                pickle.dump(crop_alloc, fp)
                pickle.dump(meta_sol, fp)     
                
# %% a.3) Run model for different percentages for guaranteed income
                
perc_guaranteeds = [0.6, 0.9]
probFs = [0.95]
probSs = [0.85]

np.warnings.filterwarnings('ignore')

for perc_guaranteed in perc_guaranteeds:
    for probF in probFs:
        for probS in probSs:
            print("Percentage: " + str(perc_guaranteed) + ", probF: " + \
                  str(probF)+ ", probS: " + str(probS))
            crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                    OF.OptimizeFoodSecurityProblem(probF, probS, \
                                       rhoFini = 1e-3, rhoSini = 100, \
                                       perc_guaranteed = perc_guaranteed, \
                                       N_c = 10000) 
            with open("StoOptMultipleYears/OtherParameters/" + \
                      "PercGuaranteed" + str(perc_guaranteed) + \
                      "_probF" + str(probF) + "probS" + str(probS) + \
                      "_Nc10000.txt", "wb") as fp: 
                pickle.dump(["settings", "penalties", "probs", \
                             "crop_alloc", "meta_sol"], fp)
                pickle.dump(settings, fp)
                pickle.dump([rhoF, rhoS], fp)   
                pickle.dump([probF, probS], fp) 
                pickle.dump(crop_alloc, fp)
                pickle.dump(meta_sol, fp)  
                
# %% a.4) Get default runs from the scenario runs
# (these were already run, but we copy them to the OtherParameters folder, 
# to be able to use loops over the settings when loading data)
                
probFs = [0.95]
probSs = [0.85]

for probF in probFs:
    for probS in probSs:
        with open("StoOptMultipleYears/Scenarios/fixedYldsfixedPop" + \
                  "_probF" + str(probF) + "probS" + str(probS) + \
                  "_Nc10000.txt", "rb") as fp: 
            meta = pickle.load(fp)
            settings = pickle.load(fp)
            [rhoF, rhoS] = pickle.load(fp)
            [probF, probS] = pickle.load(fp)
            crop_alloc = pickle.load(fp)
            meta_sol = pickle.load(fp)
        with open("StoOptMultipleYears/OtherParameters/Default" + \
                  "_probF" + str(probF) + "probS" + str(probS) + \
                  "_Nc10000.txt", "wb") as fp: 
            pickle.dump(["settings", "penalties", "probs", \
                         "crop_alloc", "meta_sol"], fp)
            pickle.dump(settings, fp)
            pickle.dump([rhoF, rhoS], fp)   
            pickle.dump([probF, probS], fp) 
            pickle.dump(crop_alloc, fp)
            pickle.dump(meta_sol, fp)  
    


# %% b.1) Visualize crop allocations
    
col = ["darkred", "royalblue", "darkgrey"]

probF = 0.95
probS = 0.85

risks = [10, 20, 40]
perc_guaranteeds = [0.6, 0.75, 0.9]
taxes = [0.01, 0.03, 0.05]

fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.3, top=0.7, left=0.1, right=0.9,
                wspace=0.3, hspace=0.3) 

# - for different risk levels
ax = fig.add_subplot(1, 3, 3)
for idx, risk in enumerate(risks):
    if risk == 20:
        file = "Default"
    else:
        file = "risk" + str(risk)
    with open("StoOptMultipleYears/OtherParameters/" + file + \
              "_probF" + str(probF) + "probS" + str(probS) + \
              "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    plt.plot(years, crop_alloc[:,1,0], label = str(100* (1/risk)) + " %", \
                 color = col[idx], linewidth = 2)
    plt.plot(years, crop_alloc[:,0,0], \
                 color = col[idx], \
                 linestyle = "--", linewidth = 2)
plt.xlabel("Year", fontsize = 28)
plt.ylabel("Crop area in [ha]", fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=22)
lg = plt.legend(title = r"Risk level $r$:", fontsize = 20, \
           fancybox = True, loc = 3, bbox_to_anchor=(0.015, 0.07), \
           title_fontsize = "22")
renderer = fig.canvas.get_renderer()
shift = max([t.get_window_extent(renderer).width for t in lg.get_texts()])
for t in lg.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    t.set_position((shift,0))
plt.ylim([-5e6, 7e7])
plt.title("C", fontsize = 30)
ax.yaxis.offsetText.set_fontsize(22)
    
# - for different shares S_gov for guaranteed income
ax = fig.add_subplot(1, 3, 2)
for idx, perc_guaranteed in enumerate(np.flip(perc_guaranteeds)):
    if perc_guaranteed == 0.75:
        file = "Default"
    else:
        file = "PercGuaranteed" + str(perc_guaranteed)
    with open("StoOptMultipleYears/OtherParameters/" + file + \
              "_probF" + str(probF) + "probS" + str(probS) + \
              "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    plt.plot(years, crop_alloc[:,1,0],  color = col[idx], \
             label = str(100*perc_guaranteed) + " %", linewidth = 2)
    plt.plot(years, crop_alloc[:,0,0], \
                 color = col[idx], \
                 linestyle = "--", linewidth = 2)
plt.xlabel("Year", fontsize = 28)
plt.ylabel("Crop area in [ha]", fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=22)
plt.legend(title = r"Share $S_\mathrm{gov}$:", fontsize = 20, \
           fancybox = True, loc = 3, bbox_to_anchor=(0.015, 0.07), \
           title_fontsize = "22")
plt.ylim([-5e6, 7e7])
plt.title("B", fontsize = 30)
ax.yaxis.offsetText.set_fontsize(22)
    
# - for different tax rates
ax = fig.add_subplot(1, 3, 1)
for idx, tax in enumerate(taxes):
    if tax == 0.03:
        file = "Default"
    else:
        file = "tax" + str(tax)
    with open("StoOptMultipleYears/OtherParameters/" + file + \
              "_probF" + str(probF) + "probS" + str(probS) + \
              "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    plt.plot(years, crop_alloc[:,1,0], label = str(100* tax) + " %", \
                 color = col[idx], linewidth = 2)
    plt.plot(years, crop_alloc[:,0,0], \
                 color = col[idx], \
                 linestyle = "--", linewidth = 2)
plt.xlabel("Year", fontsize = 28)
plt.ylabel("Crop area in [ha]", fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=22)
plt.legend(title = r"Tax rate $\tau$:", fontsize = 20, \
           fancybox = True, loc = 3, bbox_to_anchor=(0.015, 0.07), \
           title_fontsize = "22")
plt.ylim([-5e6, 7e7])
plt.title("A", fontsize = 30)
ax.yaxis.offsetText.set_fontsize(22)
    

fig.savefig("Figures/StoOpt/OtherParameters/CropAllocations.png", \
            bbox_inches = "tight", pad_inches = 0.5)     


# %% b.2) Visualize final fund distributions

col = ["darkred", "royalblue", "darkgrey"]
   
probF = 0.95
probS = 0.85

risks = [10, 20, 40]
perc_guaranteeds = [0.6, 0.75, 0.9]
taxes = [0.01, 0.03, 0.05]

fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.3, top=0.7, left=0.1, right=0.9,
                wspace=0.3, hspace=0.3) 

# - for different risk levels
ax = fig.add_subplot(1, 3, 3)
for idx, risk in enumerate(risks):
    if idx == 2:
        a = 0.4
    elif idx == 1:
        a = 0.5
    else:
        a = 0.6
    if risk == 20:
        file = "Default"
    else:
        file = "risk" + str(risk)
    with open("StoOptMultipleYears/OtherParameters/" + file + \
              "_probF" + str(probF) + "probS" + str(probS) + \
              "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    print("Average payouts (risk level " + str(risk) + "): " + \
          "{:.1e}".format(np.nanmean(meta_sol["payouts"]\
           [meta_sol["payouts"]>0])))
    print("Prob staying solven (risk level " + str(risk) + "): " + \
          "{:.3e}".format(meta_sol["prob_staying_solvent"]))
    plt.hist(meta_sol["final_fund"], \
             label = str(100* (1/risk)) + " %", \
             color = col[idx], alpha = a, density = True, \
             bins = np.arange(-0.25e10, 1e10, 150000000))
plt.xlabel("Final fund in [$]", fontsize = 28)
plt.ylabel("Density", fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
lg = plt.legend(title = r"Risk level $r$:", fontsize = 20, \
           fancybox = True, loc = 1, title_fontsize = "22")
renderer = fig.canvas.get_renderer()
shift = max([t.get_window_extent(renderer).width for t in lg.get_texts()])
for t in lg.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    t.set_position((shift,0))
plt.title("C", fontsize = 30)
plt.ylim([-2e-11, 1.7e-9])
plt.xlim([-2.5e9, 11e9])
ax.yaxis.offsetText.set_fontsize(22)  
ax.xaxis.offsetText.set_fontsize(22)  

# - for different shares S_gov for guaranteed income   
ax = fig.add_subplot(1, 3, 2)
for idx, perc_guaranteed in enumerate(np.flip(perc_guaranteeds)):
    if idx == 2:
        a = 0.4
    elif idx == 1:
        a = 0.5
    else:
        a = 0.6
    if perc_guaranteed == 0.75:
        file = "Default"
    else:
        file = "PercGuaranteed" + str(perc_guaranteed)
    with open("StoOptMultipleYears/OtherParameters/" + file + \
              "_probF" + str(probF) + "probS" + str(probS) + \
              "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    print("Average payouts (perc guaranteed " + str(perc_guaranteed) + \
          "): " + "{:.1e}".format(np.nanmean(meta_sol["payouts"]\
           [meta_sol["payouts"]>0])))
    print("Prob staying solven (perc guaranteed " + str(perc_guaranteed) + \
          "): " + "{:.3e}".format(meta_sol["prob_staying_solvent"]))
    plt.hist(meta_sol["final_fund"], \
             label = str(100*perc_guaranteed) + " %", \
             color = col[idx], alpha = a, density = True, \
             bins = np.arange(-0.25e10, 1e10, 150000000))
plt.xlabel("Final fund in [$]", fontsize = 28)
plt.ylabel("Density", fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.legend(title = r"Share $S_\mathrm{gov}$:", fontsize = 20, \
           fancybox = True, loc = 1, title_fontsize = "22")
plt.title("B", fontsize = 30)
plt.ylim([-2e-11, 1.7e-9])
plt.xlim([-2.5e9, 11e9])
ax.yaxis.offsetText.set_fontsize(22)
ax.xaxis.offsetText.set_fontsize(22)  
    
# - for different tax rates
ax = fig.add_subplot(1, 3, 1)
for idx, tax in enumerate(taxes):
    if idx == 2:
        a = 0.4
    elif idx == 1:
        a = 0.5
    else:
        a = 0.6
    if tax == 0.03:
        file = "Default"
    else:
        file = "tax" + str(tax)
    with open("StoOptMultipleYears/OtherParameters/" + file + \
              "_probF" + str(probF) + "probS" + str(probS) + \
              "_Nc10000.txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
    print("Average payouts (tax rate " + str(tax) + "): " + \
          "{:.1e}".format(np.nanmean(meta_sol["payouts"]\
           [meta_sol["payouts"]>0])))
    print("Prob staying solven (tax rate " + str(tax) + "): " + \
          "{:.3e}".format(meta_sol["prob_staying_solvent"]))
    plt.hist(meta_sol["final_fund"], \
             label = str(100*tax) + " %", \
             color = col[idx], alpha = a, density = True, \
             bins = np.arange(-0.25e10, 1e10, 150000000))
plt.xlabel("Final fund in [$]", fontsize = 28)
plt.ylabel("Density", fontsize = 28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.legend(title = r"Tax rate $\tau$:", fontsize = 20, \
           fancybox = True, loc = 1, title_fontsize = "22")
plt.title("A", fontsize = 30)
plt.ylim([-2e-11, 1.7e-9])
plt.xlim([-2.5e9, 11e9])
ax.yaxis.offsetText.set_fontsize(22)
ax.xaxis.offsetText.set_fontsize(22)  


fig.savefig("Figures/StoOpt/OtherParameters/FinalFund.png", \
            bbox_inches = "tight", pad_inches = 0.5)     


###############################################################################

# %% ###################### 4. ANALYZING OTHER k AND  #########################


# %% a) Run model for different cases

cluster_combinations = [(2,1), (7,1)]

probFs = [0.95]
probSs = [0.85]

N_c = 10000

np.warnings.filterwarnings('ignore')

for [k, num_cl_cat] in cluster_combinations:
    for probF in probFs:
        for probS in probSs:
            print("[k, k']: [" + str(k) + ", " + str(num_cl_cat) + \
                  "], probF: " +  str(probF)+ ", probS: " + str(probS))
            crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                    OF.OptimizeFoodSecurityProblem(probF, probS, \
                                       rhoFini = 0.00086, rhoSini = 100, \
                                       k = k, num_cl_cat = num_cl_cat, \
                                       N_c = N_c) 
            with open("StoOptMultipleYears/Clusters/" + \
                      "k" + str(k) + "kPrime" + str(num_cl_cat) + \
                      "_probF" + str(probF) + "probS" + str(probS) + \
                      ".txt", "wb") as fp: 
                pickle.dump(["settings", "penalties", "probs", \
                             "crop_alloc", "meta_sol"], fp)
                pickle.dump(settings, fp)
                pickle.dump([rhoF, rhoS], fp)   
                pickle.dump([probF, probS], fp) 
                pickle.dump(crop_alloc, fp)
                pickle.dump(meta_sol, fp)    
 
# %% rerunning with higher sample size
                
runs = ["k2kPrime1_probF0.95probS0.85", \
        "k7kPrime1_probF0.95probS0.85"]
    
for run in runs:
    with open("StoOptMultipleYears/Clusters/" + run + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
                      
    settings["N_c"] = 50000
    print("Get Parameters", flush = True)
    x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)
    
    print("Start optimization", flush = True)
    crop_alloc, meta_sol, duration = \
                    OF.OptimizeMultipleYears(crop_alloc.flatten(), \
                                const, args, meta_cobyla, rhoF, rhoS) 
    print(duration, flush = True)
    with open("StoOptMultipleYears/Clusters/" +  \
               run + "_Nc50000MaxFun70000.txt", "wb") as fp: 
        pickle.dump(["settings", "penalties", "probs", \
                     "crop_alloc", "meta_sol"], fp)
        pickle.dump(settings, fp)
        pickle.dump([rhoF, rhoS], fp)   
        pickle.dump([probF, probS], fp) 
        pickle.dump(crop_alloc, fp)
        pickle.dump(meta_sol, fp)    

 # %% rerun k = 1 with this penaltiies to compare resulting probabilities

runs = ["k2kPrime1_probF0.95probS0.85", \
        "k7kPrime1_probF0.95probS0.85"]

for run in runs:
    with open("StoOptMultipleYears/Clusters/" + run + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
                      
    settings_new = OF.DefaultSettingsExcept(k = 1,                        
                          num_cl_cat = 1,
                          num_crops = settings["num_crops"],               
                          yield_projection = settings["yield_projection"],   
                          yield_year = settings["yield_year"],          
                          stilised = settings["stilised"],            
                          pop_scenario = settings["pop_scenario"],
                          risk = settings["risk"],                          
                          N_c = settings["N_c"], 
                          T_max = settings["T_max"],                  
                          seed = settings["seed"],
                          tax = settings["tax"],
                          perc_guaranteed = settings["perc_guaranteed"])
    
    probFres = meta_sol["prob_food_security"]
    probSres = meta_sol["prob_staying_solvent"]
    print("Get Parameters", flush = True)
    x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings_new)
    
    print("Start optimization", flush = True)
    crop_alloc, meta_sol, duration = OF.OptimizeMultipleYears(x_ini, \
                                        const, args, meta_cobyla, rhoF, rhoS) 
    print(duration, flush = True)
    with open("StoOptMultipleYears/Clusters/k1_WithPenaltiesOf" +  \
               run + ".txt", "wb") as fp: 
        pickle.dump(["settings", "original_penalty_setting", \
                     "original_penalty_resulting", "probs", \
                     "crop_alloc", "meta_sol"], fp)
        pickle.dump(settings_new, fp)
        pickle.dump([rhoF, rhoS], fp)   
        pickle.dump([probF, probS], fp) 
        pickle.dump(crop_alloc, fp)
        pickle.dump(meta_sol, fp)    

    
# %% b) Get data for visualization
        
meta_l = []
settings_l = []
rhoF_l = []
rhoS_l = []
probF_l = []
probS_l = []
crop_alloc_l = []
meta_sol_l = []
prob_cat_year_l = []
terminal_years_l = []
cat_clusters_l = []
max_areas_l = []             
   
with open("StoOptMultipleYears/OtherParameters/" + \
                      "Default_probF0.95probS0.85_Nc10000.txt", "rb") as fp: 
    meta_l.append(pickle.load(fp))
    settings_l.append(pickle.load(fp))
    [rhoF, rhoS] = pickle.load(fp)
    [probF, probS] = pickle.load(fp)
    crop_alloc_l.append(pickle.load(fp))
    meta_sol_l.append(pickle.load(fp))   
rhoF_l.append(rhoF)
rhoS_l.append(rhoS)
probF_l.append(probF)
probS_l.append(probS)
x_ini, constraints, args, meta, other = OF.SetParameters(settings_l[-1], \
                                                         wo_yields = True)
prob_cat_year_l.append(other["prob_cat_year"])
terminal_years_l.append(args["terminal_years"])
cat_clusters_l.append(args["cat_clusters"])
max_areas_l.append(args["max_areas"])     
    
costs = args["costs"]
prices = args["prices"]

runs = ["k2kPrime1_probF0.95probS0.85_Nc50000"]
                


for run in runs:             
    with open("StoOptMultipleYears/Clusters/" + run + ".txt", "rb") as fp: 
        meta_l.append(pickle.load(fp))
        settings_l.append(pickle.load(fp))
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc_l.append(pickle.load(fp))
        meta_sol_l.append(pickle.load(fp))
    rhoF_l.append(rhoF)
    rhoS_l.append(rhoS)
    probF_l.append(probF)
    probS_l.append(probS)
    x_ini, constraints, args, meta, other = OF.SetParameters(settings_l[-1], \
                                                             wo_yields = True)
    prob_cat_year_l.append(other["prob_cat_year"])
    terminal_years_l.append(args["terminal_years"])
    cat_clusters_l.append(args["cat_clusters"])
    max_areas_l.append(args["max_areas"])
    

# %% c) Penalties and probabilities

print(r"K = 1")
print("Food security penalty rho_F = " + "{:.2e}".format(rhoF_l[0]))
print("Solvency penalty rho_S = " + "{:.2e}".format(rhoS_l[0]))
print("Food security probability alpha_F = " + \
      str(np.round(100*meta_sol_l[0]["prob_food_security"], 2)) + "%")
print("Solvency probability alpha_S = " + \
      str(np.round(100*meta_sol_l[0]["prob_staying_solvent"], 2)) + "%")
print("")
print(r"K = 2")
print("Food security penalty rho_F = " + "{:.2e}".format(rhoF_l[1]))
print("Solvency penalty rho_S = " + "{:.2e}".format(rhoS_l[1]))
print("Food security probability alpha_F = " + \
      str(np.round(100*meta_sol_l[1]["prob_food_security"], 2)) + "%")
print("Solvency probability alpha_S = " + \
      str(np.round(100*meta_sol_l[1]["prob_staying_solvent"], 2)) + "%")


# %% d) Visulaization of crop allocations
titles = ["A", "B"]
fig = plt.figure(figsize = figsize)  
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.2, right=0.9,
                wspace=0.3, hspace=0.3)
for idx, r in enumerate([0, 1]):
    ax = fig.add_subplot(1, 2, idx + 1)
    plt.plot(years,np.sum(crop_alloc_l[r], axis = (1,2)), color = "k", \
             lw = 3, alpha = 0.7)
    for cl in range(0,settings_l[r]["k"]):
        plt.plot(years, np.repeat(max_areas_l[r][cl], len(years)), \
                 color = cols_b[cl], lw = 5, alpha = 0.4)
        plt.plot(years, crop_alloc_l[r][:,0,cl], color = cols[cl], \
                 lw = 2, linestyle = "--")
        plt.plot(years, crop_alloc_l[r][:,1,cl], color = cols[cl], \
                 lw = 2, label = "Cluster " + str(cl + 1))
    plt.legend(loc = 2, \
               bbox_to_anchor=(0.66, 0.95), fontsize = 18, \
               fancybox = True, title_fontsize = "20")
    plt.ylim([-2e6, 7e7])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.offsetText.set_fontsize(20)
    plt.xlabel("Years", fontsize = 26)
    plt.ylabel(r"Crop area in [ha]", fontsize = 26)
    plt.title(titles[r], fontsize = 28)
    
fig.savefig("Figures/StoOpt/Clusters/CropAllocations.png", \
            bbox_inches = "tight", pad_inches = 0.5)         
    



# %% e) expected division of total costs on the three summands

for r in range(0,2):
    total_tmp = meta_sol_l[r]["exp_tot_costs"]
    fix_tmp = meta_sol_l[r]["yearly_fixed_costs"]
    fix_tmp = np.sum(fix_tmp, axis = 1)
    fix_tmp = np.nanmean(fix_tmp)
    sol_tmp = np.nanmean(meta_sol_l[r]["sol_penalty"])
    fd_tmp = meta_sol_l[r]["fd_penalty"]
    fd_tmp = np.sum(fd_tmp, axis = 1)
    fd_tmp = np.nanmean(fd_tmp)
    print("Total expected costs: " + "{:.2e}".format(total_tmp))
    print("Fix costs: " + "{:.2e}".format(fix_tmp) + ", share: " + \
          str(np.round(100*(fix_tmp/total_tmp), 2)) + "%")
    print("FD penalty: " + "{:.2e}".format(fd_tmp) + ", share: " + \
          str(np.round(100*(fd_tmp/total_tmp), 2)) + "%")
    print("Sol penalty: " + "{:.2e}".format(sol_tmp) + ", share: " + \
          str(np.round(100*(sol_tmp/total_tmp), 2)) + "%" + "\n")
  
    
    
# %% f) behvior regarding objectives

# - kcal shortcomings
fig1 = plt.figure(figsize = figsize)  
fig1.subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.9,
                wspace=0.3, hspace=0.3)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
ax = fig1.add_subplot(1, 2, 1)
for r in range(0,2):
    plt.plot(years, np.nanmean(meta_sol_l[r]["S"], axis = 0),\
             color = cols[r], lw = 2)
    print(np.mean(np.nanmean(meta_sol_l[r]["S"], axis = 0)[(-15):]))
    plt.ylim([-8e9, 4e11])
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
ax.xaxis.offsetText.set_fontsize(20)
plt.xlabel("Years", fontsize = 26)
plt.ylabel(r"Food demand shortcomings [kcal]", fontsize = 26)
plt.title("A", fontsize = 28)    
    
# - Distributions of final fund and solvency penalty
props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
ax = fig1.add_subplot(1, 2, 2)
plt.hist(meta_sol_l[0]["final_fund"], alpha = 0.5, \
         color = cols[0], label = r"$K = 1, \alpha_S = $"+ \
         str(np.round(100*np.sum(meta_sol_l[0]["final_fund"] >= 0) / \
                      settings_l[0]["N_c"], 2)) + "%", \
         density = True, bins = np.arange(-0.25e10, 0.85e10, 125000000))
plt.hist(meta_sol_l[1]["final_fund"], alpha = 0.5, \
         color = cols[1], label = r"$K = 2, K' = 1, \alpha_S = $" + \
         str(np.round(100*np.sum(meta_sol_l[1]["final_fund"] >= 0) /  \
                      settings_l[1]["N_c"], 2)) + "%", \
         density = True, bins = np.arange(-0.25e10, 0.85e10, 125000000))
plt.ylim([-0.15e-10,0.9e-9])
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.yaxis.offsetText.set_fontsize(20)
ax.xaxis.offsetText.set_fontsize(20)
plt.xlabel("Final fund in [$]", fontsize = 26)
plt.ylabel("Density", fontsize = 26)
plt.title("B", fontsize = 28)

fig1.savefig("Figures/StoOpt/Clusters/Penalties.png", \
            bbox_inches = "tight", pad_inches = 0.5)         
    

# %% g) Performance of crops regarding different objectives
tradeoff1 = pd.DataFrame(index = ["Rice", "Maize"], \
                        columns = ["$ profit per $ invested", \
                                   "kcal energy per $ invested"])
tradeoff2 = pd.DataFrame(index = ["Rice cluster 1", "Rice cluster 2", \
                                  "Maize cluster 1", "Maize cluster 2"], \
                        columns = ["$ profit per $ invested", \
                                   "kcal energy per $ invested", \
                                   "kcal/$ for cat yields"])

safety2 = pd.DataFrame(index = ["Rice cluster 1", "Rice cluster 2", \
                                  "Maize cluster 1", "Maize cluster 2"], \
                        columns = ["Mean [t/ha]", \
                                   "Std", \
                                   "5% thresholds", \
                                   "Expected cat yield", \
                                   "Exp yields in kcal/ha"])


# column 1: (mean yields * prices - costs)/costs
# column 2: mean yields / costs

kcal_rice = 360 * 10000             # [kcal/t]
# Maize: NDB_No 20014, "CORN GRAIN,YEL" (kcal/100g)
kcal_maize = 365 * 10000            # [kcal/t]
crop_cal = np.array([kcal_rice, kcal_maize])
costs = costs.flatten()
prices = prices.flatten()

with open("IntermediateResults/LinearRegression/GDHY/" + \
                                          "DetrYieldAvg_k1.txt", "rb") as fp:   
     yields_avg = pickle.load(fp)
     avg_pred = pickle.load(fp)
     residuals = pickle.load(fp)
     residual_means = pickle.load(fp)
     residual_stds = pickle.load(fp)
     fstat = pickle.load(fp)
     constants = pickle.load(fp)
     slopes = pickle.load(fp)
     crops = pickle.load(fp)
     
mean_yields1 = ((2016-1981)*slopes + constants).flatten()
mean_yields_kcal1 = mean_yields1*crop_cal
tradeoff1["kcal energy per $ invested"] = mean_yields_kcal1 / costs
tradeoff1["$ profit per $ invested"] = (mean_yields1 * prices - costs)/costs
 

with open("IntermediateResults/LinearRegression/GDHY/" + \
                                          "DetrYieldAvg_k2.txt", "rb") as fp:   
     yields_avg = pickle.load(fp)
     avg_pred = pickle.load(fp)
     residuals = pickle.load(fp)
     residual_means = pickle.load(fp)
     residual_stds = pickle.load(fp)
     fstat = pickle.load(fp)
     constants = pickle.load(fp)
     slopes = pickle.load(fp)
     crops = pickle.load(fp)
     
mean_yields = (((2016-1981)*slopes).flatten() + constants.flatten())
mean_yields_kcal2 = mean_yields*np.repeat(crop_cal, 2)
tradeoff2["kcal energy per $ invested"] = \
                                    mean_yields_kcal2 / np.repeat(costs, 2)
tradeoff2["$ profit per $ invested"] = (mean_yields * np.repeat(prices, 2) - \
                                       np.repeat(costs, 2))/np.repeat(costs, 2)
         
         
safety2["Mean [t/ha]"] = mean_yields
safety2["Std"] = residual_stds.flatten()
safety2["5% thresholds"] = stats.norm.ppf(0.05, mean_yields, \
                                           residual_stds.flatten())

safety2["Exp yields in kcal/ha"] = mean_yields_kcal2

# expected yield for catastrophe
realizations = np.empty([4, 1000])
quantile = stats.norm.ppf(0.05, 0, 1)
for i in range(0, 1000):
    realizations[:,i] = stats.truncnorm.rvs(- np.inf, quantile, \
                        mean_yields, residual_stds.flatten())

safety2["Expected cat yield"] = np.mean(realizations, axis = 1)   
    
    
tradeoff2["kcal/$ for cat yields"] = \
   (safety2["Expected cat yield"]*np.repeat(crop_cal, 2)) / np.repeat(costs, 2)
        
# %% h) Data for k = 7
   
props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    
with open("StoOptMultipleYears/Clusters/" + 
          "k7kPrime1_probF0.95probS0.85_Nc50000.txt", "rb") as fp: 
    meta = pickle.load(fp)
    settings = pickle.load(fp)
    [rhoF, rhoS] = pickle.load(fp)
    [probF, probS] = pickle.load(fp)
    crop_alloc = pickle.load(fp)
    meta_sol = pickle.load(fp)
    
x_ini, constraints, args, meta, other = OF.SetParameters(settings, \
                                                         wo_yields = True)   
max_areas = args["max_areas"]
prices = args["prices"]
costs = args["costs"]
t_years = args["terminal_years"]
cat_clusters_or = args["cat_clusters"]
cat_clusters = np.nansum(cat_clusters_or, axis = 2)
num_cat = np.zeros(settings["N_c"])
for s in range(0, settings["N_c"]):
    num_cat[s] = cat_clusters[s, int(t_years[s])]
    
payoutcluster = np.empty(settings["N_c"])
payoutcluster.fill(np.nan)
for s in range(0, settings["N_c"]):
    tmp = cat_clusters_or[s, int(t_years[s]), :]
    if np.nansum(tmp) != 1:
        continue
    payoutcluster[s] = np.min(np.where(tmp == 1))
        

# %% i) Visualization of crop allocation

fig = plt.figure(figsize = figsize) 
fig.subplots_adjust(bottom=0.1, top=0.8, left=0.1, right=0.9,
                wspace=0.15, hspace=0.35)
ax = fig.add_subplot(1, 2, 1)
plt.plot(years,np.sum(crop_alloc, axis = (1,2)), color = "k", \
         lw = 2, alpha = 0.7)
for cl in range(0,settings["k"]):
    plt.plot(years, np.repeat(max_areas[cl], len(years)), \
             color = cols_b[cl], lw = 5, alpha = 0.4)
    plt.plot(years, crop_alloc[:,0,cl], color = cols[cl], \
             lw = 2, linestyle = "--")
    plt.plot(years, crop_alloc[:,1,cl], color = cols[cl], \
             lw = 2, label = "Cluster " + str(cl + 1))
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.yaxis.offsetText.set_fontsize(24)
ax.xaxis.offsetText.set_fontsize(24)
plt.xlabel("Years", fontsize = 32)
plt.ylabel(r"Crop area in [ha]", fontsize = 32)
plt.title("A", fontsize = 36)

whichplot = [3,4,7,8,11,12,15]
for cl in range(0,settings["k"]):
    ax = fig.add_subplot(4,4,whichplot[cl])
    plt.plot(years, np.repeat(max_areas[cl], len(years)), \
             color = cols_b[cl], lw = 5, alpha = 0.4)
    plt.plot(years, crop_alloc[:,0,cl], color = cols[cl], \
             lw = 2, linestyle = "--")
    plt.plot(years, crop_alloc[:,1,cl], color = cols[cl], \
             lw = 2, label = "Cluster " + str(cl + 1))
    plt.ylim([-0.5e6, 1.5e7])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.xaxis.offsetText.set_fontsize(16)
    ax.text(0.05, 0.91, "Cluster " + str(int(cl+1)), \
            fontsize = 16, transform = ax.transAxes, \
            verticalalignment = 'top', bbox = props)
    if cl == 0:
        plt.title("                       B", fontsize = 36)
    
    
fig.savefig("Figures/StoOpt/Clusters/K7CropAllocationsTest.png", \
            bbox_inches = "tight", pad_inches = 0.5)   


# %% j) penalties and probabilities

print(r"K = 7")
print("Food security penalty rho_F = " + "{:.2e}".format(rhoF))
print("Solvency penalty rho_S = " + "{:.2e}".format(rhoS))
print("Food security probability alpha_F = " + \
      str(np.round(100*meta_sol["prob_food_security"], 2)) + "%")
print("Solvency probability alpha_S = " + \
      str(np.round(100*meta_sol["prob_staying_solvent"], 2)) + "%")

# %% k) Performance of crops regarding different objectives
tradeoff7 = pd.DataFrame(index = ["Rice cluster 1", "Rice cluster 2", \
                                  "Rice cluster 3", "Rice cluster 4", \
                                  "Rice cluster 5", "Rice cluster 6", \
                                  "Rice cluster 7", \
                                  "Maize cluster 1", "Maize cluster 2", \
                                  "Maize cluster 3", "Maize cluster 4", \
                                  "Maize cluster 5", "Maize cluster 6", \
                                  "Maize cluster 7"], \
                        columns = ["$ profit per $ invested", \
                                   "kcal energy per $ invested"])

savety7 = pd.DataFrame(index = ["Rice cluster 1", "Rice cluster 2", \
                                  "Rice cluster 3", "Rice cluster 4", \
                                  "Rice cluster 5", "Rice cluster 6", \
                                  "Rice cluster 7", \
                                  "Maize cluster 1", "Maize cluster 2", \
                                  "Maize cluster 3", "Maize cluster 4", \
                                  "Maize cluster 5", "Maize cluster 6", \
                                  "Maize cluster 7"], \
                        columns = ["Mean", \
                                   "Std", \
                                   "5% thresholds"])

# column 1: (mean yields * prices - costs)/costs
# column 2: mean yields / costs

kcal_rice = 360 * 10000             # [kcal/t]
# Maize: NDB_No 20014, "CORN GRAIN,YEL" (kcal/100g)
kcal_maize = 365 * 10000            # [kcal/t]
crop_cal = np.array([kcal_rice, kcal_maize])


with open("IntermediateResults/LinearRegression/GDHY/" + \
                                      "DetrYieldAvg_k7.txt", "rb") as fp:   
     yields_avg = pickle.load(fp)
     avg_pred = pickle.load(fp)
     residuals = pickle.load(fp)
     residual_means = pickle.load(fp)
     residual_stds = pickle.load(fp)
     fstat = pickle.load(fp)
     constants = pickle.load(fp)
     slopes = pickle.load(fp)
     crops = pickle.load(fp)
     
mean_yields = (((2016-1981)*slopes).flatten() + constants.flatten())
mean_yields_kcal = mean_yields*np.repeat(crop_cal, 7)
tradeoff7["kcal energy per $ invested"] = mean_yields_kcal /costs.flatten()
tradeoff7["$ profit per $ invested"] = (mean_yields * prices.flatten() - \
         costs.flatten())/costs.flatten()
         
savety7["Mean"] = mean_yields
savety7["Std"] = residual_stds.flatten()
savety7["5% thresholds"] = stats.norm.ppf(0.05, mean_yields, \
                                           residual_stds.flatten())
 
# %% l) runs with K = 1 but penalties of higher K

with open("StoOptMultipleYears/OtherParameters/" + \
          "Default_probF0.95probS0.85_Nc10000.txt", "rb") as fp: 
    meta_p1 = pickle.load(fp)
    settings_p1 = pickle.load(fp)
    [rhoF_p1, rhoS_p1] = pickle.load(fp)
    [probF_p1, probS_p1] = pickle.load(fp)
    crop_alloc_p1 = pickle.load(fp)
    meta_sol_p1 = pickle.load(fp)
    
with open("StoOptMultipleYears/Clusters/" + \
          "k1_WithPenaltiesOfk2kPrime1_probF0.95probS0.85.txt", "rb") as fp: 
    meta_p2 = pickle.load(fp)
    settings_p2 = pickle.load(fp)
    [rhoF_p2, rhoS_p2] = pickle.load(fp)
    [probF_p2, probS_p2] = pickle.load(fp)
    crop_alloc_p2 = pickle.load(fp)
    meta_sol_p2 = pickle.load(fp)
    
with open("StoOptMultipleYears/Clusters/" + \
          "k1_WithPenaltiesOfk7kPrime1_probF0.95probS0.85.txt", "rb") as fp: 
    meta_p7 = pickle.load(fp)
    settings_p7 = pickle.load(fp)
    [rhoF_p7, rhoS_p7] = pickle.load(fp)
    [probF_p7, probS_p7] = pickle.load(fp)
    crop_alloc_p7 = pickle.load(fp)
    meta_sol_p7 = pickle.load(fp)
    
print("Probabilities for K = 1:")
print("Food security: " + \
      str(np.round(100*meta_sol_p1["prob_food_security"],2)) + "%")
print("Solvency: " + \
      str(np.round(100*meta_sol_p1["prob_staying_solvent"],2)) + "%")
print("")
print("Probabilities for K = 1 using penalties of K = 2:")
print("Food security: " + \
      str(np.round(100*meta_sol_p2["prob_food_security"],2)) + "%")
print("Solvency: " + \
      str(np.round(100*meta_sol_p2["prob_staying_solvent"],2)) + "%")
print("")
print("Probabilities for K = 1 using penalties of K = 7:")
print("Food security: " + \
      str(np.round(100*meta_sol_p7["prob_food_security"],2)) + "%")
print("Solvency: " + \
      str(np.round(100*meta_sol_p7["prob_staying_solvent"],2)) + "%")

###############################################################################

# %% ############################# 5. FINAL RUN ###############################

# %% a) Run Model

k = 7         
#k = 2               
num_cl_cat = 1
yield_projection = "trend"
pop_scenario = "Medium"                 
N_c = 10000

probFs = [0.95]
probSs = [0.85]

np.warnings.filterwarnings('ignore')

for probF in probFs:
    for probS in probSs:
        print("probF: " +  str(probF)+ ", probS: " + str(probS))
        crop_alloc, meta_sol, rhoF, rhoS, settings, args = \
                OF.OptimizeFoodSecurityProblem(probF, probS, \
                                   rhoFini = 0.00086, rhoSini = 100, \
                                   k = k, num_cl_cat = num_cl_cat, \
                                   yield_projection = yield_projection, \
                                   pop_scenario = pop_scenario, N_c = N_c) 
        with open("StoOptMultipleYears/FinalRun/" + \
                      "k" + str(k) + "kPrime" + str(num_cl_cat) + \
                      "_probF" + str(probF) + "probS" + \
                      str(probS) + ".txt", "wb") as fp: 
            pickle.dump(["settings", "penalties", "probs", \
                         "crop_alloc", "meta_sol"], fp)
            pickle.dump(settings, fp)
            pickle.dump([rhoF, rhoS], fp)   
            pickle.dump([probF, probS], fp) 
            pickle.dump(crop_alloc, fp)
            pickle.dump(meta_sol, fp)    
            
                        
# %% rerunning with higher sample size

runs = ["k7kPrime1_probF0.95probS0.85"]
    
for run in runs:
    with open("StoOptMultipleYears/FinalRun/" + run + ".txt", "rb") as fp: 
        meta = pickle.load(fp)
        settings = pickle.load(fp)
        [rhoF, rhoS] = pickle.load(fp)
        [probF, probS] = pickle.load(fp)
        crop_alloc = pickle.load(fp)
        meta_sol = pickle.load(fp)
                      
    settings["N_c"] = 50000
    print("Get Parameters", flush = True)
    x_ini, const, args, meta_cobyla, other= OF.SetParameters(settings)
    
    print("Start optimization", flush = True)
    crop_alloc, meta_sol, duration = \
                            OF.OptimizeMultipleYears(crop_alloc.flatten(), \
                                        const, args, meta_cobyla, rhoF, rhoS) 
    print(duration, flush = True)
    with open("StoOptMultipleYears/FinaRun/" +  \
               run + "_Nc50000.txt", "wb") as fp: 
        pickle.dump(["settings", "penalties", "probs", \
                     "crop_alloc", "meta_sol"], fp)
        pickle.dump(settings, fp)
        pickle.dump([rhoF, rhoS], fp)   
        pickle.dump([probF, probS], fp) 
        pickle.dump(crop_alloc, fp)
        pickle.dump(meta_sol, fp)           
        
        
# %% Visualize - Data
    
with open("StoOptMultipleYears/FinalRun/" + \
          "k7kPrime1_probF0.95probS0.85_Nc50000.txt", "rb") as fp: 
    meta_fr = pickle.load(fp)
    settings_fr = pickle.load(fp)
    [rhoF_fr, rhoS_fr] = pickle.load(fp)
    [probF_fr, probS_fr] = pickle.load(fp)
    crop_alloc_fr = pickle.load(fp)
    meta_sol_fr = pickle.load(fp)
    
x_ini_fr, constraints_fr, args_fr, meta_fr, other_fr = \
                        OF.SetParameters(settings_fr, wo_yields = True)   
max_areas_fr = args_fr["max_areas"]

# %% Crop allocation


props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    
fig = plt.figure(figsize = figsize) 
fig.subplots_adjust(bottom=0.1, top=0.8, left=0.1, right=0.9,
                wspace=0.15, hspace=0.35)
ax = fig.add_subplot(1, 2, 1)
plt.plot(years,np.sum(crop_alloc_fr, axis = (1,2)), color = "k", \
         lw = 2, alpha = 0.7)
for cl in range(0,settings_fr["k"]):
    plt.plot(years, np.repeat(max_areas_fr[cl], len(years)), \
             color = cols_b[cl], lw = 5, alpha = 0.4)
    plt.plot(years, crop_alloc_fr[:,0,cl], color = cols[cl], \
             lw = 2, linestyle = "--")
    plt.plot(years, crop_alloc_fr[:,1,cl], color = cols[cl], \
             lw = 2, label = "Cluster " + str(cl + 1))
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.yaxis.offsetText.set_fontsize(24)
ax.xaxis.offsetText.set_fontsize(24)
plt.xlabel("Years", fontsize = 32)
plt.ylabel(r"Crop area in [ha]", fontsize = 32)
plt.title("A", fontsize = 36)
    
whichplot = [3,4,7,8,11,12,15]
for cl in range(0,settings_fr["k"]):
    ax = fig.add_subplot(4,4,whichplot[cl])
    plt.plot(years, np.repeat(max_areas_fr[cl], len(years)), \
             color = cols_b[cl], lw = 5, alpha = 0.4)
    plt.plot(years, crop_alloc_fr[:,0,cl], color = cols[cl], \
             lw = 2, linestyle = "--")
    plt.plot(years, crop_alloc_fr[:,1,cl], color = cols[cl], \
             lw = 2, label = "Cluster " + str(cl + 1))
    plt.ylim([-0.5e6, 1.5e7])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.xaxis.offsetText.set_fontsize(16)
    ax.text(0.05, 0.91, "Cluster " + str(int(cl+1)), \
            fontsize = 16, transform = ax.transAxes, \
            verticalalignment = 'top', bbox = props)
    if cl == 0:
        plt.title("                       B", fontsize = 36)    
    
fig.savefig("Figures/StoOpt/FinalRun/K7CropAllocations.png", \
            bbox_inches = "tight", pad_inches = 0.5)   

# %% penalties and probabilities

print(r"K = 7, final run")
print("Food security penalty rho_F = " + "{:.2e}".format(rhoF_fr))
print("Solvency penalty rho_S = " + "{:.2e}".format(rhoS_fr))
print("Food security probability alpha_F = " + \
      str(np.round(100*meta_sol_fr["prob_food_security"])) + "%")
print("Solvency probability alpha_S = " + \
      str(np.round(100*meta_sol_fr["prob_staying_solvent"])) + "%")  

          
###############################################################################

# %% ################################ &. VSS ##################################

# as the solver works with liniearization, it fails for the case N_c = 1, with 
# a piecewise linear function consisting of two linear parts. Therefore, this
# is a workaround, directly calculating the best crop allocations for average#
# yields.
# As in case of average yields it is in general cheaper to produce enough food
# than to pay the penalty, the crops with the highest kcal/$ will be used until
# the food demand is exactly met.
# Furthermore, this approach only works as long as the crops with a better 
# production per dollar can produce the food demand within the area limits,
# as the subsitution between the cheaper crop and the crop with higher yields
# per area is not included.

# %% settings for K = 7
k = 7
pop_scenario = "Medium"
yield_projection = "trend"
N_c = 10000
probF = 0.95
probS = 0.85
corresponding_file="FinalRun/k7kPrime1_probF0.95probS0.85_Nc50000"

# deterministic solution using expected yields
crop_alloc_det, args = OF.DetSolution(k = k, pop_scenario = pop_scenario, \
                                yield_projection = yield_projection, N_c = N_c)


# %% Visualize deterministic vs. robust crop allocations for K = 7

fig = plt.figure(figsize = figsize)
ax = fig.add_subplot(111)  
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, \
               left=False, right=False)
# labels
ax.set_xlabel("Years", fontsize = 32,labelpad=20)
ax.set_ylabel("Crop area in [ha]", fontsize = 32,labelpad=20)
sp = fig.subplots(2,7, sharex=True, sharey = True)
fig.subplots_adjust(bottom=0.2, top=0.7, left=0.1, right=0.9,
                wspace=0.15, hspace=0.35)        
# - deterministic  
for cl in range(0, args["k"]):
    sp[0][cl].plot(years, crop_alloc_det[:, 0, cl], lw = 3, color = cols[cl])
    sp[0][cl].plot(years, crop_alloc_det[:, 1, cl], lw = 3, color = cols[cl], \
                   linestyle = "--")
    sp[0][cl].plot(years, np.repeat(args["max_areas"][cl], len(years)), \
             color = cols_b[cl], lw = 5, alpha = 0.3)
#    plt.ylim([-0.5e6, 1.5e7])
    sp[0][cl].xaxis.set_tick_params(labelsize=18)
    sp[0][cl].yaxis.set_tick_params(labelsize=18)
    sp[0][cl].yaxis.offsetText.set_fontsize(18)
            
    
# - robust
with open("StoOptMultipleYears/" + corresponding_file + ".txt", "rb") as fp: 
    meta = pickle.load(fp)
    settings = pickle.load(fp)
    [rhoF, rhoS] = pickle.load(fp)
    [probF, probS] = pickle.load(fp)
    crop_alloc = pickle.load(fp)
    meta_sol = pickle.load(fp)       
      
for cl in range(0, args["k"]):
    sp[1][cl].plot(years, crop_alloc[:, 0, cl], color = cols[cl], lw = 3)
    sp[1][cl].plot(years, crop_alloc[:, 1, cl], color = cols[cl], lw = 3, \
                   linestyle = "--")
    sp[1][cl].plot(years, np.repeat(args["max_areas"][cl], len(years)), \
             color = cols_b[cl], lw = 5, alpha = 0.3)
#    plt.ylim([-0.5e6, 1.5e7])
    sp[1][cl].xaxis.set_tick_params(labelsize=18)
    sp[1][cl].yaxis.set_tick_params(labelsize=18)
    sp[1][cl].yaxis.offsetText.set_fontsize(18)
plt.show()

fig.savefig("Figures/StoOpt/VSS/K7FinalRun.png", \
            bbox_inches = "tight", pad_inches = 0.5)  


# %% value of stochastic solution for K = 7
np.warnings.filterwarnings('ignore')
meta_sol_det, settings, rhoF, rhoS = OF.VSS(crop_alloc_det, \
                                           probF, probS, N_c = N_c, k = k)    

print("K = " + str(k) + ", pop_scenario: " + pop_scenario + \
      ", yield_projection: " + yield_projection)
print("RS: " + "{:.2e}".format(meta_sol["exp_tot_costs"]))
print("EV: " + "{:.2e}".format(meta_sol_det["exp_tot_costs"]))
print("VSS: " + "{:.2e}".format(meta_sol_det["exp_tot_costs"] - \
                                  meta_sol["exp_tot_costs"]))
print("VSS as percentage of RS: " + \
      str(np.round(((meta_sol_det["exp_tot_costs"] - \
      meta_sol["exp_tot_costs"])/(meta_sol["exp_tot_costs"]) * 100),2)) + "%")

# %% value of stochastic solution for K = 1

k = 1
pop_scenario = "fixed"
yield_projection = "fixed"
N_c = 10000
probF = 0.95
probS = 0.85

corresponding_file = "Scenarios/fixedYldsfixedPop_probF0.95probS0.85_Nc10000"

crop_alloc_det, args = OF.DetSolution(k = k, pop_scenario = pop_scenario, \
                                yield_projection = yield_projection, N_c = N_c)

with open("StoOptMultipleYears/" + corresponding_file + ".txt", "rb") as fp: 
    meta = pickle.load(fp)
    settings = pickle.load(fp)
    [rhoF, rhoS] = pickle.load(fp)
    [probF, probS] = pickle.load(fp)
    crop_alloc = pickle.load(fp)
    meta_sol = pickle.load(fp)       
    
np.warnings.filterwarnings('ignore')
meta_sol_det, settings, rhoF, rhoS = OF.VSS(crop_alloc_det, \
                                           probF, probS, N_c = N_c, k = k)    

print("K = " + str(k) + ", pop_scenario: " + pop_scenario + \
      ", yield_projection: " + yield_projection)
print("RS: " + "{:.2e}".format(meta_sol["exp_tot_costs"]))
print("EV: " + "{:.2e}".format(meta_sol_det["exp_tot_costs"]))
print("VSS: " + "{:.2e}".format(meta_sol_det["exp_tot_costs"] - \
                                  meta_sol["exp_tot_costs"]))
print("VSS as percentage of RS: " + \
      str(np.round(((meta_sol_det["exp_tot_costs"] - \
      meta_sol["exp_tot_costs"])/(meta_sol["exp_tot_costs"]) * 100),2)) + "%")


# %%value of stochastic solution for K = 2

k = 2
pop_scenario = "fixed"
yield_projection = "fixed"
N_c = 10000
probF = 0.95
probS = 0.85
corresponding_file = "Clusters/k2kPrime1_probF0.95probS0.85_Nc50000"

crop_alloc_det, args = OF.DetSolution(k = k, pop_scenario = pop_scenario, \
                                yield_projection = yield_projection, N_c = N_c)

with open("StoOptMultipleYears/" + corresponding_file + ".txt", "rb") as fp: 
    meta = pickle.load(fp)
    settings = pickle.load(fp)
    [rhoF, rhoS] = pickle.load(fp)
    [probF, probS] = pickle.load(fp)
    crop_alloc = pickle.load(fp)
    meta_sol = pickle.load(fp)       
    
np.warnings.filterwarnings('ignore')
meta_sol_det, settings, rhoF, rhoS = OF.VSS(crop_alloc_det, \
                                           probF, probS, N_c = N_c, k = k)    

print("K = " + str(k) + ", pop_scenario: " + pop_scenario + \
      ", yield_projection: " + yield_projection)
print("RS: " + "{:.2e}".format(meta_sol["exp_tot_costs"]))
print("EV: " + "{:.2e}".format(meta_sol_det["exp_tot_costs"]))
print("VSS: " + "{:.2e}".format(meta_sol_det["exp_tot_costs"] - \
                                  meta_sol["exp_tot_costs"]))
print("VSS as percentage of RS: " + \
      str(np.round(((meta_sol_det["exp_tot_costs"] - \
      meta_sol["exp_tot_costs"])/(meta_sol["exp_tot_costs"]) * 100),2)) + "%")