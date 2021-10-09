# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 20:51:26 2021

@author: leip
"""
# set the right directory
import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import matplotlib.pyplot as plt
from ModelCode.GeneralSettings import figsize


# %% ########################### GOVERNMENT LEVERS ###########################

# Input probabiliy for food security vs. resulting probability for solvency
# Middle of the road scenario (yield trend and medium population growth)
# Example cluster in main text, other clusters in SI

alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# alphas = [0.99]

for cl in range(1, 10):
    fig = plt.figure(figsize = figsize)
    
    for risk in [0.01, 0.05]:
        for tax in [0.01, 0.05, 0.1]:
            solvency = []
            
            for alpha in alphas:
                
                tmp = FS.ReadFromPanda(output_var = ['Resulting probability for solvency'],
                              k_using = [cl],
                              risk = risk,
                              tax = tax,
                              probF = alpha,
                              yield_projection = "trend",
                              pop_scenario = "Medium")
                
                solvency.append(tmp.loc[:,"Resulting probability for solvency"].values[0])
                
            plt.scatter(alphas, solvency,
                        label = "risk " + str(risk * 100) + "%, tax " + str(tax * 100) + "%")
    plt.legend()
    plt.xlabel("Input probability for food security")
    plt.ylabel("Output probability for solvency")
    
    fig.savefig("Figures/PublicationPlots/Fig2_cl" + str(cl) + ".jpg", 
               bbox_inches = "tight", pad_inches = 1, format = "jpg")
    
    plt.close(fig)

    
    
# %% load one set of results

risk = 0.05
tax = 0.01

alpha = 0.995

y = "trend"
p = "fixed"

cl = 9

N = 100000
M = 1000



for cl in range(1, 10):
    settings, args, yield_information, population_information, \
    status, all_durations, exp_incomes, crop_alloc, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.LoadFullResults(k_using = cl,
                                       yield_projection = y,
                                       pop_scenario = p,
                                       probF = alpha)
             
    if cl == 1:
        crop_allocs = crop_alloc
    else:
        crop_allocs = np.concatenate((crop_allocs, crop_alloc), axis = 2)


tmp = FS.ReadFromPanda(output_var = ['Resulting probability for solvency', 
                                    "Sample size",
                                    "Sample size for validation",
                                    "Filename for full results"],
              k_using = cl,
              risk = risk,
              tax = tax,
              probF = alpha,
              yield_projection = "trend",
              pop_scenario = "fixed")

for cl in range(1, 10):
    settings, args, yield_information, population_information, \
    status, durations, exp_incomes, crop_alloc, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
        FS.FoodSecurityProblem(validation_size =   5000,
                               # plotTitle = "Food security probability " + str(alpha * 100) + "%",
                               k_using = 9,
                               N = 10000,
                               probF = 0.99,
                               yield_projection = "trend",
                               pop_scenario = "fixed")



FS.RemoveRun(k_using = cl,
            risk = risk,
            tax = tax,
            probF = alpha,
            yield_projection = "trend",
            pop_scenario = "fixed",
            N = 10000,
            validation_size = 5000)