# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:51:17 2021

@author: leip
"""
# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

        
# %% ############## W/O COOPERATION, ALL YIELD AND POP SCENARIOS ############## 

print("\n SECTION 1 \n\n")

for y in ["fixed", "trend"]:
    for p in ["fixed", "Low", "Medium", "High"]:
        for cl in range(1, 10):
            print("\u2017"*65, flush = True)
            print("Yields " + y + ", population trend " + p.lower() + ": cluster " + str(cl), flush = True)
            print("\u033F "*65, flush = True)
            
            settings, args, yield_information, population_information, \
            status, durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = 10000,
                                       # plotTitle = "Yields " + y + ", population trend " + p.lower() + ": cluster " + str(cl),
                                       k_using = cl,
                                       N = 50000,
                                       yield_projection = y,
                                       pop_scenario = p)
                
# %% ############### W/O COOPERATION, DIFF. INPUT PROBABILITIES ############### 

print("\n SECTION 2 \n\n")

for (y, p) in [("fixed", "fixed"), 
                ("trend", "Medium"),
                ("fixed", "High"),
                ("trend", "fixed")]:
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        for cl in range(1, 10):
            print("\u2017"*65, flush = True)
            print("Food security probability " + str(alpha * 100) + "%: cluster " + str(cl), flush = True)
            print("\u033F "*65, flush = True)
            
            settings, args, yield_information, population_information, \
            status, durations, exp_incomes, crop_alloc, meta_sol, \
            crop_allocF, meta_solF, crop_allocS, meta_solS, \
            crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.FoodSecurityProblem(validation_size = 5000,
                                       # plotTitle = "Food security probability " + str(alpha * 100) + "%: cluster " + str(cl),
                                       k_using = cl,
                                       N = 10000,
                                       probF = alpha,
                                       yield_projection = y,
                                       pop_scenario = p)
            
            
# %% ################## W/O COOPERATION, DIFF. POLICY LEVERS ################## 

print("\n SECTION 3 \n\n")

for tax in [0.01, 0.05, 0.1]:
    for risk in [0.01, 0.05]:
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            for cl in range(3,4):
                print("\u2017"*65, flush = True)
                print(str(tax*100) + "% tax, " + str(risk*100) + "% risk, " + str(alpha*100) + ": cluster " + str(cl), flush = True)
                print("\u033F "*65, flush = True)
                
                settings, args, yield_information, population_information, \
                status, durations, exp_incomes, crop_alloc, meta_sol, \
                crop_allocF, meta_solF, crop_allocS, meta_solS, \
                crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
                    FS.FoodSecurityProblem(validation_size = 1000,
                                           # plotTitle = str(tax*100) + "% tax, " + str(risk*100) + "% risk: cluster " + str(cl),
                                           k_using = cl,
                                           N = 10000,
                                           tax = tax,
                                           risk = risk,
                                           probF = alpha,
                                           yield_projection = "trend",
                                           pop_scenario = "Medium")