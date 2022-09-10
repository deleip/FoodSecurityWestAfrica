# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 13:35:40 2022

@author: leip
"""

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import pickle
import numpy as np
import matplotlib.pyplot as plt

# import all project related functions
import FoodSecurityModule as FS  

# %%

alpha = 0.70
cl = 4
y = "trend"
p = "fixed"
N = 10000

settings, args, yield_information, population_information, \
status, durations, exp_incomes, crop_alloc, meta_sol, \
crop_allocF, meta_solF, crop_allocS, meta_solS, \
crop_alloc_vs, meta_sol_vss, VSS_value, validation_values, fn = \
    FS.FoodSecurityProblem(k_using = cl,
                           # plotTitle = "Food security probability " + str(alpha * 100) + "%: cluster " + str(cl),
                           N = N,
                           probF = alpha,
                           yield_projection = y,
                           pop_scenario = p)
    
    
fig = plt.figure()
plt.hist(args["ylds"][:,:,0,0].flatten(), bins = 100, color = "yellow", alpha = 0.4)
plt.hist(args["ylds"][:,:,1,0].flatten(), bins = 100, color = "green", alpha = 0.4)

np.nansum(args["ylds"][:,:,0,0] < 0.3)
np.nanmin(args["ylds"][:,:,0,0])
np.nansum(args["ylds"][:,:,1,0] < 0.3)
np.nanmin(args["ylds"][:,:,1,0])