#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:06:23 2021

@author: debbora
"""
import numpy as np
import time as tm 
import itertools as it
import warnings as warn
import gurobipy as gp

from ModelCode.Auxiliary import printing
from ModelCode.Auxiliary import flatten
from ModelCode.MetaInformation import GetMetaInformation

# %% ############ IMPLEMENTING AND SOLVING LINEAR VERSION OF MODEL ############

def SolveReducedcLinearProblemGurobiPy(args, rhoF, rhoS, probS = None, prints = True, logs_on = False):
    """
    Sets up and solves the linear form of the food security problem.

    Parameters
    ----------
    args : dict
        Dictionary of arguments needed as model input (as given by 
        SetParameters()).
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.
    prints : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is True.

    Returns
    -------
    status : int
        status of solver (optimal: 2)
    crop_alloc : np.array
        gives the optimal crop areas for all years, crops, clusters
    meta_sol : dict 
        additional information about the model output
    prob : gurobi model
        The food security model that was set up.
    durations : list
        time for setting up the model, time for solving, and total time (in
        sec.)

    """
    printing("\nSolving Model", prints = prints, logs_on = logs_on)
    
    start = tm.time()
    
# no output to console
    env = gp.Env(empty = True)    
    env.setParam('OutputFlag', 0)
    env.start()
    
# problem
    prob = gp.Model("SustainableFoodSecurity", env = env)
    
    # dimension stuff
    T = args["T"]
    K = len(args["k_using"])
    J = args["num_crops"]
    N = args["N"]
    termyear_p1 = args["terminal_years"] + 1
    termyear_p1[termyear_p1 == 0] = T
    termyear_p1 = termyear_p1.astype(int)
    
# index tupes for variables and constraints
    indVfood = flatten([[(t, s) for t in range(0, termyear_p1[s])] \
                        for s in range(0, N)])
    
    indW = flatten(flatten([[[(t, k, s) for t in range(0, termyear_p1[s])] \
                             for k in range(0,K)] for s in range(0, N)]))   
            
    indCultCosts = flatten([[(t, j, k) for (t,j,k) in \
              it.product(range(0,termyear_p1[s]), range(0, J), range(0, K))] \
              for s in range(0, N)])
            
    indMaxArea = list(it.product(range(0, K), range(0, T)))
    indCropsClusters = list(it.product(range(0, J), range(0, K)))
    
# variables
    x = prob.addVars(range(0, T), range(0, J), range(0, K), name = "x")
    Vfood = prob.addVars(indVfood, name = "Vfood")
    Vsol = prob.addVars(range(0, N), name = "Vsol")
    Wgov = prob.addVars(indW, name = "Wgov")


# objective function
    obj = gp.quicksum([1/N * x[t,j,k] * args["costs"][j,k] \
                        for (t,j,k) in indCultCosts] + \
                       [1/N * rhoF * Vfood[t, s] for (t, s) in indVfood] + \
                       [1/N * rhoS * Vsol[s] for s in range(0, N)] + \
                          [0 * Wgov[t, k, s] for (t, k, s) in indW])
    prob.setObjective(obj, gp.GRB.MINIMIZE)
            
         
# constraints 1
    prob.addConstrs((gp.quicksum([x[t, j, k] for j in range(0, J)]) \
                 <= args["max_areas"][k] for (k, t) in indMaxArea), "c_marea")
       
# constraints 2
    prob.addConstrs((gp.quicksum([Vfood[t, s]] + \
                [args["ylds"][s, t, j, k] * x[t, j, k] * args["crop_cal"][j] \
                          for (j, k) in indCropsClusters]) \
                 >= (args["demand"][t] - args["import"]) \
                                 for (t, s) in indVfood), "c_demand")
    
# constraints 3
    prob.addConstrs((gp.quicksum([-Vsol[s]] + \
                        [- args["tax"] * (args["ylds"][s, t, j, k] * \
                                x[t, j, k] * args["prices"][j, k] - \
                                x[t, j, k] * args["costs"][j, k]) \
                           for (j, t, k) in it.product(range(0, J), \
                                 range(0, termyear_p1[s]), range(0, K))] + \
                        [args["cat_clusters"][s, t, k] * Wgov[t, k, s] \
                           for (t, k) in it.product(range(0, termyear_p1[s]), \
                                range(0, K))]) \
                 <= args["ini_fund"] for s in range(0, N)), "c_sol")
        
# constraints 4
    prob.addConstrs((gp.quicksum([- Wgov[t, k, s]] + \
            [- args["ylds"][s, t, j, k] * x[t, j, k] * args["prices"][j, k] + \
            x[t, j, k] * args["costs"][j, k] for j in range(0, J)]) \
         <= - args["guaranteed_income"][t, k] for (t, k, s) in indW), "c_gov")

# solving
    middle = tm.time()
    
    # prob.write("../ForPublication/TestingLinearization" \
    #                                        + "/gurobipy_test.lp")
    # prob.write("../ForPublication/TestingLinearization" \
    #                                        + "/gurobipy_test.mps")
    # return()

    prob.optimize()
    
    end = tm.time()
    durationBuild = middle - start
    durationSolve = end - middle
    durationTotal = end - start
    durations = [durationBuild, durationSolve, durationTotal]
    
    status = prob.status
    
# get results
    crop_alloc = np.zeros((T, J, K))
    meta_sol = []
    
    if status != 2:
        warn.warn("Non-optimal status of solver")
        return(status, crop_alloc, meta_sol, prob, durations)
    else:        
        for t in range(0, T):
            for j in range(0, J):
                for k in range(0, K):
                    crop_alloc[t, j, k] = prob.getVarByName("x[" + str(t) + \
                                        "," + str(j) + "," + str(k) + "]").X
                  
        meta_sol = GetMetaInformation(crop_alloc, args, \
                                                    rhoF, rhoS, probS)
        
        # if meta_sol["num_years_with_losses"] > 0:
        #     warn.warn(str("Please notice that in " + \
        #               str(meta_sol["num_years_with_losses"]) + \
        #               " years/clusters profits are negative."))
            
    # printing("      " + "\u005F" * 21, prints = prints)
    printing("     Time      Setting up model: " + \
            str(np.round(durations[0], 2)) + "s", prints = prints, logs_on = logs_on)
    printing("               Solving model: " + \
            str(np.round(durations[1], 2)) + "s", prints = prints, logs_on = logs_on)
    printing("               Total: " + \
            str(np.round(durations[2], 2)) + "s", prints = prints, logs_on = logs_on) 
    # printing("      " + "\u0305 " * 21, prints = prints)           
                
    return(status, crop_alloc, meta_sol, prob, durations)