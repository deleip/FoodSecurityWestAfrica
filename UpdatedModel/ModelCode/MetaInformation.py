#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:05:49 2021

@author: Debbora Leip
"""
import numpy as np

# %% ########### FUNCTIONS TO GET META INFORMATION ON MODEL OUTPUT ############

def _ObjectiveFunction(x, num_clusters, num_crops, N, \
                    cat_clusters, terminal_years, ylds, costs, demand, imports, \
                    ini_fund, tax, prices, T, guaranteed_income, crop_cal, 
                    rhoF, rhoS): 
    """
    Given input parameters and a crop allocation, this calculates the value
    of the objevtive function, i.e. the total costs, and returns the different
    terms and aspects affecting the result as well.
    
    Parameters
    ----------
    x : np.array of size (T, num_crops, len(k_using),)
        Gives allocation of area to each crop in each cluster.
    num_clusters : int
        The number of cluters that are used.
    num_crops : int
        The number of crops that are used.
    N : int
        Number of yield samples used to approximate the expected value
        in the original objective function.
    cat_clusters : np.array of size (N, T, len(k_using))
        Indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0.
    terminal_years : np.array of size (N,) 
        Indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample.
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        Yield samples in t/ha according to the presence of 
        catastrophes.
    costs : np array of size (num_crops,) 
        Cultivation costs for each crop in 10^9$/10^6ha.
    demand : np.array of size (T,)
        Total food demand for each year in 10^12kcal.
    imports : float
        Amount of food that will be imported and therefore is substracted from
        the demand.
    ini_fund : float
        Initial fund size in 10^9$.
    tax : float
        Tax rate to be paied on farmers profits.
    prices : np.array of size (num_crops,) 
        Farm gate prices.
    T : int
        Number of years to cover in the simulation.
    guaranteed_income : np.array of size (T, len(k_using)) 
        Income guaranteed by the government for each year and cluster in case 
        of catastrophe in 10^9$.
    crop_cal : np.array of size (num_crops,)
        Calorie content of the crops in 10^12kcal/10^6t.
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.
        
    Returns
    -------
    exp_tot_costs :  float
        Final value of objective function, i.e. sum of cultivation and penalty
        costs in 10^9$.
    fixcosts : np.array of size (N,)
        Cultivation costs in 10^9$ for each yield sample (depends only on the 
        final year of simulation for each sample).
    shortcomings : np.array of size (N, T)
        Shortcoming of the food demand in 10^12kcal for each year in each 
        sample.
    avg_profits : np.array of size (T, len(k_using))
        Average Profits of farmers in 10^9$ per cluster and year.
    avg_profits_afterTax : np.array of size (T, len(k_using))
        AverageProfits of farmers in 10^9$ after subtracting tax payments per
        cluster and year.
    profits : np.array of size (N, T, len(k_using))
        Profits of farmers in 10^9$ per cluster and year for each sample.
    profits_afterTax : np.array of size (N, T, len(k_using))
        Profits of farmers in 10^9$ after subtracting tax payments per cluster 
        and year for each sample.
    avg_shortcomings : np.array of size (T,)
        Average shortcoming of the food demand in 10^12kcal in each year.
    fd_penalties : np.array of size (N, T) 
        Penalty payed because of food shortages in each year for each sample.
    avg_fd_penalties : np.array of size (T,)
        Average penalty payed because of food shortages in each year.
    sol_penalties : np.array of size (N,)
        Penalty payed because of insolvency in each sample.
    final_fund : np.array of size (N,)
        The fund size after payouts in the catastrophic year for each sample.
    payouts : np.array of size (N, T, len(k_using))
        Payouts from the government to farmers in case of catastrophe per year
        and cluster for each sample. 
    yearly_fixed_costs : np.array of size (N, T, len(k_using)) 
        Total cultivation costs per cluster in each year for each sample.   
    food_supply : np.array of size (N, T)
        Food supply in kcal (production plus import if any), for each sample
        and year.
    """

    # preparing x for all realizations
    X = np.repeat(x[np.newaxis, :, :, :], N, axis=0)
    for c in range(0, N):
        t_c = int(terminal_years[c])                   # catastrophic year
        if t_c > -1:
            X[c, (t_c + 1) : , :, :] = np.nan  # no crop area after catastrophe
            
    # Production
    prod = X * ylds                          # nan for years after catastrophe
    kcal  = np.swapaxes(prod, 2, 3)
    kcal = kcal * crop_cal                   # relevant for realistic values
    food_supply = imports + np.sum(kcal, axis = (2, 3))
    
    # Shortcomings
    S = demand - food_supply # nan for years after catastrophe
    np.seterr(invalid='ignore')
    S[S < 0] = 0
    np.seterr(invalid='warn')
    
    # fixed costs for cultivation of crops
    fixed_costs =  X * costs
    
    # Yearly profits
    P =  prod*prices - fixed_costs          # still per crop and cluster, 
                                            # nan for years after catastrophe
    P = np.sum(P, axis = 2)                 # now per cluster
    P_afterTax = (1 - tax) * P
    # calculate average profits
    avg_profits = np.nanmean(P, axis = 0) # per year and cluster
    avg_profits_afterTax = np.nanmean(P_afterTax, axis = 0) # per year and cluster
    # P[P < 0] = 0   # we removed the max(0, P) and min(I_gov, I_gov-P) for
                     # linearization prurposes
 
    # Payouts
    payouts = guaranteed_income - P  # as we set negative profit to zero,
                                     # government doesn't cover those
                                     # it only covers up to guaranteed income.
                                     # !! -> this is not true any more !!
                                     # (due o linearization)
    np.seterr(invalid='ignore')
    payouts[(~np.isnan(payouts) * (cat_clusters == 0)) + (payouts < 0)] = 0
    np.seterr(invalid='warn')
                # no payout if there is no catastrophe, even if guaranteed 
                # income is not reached
                # in the unlikely case where a cluster makes more than the
                # guaranteed income despite catastrophe, no negative payouts!
                      
    # Final fund
    ff = ini_fund + tax * np.nansum(P, axis = (1,2)) - \
                                 (1 - tax) * np.nansum(payouts, axis = (1,2))
    ff[ff > 0] = 0
    
    # expected total costs
    exp_tot_costs = np.mean(np.nansum(fixed_costs, axis = (1,2,3)) + \
                            rhoF * np.nansum(S, axis = 1) + rhoS * (- ff))
    
    return(exp_tot_costs, 
        np.nansum(fixed_costs, axis = (1,2,3)), #  fixcosts (N)
        S, # shortcomings per realization and year
        avg_profits, # average profits (T, k) before ta payments
        avg_profits_afterTax, # average final profits after tax payments
        P, # profits before tax payments
        P_afterTax, # final profits after tax payment
        np.nanmean(S, axis = 0), # yearly avg shortcoming (T)
        rhoF * S, # yearly food demand penalty (N x T)
        np.nanmean(rhoF * S, axis = 0), # yearly avg fd penalty (T)
        rhoS * (- ff), # solvency penalty (N)
        ini_fund + tax * np.nansum(P, axis = (1,2)) - \
          np.nansum(payouts, axis = (1,2)), # final fund per realization
        payouts, # government payouts (N, T, k)
        np.nansum(fixed_costs, axis = 2), #  fixcosts (N, T, k)
        food_supply # food supply in kcal (production plus import if any)
        ) 

def GetMetaInformation(crop_alloc, args, rhoF, rhoS):
    """
    To get metainformation on a given crop allocation.

    Parameters
    ----------
    crop_alloc : np.array of size (T*num_crops*len(k_using),)
        Gives allocation of area to each crop in each cluster.
    args : dict
        Dictionary of arguments needed as model input (as given by 
        SetParameters())
    rhoF : float
        The penalty for shortcomings of the food demand.
    rhoS : float
        The penalty for insolvency.
   
    Returns
    -------
    meta_sol : dict 
        additional information about the model output.
        
        - exp_tot_costs: Final value of objective function, i.e. sum of 
          cultivation and penalty costs in 10^9$.
        - fix_costs: Cultivation costs in 10^9$ for each yield sample (depends 
          only on the final year of simulation for each sample).
        - yearly_fixed_costs: Total cultivation costs per cluster in each 
          year for each sample.   
        - fd_penalty: Penalty payed because of food shortages in each year 
          for each sample.
        - avg_fd_penalties: Average penalty payed because of food shortages in 
          each year.
        - sol_penalty: Penalty payed because of insolvency in each sample.
        - shortcomings: Shortcoming of the food demand in 10^12kcal for each year in each 
          sample.
        - exp_shortcomings: Average shortcoming of the food demand in 
          10^12kcal in each year.
        - avg_profits_preTax: Average profits of farmers in 10^9$ before paying
          taxes for each cluster in each year.
        - avg_profits_afterTax: Average profits of farmers in 10^9$ after 
          subtracting tax payments for each cluster in each year.
        - food_supply: Food supply (production plust import if there is any) 
          in 10^12kcal for each year in each sample
        - profits_preTax: Profits of farmers in 10^9$ before paying taxes
          per cluster and year for each sample.
        - profits_afterTax: Profits of farmers in 10^9$ after tax paxments
          per cluster and year for each sample.
        - num_years_with_losses: Number of occurences where farmers of a 
          cluster have negative profits.
        - payouts: Payouts from the government to farmers in case of catastrope 
          per year and cluster for each sample. 
        - final_fund: The fund size after payouts in the catastrophic year for 
          each sample.
        - probF: Probability for meeting the food femand.
        - probS: Probability for solvency of the government fund
          after payouts.
        - avg_nec_import: Average import needed to cover full food demand
        - avg_nec_debt: Average debt needed to cover full government payouts
        - guaranteed_income: guaranteed_income as given in args
    """
    
    # running the objective function with option to get 
    # intermediate results of the calculation
    exp_tot_costs, fix_costs, shortcomings, avg_profits, avg_profits_afterTax, \
    profits, profits_afterTax, exp_shortcomings, fd_penalty, avg_fd_penalty, \
    sol_penalty, final_fund, payouts, yearly_fixed_costs, food_supply \
        = _ObjectiveFunction(crop_alloc, 
                             args["k"], 
                             args["num_crops"],
                             args["N"], 
                             args["cat_clusters"], 
                             args["terminal_years"],
                             args["ylds"], 
                             args["costs"], 
                             args["demand"],
                             args["import"],
                             args["ini_fund"],
                             args["tax"],
                             args["prices"],
                             args["T"],
                             args["guaranteed_income"],
                             args["crop_cal"], 
                             rhoF, 
                             rhoS)
                         
    # calculationg additional quantities:
    # probability of solvency in case of catastrophe
    prob_staying_solvent = np.sum(final_fund >= 0) /  args["N"]
    
    # probability for food security
    tmp = np.copy(shortcomings)
    np.seterr(invalid='ignore')
    tmp[tmp > 0] = 1
    np.seterr(invalid='warn')
    prob_food_security = 1 - np.nanmean(tmp)
    
    # number of years in which farmers have losses
    np.seterr(invalid='ignore')
    num_years_with_losses = np.sum(profits<0, axis = (0,1))
    np.seterr(invalid='warn')
    
    # debt (based on final fund)
    debt = -np.copy(final_fund)
    np.seterr(invalid='ignore')
    debt[debt < 0] = 0
    np.seterr(invalid='warn')
    
    # group information into a dictionary
    meta_sol = {"exp_tot_costs": exp_tot_costs,
                "fix_costs": fix_costs,
                "yearly_fixed_costs": yearly_fixed_costs,
                "fd_penalty": fd_penalty,
                "avg_fd_penalty": avg_fd_penalty,
                "sol_penalty": sol_penalty,
                "shortcomings": shortcomings,
                "exp_shortcomings": exp_shortcomings, # per year
                "avg_profits_preTax": avg_profits,
                "avg_profits_afterTax": avg_profits_afterTax,
                "food_supply": food_supply,
                "profits_preTax": profits,
                "profits_afterTax": profits_afterTax,
                "num_years_with_losses": num_years_with_losses,
                "payouts": payouts,
                "final_fund": final_fund,
                "probF": prob_food_security,
                "probS": prob_staying_solvent,
                "avg_nec_import": np.nanmean(exp_shortcomings),
                "avg_nec_debt": np.nanmean(debt),
                "guaranteed_income": args["guaranteed_income"]}
    
    return(meta_sol)  