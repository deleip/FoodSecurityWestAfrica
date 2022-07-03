#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:59:12 2021

@author: Debbora Leip
"""
import numpy as np
from scipy.special import comb
import scipy.stats as stats
import pickle
import sys

from ModelCode.Auxiliary import _printing
from ModelCode.Auxiliary import _GetDefaults

# %% ############# FUNCTIONS TO GET INPUT FOR FOOD SECURITY MODEL #############

def DefaultSettingsExcept(PenMet = "default",
                          probF = "default",
                          probS = "default", 
                          rhoF = "default",
                          rhoS = "default",
                          solv_const = "default",
                          k = "default",     
                          k_using = "default",
                          num_crops = "default",
                          yield_projection = "default",   
                          sim_start = "default",
                          pop_scenario = "default",
                          risk = "default",                          
                          N = "default", 
                          validation_size = "default",
                          T = "default",
                          seed = "default",
                          tax = "default",
                          perc_guaranteed = "default",
                          ini_fund = "default",
                          food_import = "default",
                          accuracyF_demandedProb = "default", 
                          accuracyS_demandedProb = "default",
                          accuracyF_maxProb = "default", 
                          accuracyS_maxProb = "default",
                          accuracyF_rho = "default",
                          accuracyS_rho = "default", 
                          accuracy_help = "default"):     
    """
    Using the default for all settings not specified, this creates a 
    dictionary of all settings.

    Parameters
    ----------
    PenMet : "prob" or "penalties", optional
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly. The default is defined in ModelCode/DefaultModelSettings.py.
    probF : float, optional
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is defined in
        ModelCode/DefaultModelSettings.py.
    probS : float, optional
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is defined in 
        ModelCode/DefaultModelSettings.py.
    rhoF : float or None, optional 
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoF is None, a initial guess for rhoF will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for reaching 
        food demand. The default is defined in ModelCode/DefaultModelSettings.py.
    rhoS : float or None, optional 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoS is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
        The default is defined in ModelCode/DefaultModelSettings.py.
    solv_const : "on" or "off", optional
        Specifies whether the solvency constraint should be included in the 
        model. If "off", probS and rhoS are ignored, and the penalty for 
        insolvency is set to zero instead.
    k : int, optional
        Number of clusters in which the area is to be devided. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    k_using : "all" or a list of int i\in{1,...,k}, optional
        Specifies which of the clusters are to be considered in the model. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    num_crops : int, optional
        The number of crops that are used. The default is defined in
        ModelCode/DefaultModelSettings.py.
    yield_projection : "fixed" or "trend", optional
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
        The default is defined in ModelCode/DefaultModelSettings.py.
    sim_start : int, optional
        The first year of the simulation. The default is defined in
        ModelCode/DefaultModelSettings.py.
    pop_scenario : str, optional
        Specifies which population scenario should be used. "fixed" uses the
        population of the year prior to the first year of the simulation for
        all years. The other options are 'Medium', 'High', 'Low', 
        'ConstantFertility', 'InstantReplacement', 'ZeroMigration', 
        'ConstantMortality', 'NoChange' and 'Momentum', referring to different
        UN_WPP population scenarios. All scenarios have the same estimates up 
        to (including) 2019, scenariospecific predictions start from 2020
        The default is defined in ModelCode/DefaultModelSettings.py.
    risk : int, optional
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic. The default is defined in
        ModelCode/DefaultModelSettings.py.
    N : int, optional
        Number of yield samples to be used to approximate the expected value
        in the original objective function. The default is defined in
        ModelCode/DefaultModelSettings.py.
    validation_size : None or int, optional
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    T : int, optional
        Number of years to cover in the simulation. The default is defined in 
        ModelCode/DefaultModelSettings.py.
    seed : int, optional
        Seed to allow for reproduction of the results. The default is defined 
        in ModelCode/DefaultModelSettings.py.
    tax : float, optional
        Tax rate to be paied on farmers profits. The default is defined in#
        ModelCode/DefaultModelSettings.py.
    perc_guaranteed : float, optional
        The percentage that determines how high the guaranteed income will be 
        depending on the expected income of farmers in a scenario excluding
        the government. The default is defined in ModelCode/DefaultModelSettings.py.
    ini_fund : float, optional
        Initial fund size. The default is defined in ModelCode/DefaultModelSettings.py.
    food_import : float, optional
        Amount of food that is imported (and therefore substracted from the
        food demand). The default is defined in ModelCode/DefaultModelSettings.py.
    accuracyF_demandedProb : float, optional
        Accuracy demanded from the food availability probability as share of 
        demanded probability (for target prob method). The default is defined in
        ModelCode/DefaultModelSettings.py.
    accuracyS_demandedProb : float, optional
        Accuracy demanded from the solvency probability as share of demanded
        probability (for target prob method). The default is defined in
        ModelCode/DefaultModelSettings.py.
    accuracyF_maxProb : float, optional
        Accuracy demanded from the food demand probability as share of maximum
        probability (for maxProb method). The default is defined in
        ModelCode/DefaultModelSettings.py.
    accuracyS_maxProb : float, optional
        Accuracy demanded from the solvency probability as share of maximum
        probability (for maxProb method). The default is defined in 
        ModelCode/DefaultModelSettings.py.
    accuracyF_rho : float, optional
        Accuracy of the food security penalty given thorugh size of the accuracy
        interval: the size needs to be smaller than final rhoF * accuracyF_rho. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    accuracyS_rho : float, optional
        Accuracy of the solvency penalty given thorugh size of the accuracy
        interval: the size needs to be smaller than final rhoS * accuracyS_rho. 
        The default is defined in ModelCode/DefaultModelSettings.py.
    accuracy_help : float, optional
        If method "MinHelp" is used to find the correct penalty, this defines the 
        accuracy demanded from the resulting necessary help in terms of distance
        to the minimal necessary help (given as share of the minimum nevessary
        help). The default is defined in ModelCode/DefaultModelSettings.py.
        
    Returns
    -------
    settings : dict
        A dictionary that includes all of the above settings.
    """
    
    # getting defaults for the not-specified settings
    PenMet, probF, probS, rhoF, rhoS, solv_const, k, k_using, \
    num_crops, yield_projection, sim_start, pop_scenario, \
    risk, N, validation_size, T, seed, tax, perc_guaranteed, \
    ini_fund, food_import, accuracyF_demandedProb, accuracyS_demandedProb, \
    accuracyF_maxProb, accuracyS_maxProb, accuracyF_rho, \
    accuracyS_rho, accuracy_help = _GetDefaults(PenMet, probF, probS, rhoF, rhoS,
                solv_const, k, k_using, num_crops, yield_projection, 
                sim_start, pop_scenario, risk, N, validation_size, T, 
                seed, tax, perc_guaranteed, ini_fund, food_import,
                accuracyF_demandedProb, accuracyS_demandedProb,
                accuracyF_maxProb, accuracyS_maxProb, accuracyF_rho,
                accuracyS_rho, accuracy_help)

    # making sure the current clusters are given as a list
    if type(k_using) is int:
        k_using = [k_using]
        
    if type(k_using) is tuple:
        k_using = list(k_using)
        
    if k_using == "all":
        k_using = list(range(1, k + 1))
        
    k_using_tmp = k_using.copy()
    
    # This will always be True except when the function is called from 
    # GetResultsToCompare() for multiple subsets of clusters 
    # (e.g. k_using = [(1,2),(3,6)])
    if sum([type(i) is int for i in k_using_tmp]) == len(k_using_tmp):
        k_using_tmp.sort()
   
    # check modus (probabilities or penalties given?)
    if PenMet == "penalties":
        probS = None
        probF = None
    elif PenMet != "prob":
        if PenMet != "NotApplicable":
            sys.exit("A non-valid penalty method was chosen (PenMet must " + \
                     "be either \"prob\" or \"penalties\").")     
        
    # create dictionary of settings
    settings =  {"PenMet": PenMet,
                 "probF": probF,
                 "probS": probS,
                 "rhoF": rhoF,
                 "rhoS": rhoS,
                 "solv_const": solv_const,
                 "k": k,
                 "k_using": k_using_tmp,
                 "num_crops": num_crops,
                 "yield_projection": yield_projection,
                 "sim_start": sim_start, 
                 "pop_scenario": pop_scenario,
                 "risk": risk,
                 "N": N,
                 "validation_size": validation_size,
                 "T": T,
                 "seed": seed, 
                 "tax": tax,
                 "perc_guaranteed": perc_guaranteed,
                 "ini_fund": ini_fund,
                 "import": food_import,
                 "accuracyF_demandedProb": accuracyF_demandedProb,
                 "accuracyS_demandedProb": accuracyS_demandedProb,
                 "accuracyF_maxProb": accuracyF_maxProb,
                 "accuracyS_maxProb": accuracyS_maxProb,
                 "accuracyF_rho": accuracyF_rho,
                 "accuracyS_rho": accuracyS_rho,
                 "accuracy_help": accuracy_help}   
     
    return(settings)

def SetParameters(settings, 
                  expected_incomes = None,
                  wo_yields = False,
                  VSS = False, 
                  console_output = None, 
                  logs_on = None):
    """
    
    Based on the settings, this sets most of the parameters that are needed as
    input to the model.    
    
    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    expected_incomes : np.array of size (len(k_using),)
        The expected income of farmers in a scenario where the government is
        not involved.
    wo_yields : boolean, optional
        If True, the function will do everything execept generating the yield
        samples (and return an empty list as placeholder for the ylds 
        parameter). The default is False.
    VSS : boolean, optional
        If True, instead of yield samples only the average yields will be 
        returned and all clusters will be indicated as non-catastrophic by 
        cat_clusters, as needed to calculate the deterministic solution on 
        which the VSS is based. The default is False.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. The default is defined in ModelCode/GeneralSettings.
    logs_on : boolean, optional
        Specifying whether the progress should be documented in a log document.
        The default is defined in ModelCode/GeneralSettings.
        

    Returns
    -------
    args : dict
        Dictionary of arguments needed as model input. 
        
        - k: number of clusters in which the area is devided.
        - k_using: specifies which of the clusters are to be considered in 
          the model. 
        - num_crops: the number of crops that are used
        - N: number of yield samples to be used to approximate the 
          expected value in the original objective function
        - cat_cluster: np.array of size (N, T, len(k_using)), 
          indicating clusters with yields labeled as catastrophic with 1, 
          clusters with "normal" yields with 0
        - terminal years: np.array of size (N,) indicating the year in 
          which the simulation is terminated (i.e. the first year with a 
          catastrophic cluster) for each sample
        - ylds: np.array of size (N, T, num_crops, len(k_using)) of
          yield samples in 10^6t/10^6ha according to the presence of 
          catastrophes
        - costs: np array of size (num_crops,) giving cultivation costs 
          for each crop in 10^9$/10^6ha
        - demand: np.array of size (T,) giving the total food demand
          for each year in 10^12kcal
        - ini_fund: initial fund size in 10^9$
        - import: given import that will be subtraced from demand in 10^12kcal
        - tax: tax rate to be paied on farmers profits
        - prices: np.array of size (num_crops,) giving farm gate prices 
          farmers earn in 10^9$/10^6t
        - T: number of years covered in model          
        - guaranteed_income: np.array of size (T, len(k_using)) giving 
          the income guaranteed by the government for each year and cluster
          in case of catastrophe in 10^9$, based on expected income
        - crop_cal : np.array of size (num_crops,) giving the calorie content
          of the crops in 10^12kcal/10^6t
        - max_areas: np.array of size (len(k_using),) giving the upper 
          limit of area available for agricultural cultivation in each
          cluster
        - probF : float or None, gigving demanded probability of keeping the
          food demand constraint.
        - probS : float or None, giving demanded probability of keeping the 
          solvency constraint.
    yield_information : dict
        Dictionary giving additional information on the yields
     
        - slopes: slopes of the yield trends
        - constants: constants of the yield trends
        - yld_means: average yields 
        - residual_stds: standard deviations of the residuals of the yield 
          trends
        - prob_cat_year: probability for a catastrophic year given the 
          covered risk level
        - share_no_cat: share of samples that don't have a catastrophe within 
          the considered timeframe
        - y_profit: minimal yield per crop and cluster needed to not have 
          losses
        - share_rice_np: share of cases where rice yields are too low to 
          provide profit
        - share_maize_np: share of cases where maize yields are too low to 
          provide profit
        - exp_profit: expected profits in 10^9$/10^6ha per year, crop, cluster
    population_information: dict
        Dictionary giving additional information on the population size in the
        model.
        
        - population : np.array of size (T,) giving estimates of population 
          in considered clusters from simulation start to end for given population 
          scenario (from UN PopDiv)
        - total_pop_scen : np.array of size (T,) giving estimates of population 
          in West Africa from simulation start to end for given population 
          scenario (from UN PopDiv)
        - pop_cluster_ratio2015 : np.array of size (len(k_using),) giving the
          share of population of a cluster to total population of West Africa
          in 2015.
    """
    
    crops = ["Rice", "Maize"]
    
    # 0. extract settings from dictionary
    k = settings["k"]
    k_using = settings["k_using"]
    num_crops = settings["num_crops"]
    yield_projection = settings["yield_projection"]
    sim_start = settings["sim_start"]
    pop_scenario = settings["pop_scenario"]
    risk = settings["risk"]
    if VSS:
        N = 1
    else:
        N = settings["N"]
    T = settings["T"]
    seed = settings["seed"]
    tax = settings["tax"]
    perc_guaranteed = settings["perc_guaranteed"]
    if expected_incomes is None:
        expected_incomes = np.zeros(len(k_using))
    
    # 1. get cluster information (clusters given by k-Medoids on SPEI data) 
    with open("InputData/Clusters/Clustering/kMediods" + \
                        str(k) + "_PearsonDistSPEI.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
    
    # 2. calculate total area and area proportions of the different clusters
    with open("InputData/Population/land_area.txt", "rb") as fp:    
        land_area = pickle.load(fp)
    cluster_areas = np.zeros(len(k_using))
    for i, cl in enumerate(k_using):
        cluster_areas[i] = np.nansum(land_area[clusters == cl])
    cluster_areas = cluster_areas * 100 # convert sq km to ha
                                        
    # 3. Share of population in the area we use (in 2015):
    with open("InputData/Population/GPW_WA.txt", "rb") as fp:    
        gridded_pop = pickle.load(fp)[3,:,:]
    with open("InputData/Population/" + \
                  "UN_PopTotal_Prospects_WesternAfrica.txt", "rb") as fp:    
        total_pop = pickle.load(fp)
        scenarios = pickle.load(fp)
    total_pop_est_past = total_pop[np.where(scenarios == "Medium")[0][0],:][0:71]
    if pop_scenario == "fixed":
        total_pop_scen = np.repeat(total_pop[np.where(scenarios == "Medium")[0][0],:][(sim_start-1)-1950], T)
    else:
        total_pop_scen = total_pop[np.where(scenarios == pop_scenario)[0][0],:]\
                                    [(sim_start - 1950):(sim_start + T - 1950)]
        total_pop_year_before = total_pop[np.where(scenarios == pop_scenario)[0][0],:]\
                                    [sim_start - 1 - 1950]                            
    total_pop_UN_2015 = total_pop_est_past[2015-1950]
    cluster_pop = np.zeros(len(k_using))
    for i, cl in enumerate(k_using):
        cluster_pop[i] = np.nansum(gridded_pop[clusters == cl])
    total_pop_GPW = np.sum(cluster_pop)
    cluster_pop_ratio_2015 = cluster_pop/total_pop_UN_2015
    total_pop_ratio_2015 = total_pop_GPW/total_pop_UN_2015
    considered_pop_scen = total_pop_scen * total_pop_ratio_2015 # use 2015 ratio to scale down population scneario to considered area

    # 5. Per person/day demand       
    # based on country specific caloric demand from a paper on food waste, 
    # averaged based on area. For more detail see DataPreparation_DoNotRun.py    
    with open("InputData/Other/AvgCaloricDemand.txt", "rb") as fp:
        ppdemand = pickle.load(fp)
    
    # 6. cultivation costs of crops
    # average cultivation costs based on literature data for some West
    # African countries (see DataPreparation_DoNotRun for details and
    # sources)
    with open("InputData/Other/CultivationCosts.txt", "rb") as fp:
        costs = pickle.load(fp)
    costs = np.transpose(np.tile(costs, (len(k_using), 1)))
        
    # 7. Energy value of crops
    with open("InputData/Other/CalorieContentCrops.txt", "rb") as fp:
        crop_cal = pickle.load(fp)
    
    # 8. Food demand
    # based on the demand per person and day (ppdemand) and assuming no change
    # of per capita daily consumption we use UN population scenarios for West 
    # Africa and scale them down to the area we use, using the ratio from 2015 
    # (from gridded GPW data)
    demand = ppdemand * 365 * total_pop_scen
    demand = demand * total_pop_ratio_2015
    # in 10^12 kcal
    demand = 1e-12 * demand

    # 9. guaranteed income as share of expected income w/o government
    # if expected income is not given...
    guaranteed_income = np.repeat(expected_incomes[np.newaxis, :], T, axis=0)    
    guaranteed_income = perc_guaranteed * guaranteed_income                     
    # guaraneteed income per person assumed to be constant over time, 
    # therefore scale with population size
    if not pop_scenario == "fixed":
        total_pop_ratios = total_pop_scen / total_pop_year_before
        guaranteed_income = (guaranteed_income.swapaxes(0,1) * total_pop_ratios).swapaxes(0,1)
          
    # 10. prices for selling crops, per crop and cluster (but same over all clusters)
    with open("InputData//Prices/RegionFarmGatePrices.txt", "rb") as fp:    
        prices = pickle.load(fp)
    prices = np.transpose(np.tile(prices, (len(k_using), 1)))
    
    # 11. thresholds for yields being profitable
    y_profit = costs/prices
        
    # 12. Agricultural Areas: 
    # Landscapes of West Africa - A Window on a Changing World: 
    # "Between 1975 and 2013, the area covered by crops doubled in West 
    # Africa, reaching a total of 1,100,000 sq km, or 22.4 percent, of the 
    # land surface." 
    # (https://eros.usgs.gov/westafrica/agriculture-expansion)
    # => Approximate max. agricultural area by 22.4% of total cluster land 
    #   area (assuming agricultural area is evenly spread over West Africa)
    # Their area extends a bit more north, where agricultural land is 
    # probably reduced due to proximity to desert, so in our region the 
    # percentage of agricultural might be a bit higher in reality. But 
    # assuming evenly spread agricultural area over the area we use is 
    # already a big simplification, hence this will not be that critical.
    max_areas = cluster_areas * 0.224        
    # in 10^6ha
    max_areas = 1e-6 * max_areas

    # 13. generating yield samples
    # using historic yield data from GDHY  
    with open("InputData/YieldTrends/DetrYieldAvg_k" + str(k) + ".txt", "rb") as fp:   
         pickle.load(fp) # yields_avg 
         pickle.load(fp) # avg_pred
         pickle.load(fp) # residuals
         pickle.load(fp) # residual_means
         residual_stds = pickle.load(fp)
         pickle.load(fp) # fstat
         constants = pickle.load(fp)
         slopes = pickle.load(fp)
         pickle.load(fp) # crops
         pickle.load(fp) # years
    residual_stds = residual_stds[:, [i - 1 for i in k_using]]
    constants = constants[:, [i - 1 for i in k_using]] 
    slopes = slopes[:, [i - 1 for i in k_using]]

    # get yield realizations:
    # what is the probability of a catastrophic year for given settings?
    _printing("\nOverview on yield samples", console_output = console_output, logs_on = logs_on)
    prob_cat_year = RiskForCatastrophe(risk, len(k_using))
    _printing("     Prob for catastrophic year: " + \
             str(np.round(prob_cat_year*100, 2)) + "%", \
             console_output = console_output, logs_on = logs_on)    
    # create realizations of presence of catastrophic yields and corresponding
    # yield distributions
    np.random.seed(seed)
    cat_clusters, terminal_years, ylds, yld_means = \
          _YieldRealisations(slopes, constants, residual_stds, sim_start, \
                           N, risk, T, len(k_using), num_crops, \
                           yield_projection, VSS, wo_yields)
    # probability to not have a catastrophe
    no_cat = np.sum(terminal_years == -1) / N
    _printing("     Share of samples without catastrophe: " + str(np.round(no_cat*100, 2)), \
              console_output = console_output, logs_on = logs_on) 
    # share of non-profitable crops
    if wo_yields:
        share_rice_np = 0
        share_maize_np = 0
    else:
        share_rice_np = np.sum(ylds[:,:,0,:] < y_profit[0,:])/np.sum(~np.isnan(ylds[:,:,0,:]))
        _printing("     Share of cases with rice yields too low to provide profit: " + \
                 str(np.round(share_rice_np * 100, 2)), console_output = console_output, \
                 logs_on = logs_on)
        share_maize_np = np.sum(ylds[:,:,1,:] < y_profit[1,:])/np.sum(~np.isnan(ylds[:,:,1,:]))
        _printing("     Share of cases with maize yields too low to provide profit: " + \
                 str(np.round(share_maize_np * 100, 2)), console_output = console_output, \
                 logs_on = logs_on)
    # in average more profitable crop
    exp_profit = yld_means * prices - costs
    avg_time_profit = np.nanmean(exp_profit, axis = 0)
    more_profit = np.argmax(avg_time_profit, axis = 0) # per cluster
    _printing("     On average more profit (per cluster): " + \
             str([crops[i] for i in more_profit]), \
             console_output = console_output, logs_on = logs_on)
    # in average more productive crop
    avg_time_production = np.nanmean(yld_means, axis = 0)
    more_food = np.argmax(avg_time_production, axis = 0)
    _printing("     On average higher productivity (per cluster): " + \
             str([crops[i] for i in more_food]) + "\n", \
             console_output = console_output, logs_on = logs_on)
    
    # 14. group output into different dictionaries
    # arguments that are given to the objective function by the solver
    args = {"k": k,
            "k_using": k_using,
            "num_crops": num_crops,
            "N": N,
            "cat_clusters": cat_clusters,
            "terminal_years": terminal_years,
            "ylds": ylds,
            "costs": costs,
            "demand": demand,
            "ini_fund": settings["ini_fund"],
            "import": settings["import"],
            "tax": tax,
            "prices": prices,
            "T": T,
            "guaranteed_income": guaranteed_income,
            "crop_cal": crop_cal,
            "max_areas": max_areas,
            "probF": settings["probF"],
            "probS": settings["probS"]}
        
    # information not needed by the solver but potentially interesting 
    yield_information = {"slopes": slopes,
             "constants": constants,
             "yld_means": yld_means,
             "residual_stds": residual_stds,
             "prob_cat_year": prob_cat_year,
             "share_no_cat": no_cat,
             "y_profit": y_profit,
             "share_rice_np": share_rice_np,
             "share_maize_np": share_maize_np,
             "exp_profit": exp_profit}
    
    population_information = {"population": considered_pop_scen,
                              "total_pop_scen": total_pop_scen, # full area WA
                              "pop_cluster_ratio2015": cluster_pop_ratio_2015}
        
    return(args, yield_information, population_information)

def RiskForCatastrophe(risk, num_clusters):
    """
    The risk level covered by the government is an input parameter to the 
    model. The resulting probability for a catastrophic year depends on the
    number of clusters used by the model.

    Parameters
    ----------
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    num_clusters : int
        The number of clusters that are included.

    Returns
    -------
    res : The probability for a year to be catastrophic.

    """
    res = 0
    for i in range(1, num_clusters + 1):
        res += comb(num_clusters, i) * (risk**i) * (1-risk)**(num_clusters-i)
             
    return(res)

def _YieldRealisations(yld_slopes, yld_constants, resid_std, sim_start, N, \
                      risk, T, num_clusters, num_crops, \
                      yield_projection, VSS = False, wo_yields = False):  
    """
    Depending on the risk level the occurence of catastrophes is generated,
    which is then used to generate yield samples from the given yield 
    distributions.
    
    Parameters
    ----------
    yld_slopes : np.array of size (num_crops, len(k_using))
        Slopes of the yield trends.
    yld_constants : np.array of size (num_crops, len(k_using))
        Constants of the yield trends.
    resid_std : np.array of size (num_crops, len(k_using))
        Standard deviations of the residuals of the yield trends.
    sim_start : int
        The first year of the simulation.
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function.
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    T : int
        Number of years to cover in the simulation.
    num_clusters : int
        Number of clusters considered by model (corresponding to len(k_using))
    num_crops : int
        The number of crops that are used.
    yield_projection : "fixed" or "trend"
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
    VSS : boolean, optional
        If True, all clusters will be indicated as non-catastrophic, as needed 
        to calculate the deterministic solution on which the VSS is based. 
        The default is False.
    wo_yields : boolean, optional
        If True, the function will do everything execept generating the yield
        samples (and return an empty list as placeholder for the ylds 
        parameter). The default is False.
        
    Returns
    -------
    cat_cluters : np.array of size (N, T, len(k_using)) 
        indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0
    terminal_years : np.array of size (N,) 
        indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample, with 0 for 
        catastrophe in first year, 1 for second year, etc. If no catastrophe 
        happens before T, this is indicated by -1
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        yield samples in 10^6t/10^6ha according to the presence of 
        catastrophes
    yld_means : np.array of size (T, num_crops, len(k_using))
        Average yields in 10^6t/10^6ha.

    """
    
    # generating catastrophes 
    cat_clusters, terminal_years = _CatastrophicYears(risk, \
                                    N, T, num_clusters, VSS)
    
    # generating yields according to catastrophes
    ylds, yld_means = _ProjectYields(yld_slopes, yld_constants, resid_std, sim_start, \
                             N, cat_clusters, terminal_years, T, risk, \
                             num_clusters, num_crops, yield_projection, \
                             VSS, wo_yields)
        
    # set negative yields to zero
    if not wo_yields:
        np.seterr(invalid='ignore') # comparing with np.nans leads to annoying warnings, so we turn them off  
        ylds[ylds<0] = 0
        np.seterr(invalid='warn')
    
    return(cat_clusters, terminal_years, ylds, yld_means)

def _CatastrophicYears(risk, N, T, num_clusters, VSS):
    """
    Given the risk level that is to be covered, this creates a np.array 
    indicating which clusters should be catastrophic for each year.

    Parameters
    ----------
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function.
    T : int
        Number of years to cover in the simulation.
    num_clusters : int
        Number of clusters clusters that are considered
    VSS : boolean
        If True, all clusters will be indicated as non-catastrophic, and 
        average yields will be returned instead of yield samples, as needed 
        to calculate the deterministic solution on which the VSS is based. 

    Returns
    -------
    cat_cluters : np.array of size (N, T, len(k_using)) 
        indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0
    terminal_years : np.array of size (N,) 
        indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample

    """

    if VSS:
        cat_clusters = np.zeros((N, T, num_clusters))
        terminal_years = np.repeat(-1, N) 
        return(cat_clusters, terminal_years)
    
    # generating uniform random variables between 0 and 1 to define years
    # with catastrophic cluster (catastrohic if over threshold)
    threshold = 1 - risk
    cat_clusters = np.random.uniform(0, 1, [N, T, num_clusters])
    cat_clusters[cat_clusters > threshold] = 1
    cat_clusters[cat_clusters <= threshold] = 0
    
    # year is catastrophic if min. one cluster is catastrophic
    cat_years = np.sum(cat_clusters, axis = 2)
    cat_years[cat_years < 1] = 0
    cat_years[cat_years >= 1] = 1
    
    # terminal year is first catastrophic year. If no catastrophic year before
    # T, we set terminal year to -1
    terminal_years = np.sum(cat_years, axis = 1)
    for i in range(0, N):
        if terminal_years[i] == 0:
            terminal_years[i] = -1
        else:
            terminal_years[i] = np.min(np.where(cat_years[i, :] == 1))
    return(cat_clusters, terminal_years)

def _ProjectYields(yld_slopes, yld_constants, resid_std, sim_start, \
                  N, cat_clusters, terminal_years, T, risk, \
                  num_clusters, num_crops, yield_projection, \
                  VSS = False, wo_yields = False):
    """
    Depending on the occurence of catastrophes yield samples are generated
    from the given yield distributions.    

    Parameters
    ----------
    yld_slopes : np.array of size (num_crops, len(k_using))
        Slopes of the yield trends.
    yld_constants : np.array of size (num_crops, len(k_using))
        Constants of the yield trends.
    resid_std : np.array of size (num_crops, len(k_using))
        Standard deviations of the residuals of the yield trends.
    sim_start : int
        The first year of the simulation.
    N : int
        Number of yield samples to be used to approximate the expected value
        in the original objective function.
    cat_cluters : np.array of size (N, T, len(k_using)) 
        indicating clusters with yields labeled as catastrophic with 1, 
        clusters with "normal" yields with 0
    terminal_years : np.array of size (N,) 
        indicating the year in which the simulation is terminated (i.e. the 
        first year with a catastrophic cluster) for each sample, with 0 for 
        catastrophe in first year, 1 for second year, etc. If no catastrophe 
        happens before T, this is indicated by -1
    T : int
        Number of years to cover in the simulation.
    risk : int
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic.
    num_clusters : int
        Number of clusters considered by model (corresponding to len(k_using))
    num_crops : int
        The number of crops that are used.
    yield_projection : "fixed" or "trend"
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
    VSS : boolean, optional
        If True, average yields will be returned instead of yield samples, as 
        needed to calculate the deterministic solution on which the VSS is
        based. The default is False.
    wo_yields : boolean, optional
        If True, the function will do everything execept generating the yield
        samples (and return an empty list as placeholder for the ylds 
        parameter). The default is False.

    Returns
    -------
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        yield samples in 10^6t/10^6ha according to the presence of 
        catastrophes. If wo_yields = True, this is an empty list.
    yld_means : np.array of size (T, num_crops, len(k_using))
        Average yields in 10^6t/10^6ha.

    """    
    # project means of yield distributions (either repeating fixed year or by
    # using trend)
    if yield_projection == "fixed": # for fixed we use the year before start
        year_rel = (sim_start - 1) - 1981
        yld_means = yld_constants + year_rel * yld_slopes
        yld_means = np.repeat(yld_means[np.newaxis, :, :], T, axis = 0)
            
    elif yield_projection == "trend":
        year_rel = sim_start - 1981
        years = np.transpose(np.tile(np.array(range(year_rel, year_rel + \
                                        T)), (num_clusters, num_crops, 1)))
        yld_means = yld_constants + years * yld_slopes
    resid_std = np.repeat(resid_std[np.newaxis,:, :], T, axis=0)
    
    # needed for VSS
    if VSS == True:
        return(np.expand_dims(yld_means, axis = 0), yld_means)
    
    if wo_yields:
        return([], yld_means)
    
    # initializing yield array
    ylds = np.empty([N, T, num_crops, num_clusters])
    ylds.fill(np.nan)
    
    # calculating quantile of standard normal distribution corresponding to 
    # the catastrophic yield quantile for truncnorm fct
    quantile_low = stats.norm.ppf(risk, 0, 1)
    
    # generating yields: for catastrophic clusters from lower quantile, for
    # normal years from upper quantile. Dependence between crops, i.e. if 
    # cluster is catastrophic, both crops get yields from lower quantile of 
    # their distributions
    for run in range(0, N):
        if int(terminal_years[run]) == -1:
            ylds[run, :, :, :] = stats.truncnorm.rvs(quantile_low, np.inf, \
                                                     yld_means, resid_std)
        else:
            term_year = int(terminal_years[run])
            ylds[run, :(term_year+1), :, :] = \
                stats.truncnorm.rvs(quantile_low, np.inf, \
                                    yld_means[:(term_year+1), :, :], \
                                    resid_std[:(term_year+1), :, :])
            for cl in range(0, num_clusters):
                if cat_clusters[run, term_year, cl] == 1:
                    ylds[run, term_year, :, cl] = \
                        stats.truncnorm.rvs(- np.inf, quantile_low, \
                                            yld_means[term_year, :, cl], \
                                            resid_std[term_year, :, cl])
    
    return(ylds, yld_means)  
