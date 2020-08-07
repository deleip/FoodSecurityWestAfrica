#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:18:32 2020

@author: debbora
"""

# %% IMPORTING NECESSARY PACKAGES 

import numpy as np
import scipy.optimize as opt
from scipy.special import comb
import scipy.stats as stats
import pickle
import pandas as pd
import time as tm 

# %% ######### FUNCTIONS STOCHASTIC OPTIMIZATION SINGLE YEAR ##################

# generating yields from given parameters of the distribution
def YieldRealisationsSingleYear(yld_slopes, yld_constants, resid_std, year, \
                                num_realisations, num_clusters, num_crops):    
# yld_slopes, yld_constants: slope and constant of linear trend in yield 
#                            averages, per crop & cluster  
# resid_std: standard deviation of residuals of yield trend per crop & cluster
# year: year to use for yield generations
# num_realisations: number of realisations to generate
# num_clutsers: number of clusters used
# num_crops: number of crops used
    
    # calculated mean of yield distribution from linear trend
    year_rel = year - 1981
    yld_means = yld_constants + year_rel * yld_slopes
    
    # generate yield realisations
    ylds = np.random.normal(yld_means, resid_std, \
                               (num_realisations, num_crops, num_clusters))
    return(ylds)
    
# finding robust solution for crop allocation
def OptimizeSingleYear(crop_areas_ini, constraints, 
                      num_realisations, max_areas, const,
                      crop_costs, penalty, demand, crop_cal,
                      num_crops, num_clusters, seed, rhobeg_cobyla, ylds):
# crop_areas_ini: initial guess for crop allocations
# constraints: constraint functions for crop allocation - no negative crop
#              areas and total area within a cluster below available agri-
#              cultural area in that cluster
# const: a matrix needed for the constraint functions
# seed: seed which is set before calling the solver for reproducebility
# rhobeg_cobyla: initial size of trust region for COBYLA solver
# other parameters are directly passed to objective function
    
    np.random.seed(seed)    
    
    ObjectiveFunctionSingleYear.Calls = 0
    
    # calling solver
    crop_alloc = opt.fmin_cobyla(func = ObjectiveFunctionSingleYear, \
                                 x0 = crop_areas_ini, \
                                 cons = constraints,          
                                 args = (num_realisations, ylds, max_areas, \
                                         crop_costs, penalty, demand, \
                                         crop_cal, num_crops, num_clusters),
                                 consargs = (max_areas, const), \
                                 rhobeg = rhobeg_cobyla)
    
    return(crop_alloc) 

# objective function: total costs that have to be minimized
def ObjectiveFunctionSingleYear(crop_areas, num_realisations, ylds, max_areas, 
                              crop_costs, penalty, demand, crop_cal,
                              num_crops, num_clusters, meta = False):
# crop_areas:       num_crops x num_clusters, [ha]
#                   gives allocation of area to each crop in each cluster
# num_realisations: sample size (to approximated the expected value reasonably)
# ylds:             num_realisations x num_crops x num_clusters, [t/ha]
#                   yield realisations from distribution, per cluster & crop
# max_areas:        1 x num_cluster, [ha]
#                   gives max. available land for agriculture in each cluster
# crop_costs:       num_crops x num_clusters, [$/ha]
#                   gives cluster and crop specific costs of cultivation per ha
# penalty:          1 x 1, [$/kcal]
#                   costs of importing food
# demand:           1 x 1, [kcal]
#                   total food demand (all clusters) given in energetic value
# crop_cal:         1 x num_crops, [kcal/t]
#                   energetic value of crops
# num_crops:        number of crops used
# num_clutsers:     number of clusters used
# meta:             if False only the expected costs are returned, if True some
#                   other information are returned as well
    
    ObjectiveFunctionSingleYear.Calls += 1
    
    # as solver can only take 1d optimization variable, we first 
    # reshape the crop areas to num_crops x num_clusters 
    crop_areas = crop_areas.reshape(crop_costs.shape)        
                  
    # costs of agricultural production (all crops, all clusters)        
    fixed_costs = np.sum(crop_areas * crop_costs)                             
  
    # production per realisation, crop & cluster in t
    prod = ylds * crop_areas 
    kcal  = np.swapaxes(prod, 1, 2)
    # per realisation, cluster & crop in kcal
    kcal = kcal * crop_cal 
    # per realisation
    kcal = np.sum(kcal, (1,2))
    shortcoming = demand - kcal 
    # only need to pay if there really is a shortcoming
    shortcoming[shortcoming < 0] = 0  
    indirect_costs = penalty * shortcoming
    # eypected value approximated by sample mean
    exp_add_costs = np.mean(indirect_costs)
                                         
    # approximated expected value of total costs
    expected_costs = fixed_costs + exp_add_costs                              
                                                                              
    if meta:
        return(expected_costs, fixed_costs, exp_add_costs, shortcoming)
        
    return(expected_costs)


# %% ######### FUNCTIONS STOCHASTIC OPTIMIZATION INCLUDING TIME ###############
   
# get all settings using the default except stated otherwise
# if not given otherwise, the expected income will be calculated
def DefaultSettingsExcept(k = 1,                        
                          num_cl_cat = 1,               # is not varied 
                          num_crops = 2,                # is not varied
                          yield_projection = "fixed",   
                          yield_year = 2017,            # is not varied
                          stilised = False,             # is not varied
                          pop_scenario = "fixed",
                          risk = 20,                          
                          N_c = 3500, 
                          T_max = 25,                   # is not varied
                          seed = 150620,
                          tax = 0.03,
                          perc_guaranteed = 0.75,
                          exp_income = None):     # will be calculated 
                                                  # if not given
                                                  
# input are all settings, which are eplained in StochasticOptimization.py
# for any setting that is not given when the function is called, the default
# value will be used.
            
    # create dictionary of settings
    settings =  {"k": k,
                 "num_cl_cat": num_cl_cat,
                 "num_crops": num_crops,
                 "yield_projection": yield_projection,
                 "yield_year": yield_year, 
                 "stilised": stilised,
                 "pop_scenario": pop_scenario,
                 "risk": risk,
                 "N_c": N_c,
                 "T_max": T_max,
                 "seed": seed, 
                 "tax": tax,
                 "perc_guaranteed": perc_guaranteed, 
                 # format needs to be (1,k) as we calculate it for one year
                 "expected_incomes": np.zeros([1,k])}   
     
    # not all settings affect the expected income (as no governmetn is 
    # included)
    # For k = 1 it is not different because it is actually influenced by more
    # settings, but because I ran this while the model was still slightly
    # different, which however didn't effect k = 1, only runs with k > 1
    if k == 1:
        SettingsAffectingGuaranteedIncome = "k" + str(settings["k"]) + \
                         "num_cl_cat" + str(settings["num_cl_cat"]) + \
                         "num_crops" + str(settings["num_crops"]) + \
                         "yield_year" + str(settings["yield_year"]) + \
                         "stilised" + str(settings["stilised"]) + \
                         "risk" + str(settings["risk"]) + \
                         "N_c" + str(settings["N_c"])
    else:
        SettingsAffectingGuaranteedIncome = "k" + str(settings["k"]) + \
                         "num_crops" + str(settings["num_crops"]) + \
                         "yield_year" + str(settings["yield_year"]) + \
                         "stilised" + str(settings["stilised"]) + \
                         "N_c" + str(settings["N_c"]) 
    
    # if wexpected income is not given...
    if exp_income is None:    
        # open list in which all settings for which expected income was already
        # calculated are saved
        with open("StoOptMultipleYears/ExpectedIncomes/list.txt", "rb") as fp:    
            list_incomes = pickle.load(fp)
        
        # check if the vurrent settings are in the list. if yes load the
        # expected income
        if SettingsAffectingGuaranteedIncome in list_incomes:
            print("Fetching expected income", flush = True)
            with open("StoOptMultipleYears/ExpectedIncomes/" + \
                      SettingsAffectingGuaranteedIncome + ".txt", "rb") as fp:    
                settings["expected_incomes"] = pickle.load(fp)
        # else calculate it (and save it)
        else:
            settings["expected_incomes"] = CalcExpectedIncome(settings, \
                                            SettingsAffectingGuaranteedIncome)
            
    # if expected income is given, set this value in the settings
    else:
        settings["expected_incomes"] = exp_income        
        
    # return dictionary of all settings
    return(settings)

# function to calculate the expected income in 2016 (year before T_0)        
def CalcExpectedIncome(settings, SettingsAffectingGuaranteedIncome):
# settings: dictionary of all settings given by DefaultSettingsExcept()
#           settings are eplained in StochasticOptimization.py
# SettingsAffectingGuaranteedIncome: a string giving all settings that are 
#                                    relevant for calculating the expected 
#                                    income, to check whether it was already
#                                    calculated
    
    print("Calculating expected income (new version)", flush=True)
    settings_tmp = settings.copy()

    # change some settings: we are interested in the expected income in 2016
    settings_tmp["seed"] = 150620
    settings_tmp["yield_projection"] = "fixed"
    settings_tmp["pop_scenario"] = "fixed"
    settings_tmp["T_max"] = 1
    
    # we assume that without government farmers aim for 95% probability of 
    # food security, therefore we find the right penalty for probF = 95%.
    # As we want the income in a scenario without government, the final run of
    # GetRhoF (with rohS = 0) automatically is the right run
    rhoF, crop_alloc, meta_sol = \
           GetRhoF(settings_tmp, 0.95, 1e-2, accuracy = 3)
           
    # save expected income such that it can be reused if same settings are 
    # used for other runs
    with open("StoOptMultipleYears/ExpectedIncomes/" + \
              SettingsAffectingGuaranteedIncome + ".txt", "wb") as fp:    
        pickle.dump(meta_sol["exp_incomes"], fp)   
        pickle.dump(rhoF, fp)
        pickle.dump(crop_alloc, fp) 
        pickle.dump(meta_sol, fp)   
        
    # add settings to the list of calculated incomes, such that it can be
    # checked whether it was already calculated
    with open("StoOptMultipleYears/ExpectedIncomes/list.txt", "rb") as fp:    
        list_incomes = pickle.load(fp)
    list_incomes.append(SettingsAffectingGuaranteedIncome)
    with open("StoOptMultipleYears/ExpectedIncomes/list.txt", "wb") as fp:    
        pickle.dump(list_incomes, fp)    
        
    return(meta_sol["exp_incomes"])

# set other parameters depending on settings
def SetParameters(settings, x_ini = None, wo_yields = False, \
                  returnyieldmean = False):
# settings: dictionary of all settings given by DefaultSettingsExcept()
#           settings are eplained in StochasticOptimization.py
# x_ini: initial guess for the optimizer If not given a first guess will be 
#        calculated by this function
# wo_yields: if this is run to get information like the terminal years, but 
#            yields are not needed as the optimizer is not called, wo_yields
#            can be set to True to exclude generation of yields (which takes 
#            by far the most time in this function, especially for high N_c)
    
    # extract settings from dictionary
    k = settings["k"]
    num_cl_cat = settings["num_cl_cat"]
    num_crops = settings["num_crops"]
    yield_projection = settings["yield_projection"]
    yield_year = settings["yield_year"]
    stilised = settings["stilised"]
    pop_scenario = settings["pop_scenario"]
    risk = settings["risk"]
    N_c = settings["N_c"]
    T_max = settings["T_max"]
    seed = settings["seed"]
    tax = settings["tax"]
    perc_guaranteed = settings["perc_guaranteed"]
    expected_incomes = settings["expected_incomes"]
    
    # LOAD DATA
    # 1. get cluster information (clusters given by k-Medoids on SPEI data) 
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                        str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
    cluster = clusters[-1]
    
    # calculate total area and area proportions of the different clusters
    with open("IntermediateResults/PreparedData/Population/" + \
                                          "land_area.txt", "rb") as fp:    
        land_area = pickle.load(fp)
    cluster_areas = np.zeros(k)
    for cl in range(0, k):
        cluster_areas[cl] = np.nansum(land_area[cluster == (cl + 1)])
    cluster_areas = cluster_areas * 100 # convert sq km to ha
    total_area = np.sum(cluster_areas)
    area_proportions = cluster_areas/total_area
                                        
    # 2. Share of population in the area we use (in 2015):
    with open("IntermediateResults/PreparedData/Population/" + \
                                      "GPW_WA.txt", "rb") as fp:    
        gridded_pop = pickle.load(fp)[3,:,:]
    with open("IntermediateResults/PreparedData/Population/" + \
                  "UN_PopTotal_Prospects_WesternAfrica.txt", "rb") as fp:    
        total_pop = pickle.load(fp)
        scenarios = pickle.load(fp)
    total_pop_scen = total_pop[np.where(scenarios == "Medium")[0][0],:]
    total_pop_UN = total_pop_scen[2015-1950]
    cluster_pop = np.zeros(k)
    for cl in range(0, k):
        cluster_pop[cl] = np.nansum(gridded_pop[cluster == (cl + 1)])
    total_pop_GPW = np.sum(cluster_pop)
    pop_ratio = total_pop_GPW/total_pop_UN
    
    # 3. Country shares of area (for price calculation)
    with open("IntermediateResults/PreparedData/Prices/" + \
                                  "CountryCodesGridded.txt", "rb") as fp:    
        country_codes_gridded = pickle.load(fp)
    country_codes = pd.read_csv("Data/Prices/CountryCodes.csv")
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                   "RiceMaizeJoined_mask.txt", "rb") as fp:   
        yld_mask = pickle.load(fp)
    country_codes_gridded[yld_mask == 0] = np.nan
    total_land_cells = np.sum(yld_mask == 1)
    
    # liberia has no price data
    liberia_landcells = np.sum(country_codes_gridded == country_codes.loc \
                       [(country_codes.CountryName=="Liberia")]["Code"].values)
    total_cells_for_average = total_land_cells - liberia_landcells 
                    
    # Mauritania has no rice price data
    mauritania_landcells = np.sum(country_codes_gridded == country_codes.loc \
                    [(country_codes.CountryName=="Mauritania")]["Code"].values)
    total_cells_for_average_excl_mauritania = \
                            total_cells_for_average - mauritania_landcells 
                            
    country_codes["Shares"] = 0
    country_codes["Shares excl. Mauritania"] = 0
    
    country_codes_gridded[country_codes_gridded == country_codes.loc \
             [(country_codes.CountryName=="Liberia")]["Code"].values] = np.nan
    for idx, c in enumerate(country_codes["Code"].values):
        country_codes.iloc[idx, 2] = \
                    np.sum(country_codes_gridded == c)/total_cells_for_average
    
    country_codes_gridded[country_codes_gridded == country_codes.loc \
           [(country_codes.CountryName=="Mauritania")]["Code"].values] = np.nan
    for idx, c in enumerate(country_codes["Code"].values):
        country_codes.iloc[idx, 3] = np.sum(country_codes_gridded == c)/ \
                                        total_cells_for_average_excl_mauritania
    
    # removing countries with no share as they don't show up in prices df below
    country_codes = country_codes.drop(axis = 0, labels = [4, 5, 9])                                   
    
    
    # 4. load information on yield distributions,
    # using historic yield data from GDHY  
    with open("IntermediateResults/LinearRegression/GDHY/DetrYieldAvg_k" + \
                              str(k) + ".txt", "rb") as fp:   
         yields_avg = pickle.load(fp)
         avg_pred = pickle.load(fp)
         residuals = pickle.load(fp)
         residual_means = pickle.load(fp)
         residual_stds = pickle.load(fp)
         fstat = pickle.load(fp)
         constants = pickle.load(fp)
         slopes = pickle.load(fp)
         crops = pickle.load(fp)
    
    
    # SET OTHER VALUES
    if stilised:
        # cultivation costs for different crops in different clusters
        costs = np.ones([num_crops, k])
        # energetic value of crops
        crop_cal = np.array([1,1])
        # food demand
        demand = np.repeat(100, T_max)
        #initial fund size
        ini_fund = 0
        # prices for selling crops, per crop and cluster
        prices = np.ones([num_crops, k])*1.5
        # total available agricultural area
        maxarea = 400
        # available agricultural area per cluster
        max_areas = maxarea * area_proportions
        # income guaranteed by government per cluster
        guaranteed_income = 150 * area_proportions
        # initial search radius for cobyla
        rhobeg_cobyla = 10
        rhoend_cobyla = 1e-3
        # initial guess for crop allocations if none is given
        if x_ini is None:
            x_ini = np.tile(max_areas.flatten()/(num_crops + 1), \
                                                        T_max*num_crops)
        # define constraints
        constraints = DefineConstraints(k, num_crops, max_areas, T_max)
    else:
        # cultivation costs of crops
        # RICE:
        # Liberia: "The cost of production of swampland Nerica rice (farming 
        # and processing of the paddy) is $308 per metric tons [...]. 
        # Swampland Nerica rice has a yield of 2.8 metric tons per hectare in 
        # Liberia."  
        # https://ekmsliberia.info/document/liberia-invest-agriculture/
        # => 862.4USD/ha (2015)
        # Nigeria: "COST AND RETURNS OF PADDY RICE PRODUCTION IN KADUNA STATE"
        # (Online ISSN: ISSN 2054-6327(online)) 
        # 1002USD/ha 
        # Benin: 
        # 105 FCFA/kg, 3t/ha =>  315000 FCFA/ha => 695.13USD/ha (2011)
        # Burkina-Faso: 
        # Rainfed: 50 FCFA/kg, 1t/ha => 50000 FCFA/ha => 104.62USD/ha (2011)
        # Lowland (bas-fonds): 55 FCFA/kg, 2t/ha => 110000 FCFA/ha 
        #                                               => 230.17 USD/ha
        # (I exclude the value for irrigated areas, as in West Africa 
        # agriculture is mainly rainfed)
        # Mali:
        # 108FCFA/kg, 2.7 t/ha => 291600 FCFA/ha => 589.27 USD/ha
        # Senegal: 
        # 101FCFA/kg, 5 t/ha => 505000 FCFA/ha => 1020.51 USD/ha
        # For Benin, Burkina-Faso, Mali, Senegal:
        # http://www.roppa-afrique.org/IMG/pdf/
        #                       rapport_final_synthese_regionale_riz_finale.pdf
        # in 2011 average exchange rate to USD 477.90 FCFA for 1 USD 
        # in 2014 average exchange rate to USD 494.85 FCFA for 1 USD
        # (https://www.exchangerates.org.uk/
        #                   USD-XOF-spot-exchange-rates-history-2011.html)
        # On average: 
        # (862.4+1002+695.13+104.62+230.17+589.27+1020.51)/7 = 643.44
        # MAIZE
        # "Competiveness of Maize Value Chains for Smallholders in West Africa"
        # DOI: 10.4236/as.2017.812099
        # Benin: 304.6 USD/ha (p. 1384, rainfed)
        # Ivory Coast: 305 USD/ha (p. 1384)
        # Ghana: 301.4 USD/ha (p. 1392) 
        # Nigeria: Field surey 2010 (e-ISSN: 2278-4861)
        # 32079.00 ₦/ha => 213.86 USD/ha
        # (https://www.exchangerates.org.uk/USD-NGN-spot-exchange-rates-
        # history-2010.html)
        # On average: (304.6 + 305 + 301.4 + 213.86)/4 = 281.22 
        costs = np.transpose(np.tile(np.array([643.44, 281.22]), (k, 1)))
       
        # Energy value of crops
        # https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/
        # beltsville-human-nutrition-research-center/methods-and-application-
        # of-food-composition-laboratory/mafcl-site-pages/sr11-sr28/
        # Rice: NDB_No 20450, "RICE,WHITE,MEDIUM-GRAIN,RAW,UNENR" [kcal/100g]
        kcal_rice = 360 * 10000             # [kcal/t]
        # Maize: NDB_No 20014, "CORN GRAIN,YEL" (kcal/100g)
        kcal_maize = 365 * 10000            # [kcal/t]
        crop_cal = np.array([kcal_rice, kcal_maize])
        # Food demand
        # assuming no change of per capita daily consumption (2360kcal in 2015)
        # (https://www.who.int/nutrition/topics/3_foodconsumption/en/)
        # we use UN population scenarios for West Africa and scale them down to
        # the area we use, using the ratio from 2015 (from gridded GPW data)
        with open("IntermediateResults/PreparedData/Population/" + \
                      "UN_PopTotal_Prospects_WesternAfrica.txt", "rb") as fp:    
            total_pop = pickle.load(fp)
            scenarios = pickle.load(fp)
        if pop_scenario == "fixed":
            total_pop_scen = total_pop[np.where(scenarios == "Medium")[0][0],:]
            demand = np.repeat(2360*365*total_pop_scen[(yield_year-1)-1950],\
                       T_max) # for fixed we use pop of 2016 if run starts 2017
            demand = demand * pop_ratio
        else:
            total_pop_scen = total_pop[np.where(scenarios ==  \
                                                    pop_scenario)[0][0],:]
            demand = 2360*365*total_pop_scen[(yield_year-1950): \
                                                    (yield_year-1950+T_max)]
            demand = demand * pop_ratio
        # guaranteed income as share of expected income w/o government
        expected_incomes = np.repeat(expected_incomes, T_max, axis=0)                   
        guaranteed_income = perc_guaranteed * expected_incomes                     
        # guaraneteed income per person assumed to be constant over time, 
        # therefore scale with population size
        if not pop_scenario == "fixed":
            total_pop_ratios = total_pop_scen / \
                                    total_pop_scen[(yield_year-1)-1950]
            guaranteed_income = (guaranteed_income.swapaxes(0,1) * \
                                total_pop_ratios[(yield_year-1950): \
                                                    (yield_year-1950+T_max)]) \
                                .swapaxes(0,1)
        #initial fund size
        ini_fund = 0
        # prices for selling crops, per crop and cluster
        with open("IntermediateResults/PreparedData/Prices/" + \
                                  "CountryAvgFarmGatePrices.txt", "rb") as fp:    
            country_avg_prices = pickle.load(fp)
        # Gambia is not included in our area
        country_avg_prices = country_avg_prices.drop(axis = 0, labels = [4])             
            
        price_maize = np.nansum(country_avg_prices["Maize"].values * \
                                                country_codes["Shares"].values)
        price_rice = np.nansum(country_avg_prices["Rice"].values * \
                              country_codes["Shares excl. Mauritania"].values)
        
        prices = np.transpose(np.tile(np.array([price_rice, \
                                                price_maize]), (k, 1)))
        
        # Agricultural Areas: 
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
        maxarea = total_area * 0.224 
        max_areas = cluster_areas * 0.224        
        # initial search radius for cobyla
        rhobeg_cobyla = 1e6
        rhoend_cobyla = 1e3
        # initial guess for crop allocations if none is given as input 
        if x_ini is None:
            x_ini = np.tile(max_areas.flatten()/(num_crops), \
                                                    T_max*num_crops)/10
        # define constraints (positive areas and not exceeding available 
        # agricultural area per cluster)
        constraints = DefineConstraints(k, num_crops, max_areas, T_max)

    # everything except yields if wo_yields == True
    if wo_yields:
        prob_cat_year = RiskForCatastrophe(risk, k, num_cl_cat)
        cat_clusters, terminal_years = CatastrophicYears(risk, N_c, T_max, \
                                                         k, num_cl_cat)
        
        args = {"k": k,
                "num_crops": num_crops,
                "N_c": N_c,
                "cat_clusters": cat_clusters,
                "terminal_years": terminal_years,
                "costs": costs,
                "demand": demand,
                "ini_fund": ini_fund,
                "tax": tax,
                "prices": prices,
                "T_max": T_max,
                "guaranteed_income": guaranteed_income,
                "crop_cal": crop_cal,
                "max_areas": max_areas}
        # information for the solver
        meta = {"rhobeg_cobyla": rhobeg_cobyla,
                "rhoend_cobyla": rhoend_cobyla,
                "seed": seed}
        # information not needed by the solver but potentially interesting 
        other = {"slopes": slopes,
                 "constants": constants,
                 "residual_stds": residual_stds,
                 "prob_cat_year": prob_cat_year}
        
        return(x_ini, constraints, args, meta, other)
    
    if returnyieldmean == True:
        prob_cat_year = RiskForCatastrophe(risk, k, num_cl_cat)
        yld_means = \
              YieldRealisations(slopes, constants, residual_stds, yield_year, 
                               N_c, risk, T_max, k, 
                               num_cl_cat, num_crops, yield_projection, \
                               returnyieldmean)
        return(yld_means)

    # get yield realizations:
    # what is the probability of a catastrophic year for given settings?
    prob_cat_year = RiskForCatastrophe(risk, k, num_cl_cat)
    print("Risk level catastrophic year: " + str(prob_cat_year))
    np.random.seed(seed)
    # create realizations of presence of catastrophic yields and corresponding
    # yields
    cat_clusters, terminal_years, ylds = \
              YieldRealisations(slopes, constants, residual_stds, yield_year, 
                               N_c, risk, T_max, k, 
                               num_cl_cat, num_crops, yield_projection)
    
    # group output into different dictionaries
    # arguments that are given to the objective function by the solver
    args = {"k": k,
            "num_crops": num_crops,
            "N_c": N_c,
            "cat_clusters": cat_clusters,
            "terminal_years": terminal_years,
            "ylds": ylds,
            "costs": costs,
            "demand": demand,
            "ini_fund": ini_fund,
            "tax": tax,
            "prices": prices,
            "T_max": T_max,
            "guaranteed_income": guaranteed_income,
            "crop_cal": crop_cal,
            "max_areas": max_areas}
    # information for the solver
    meta = {"rhobeg_cobyla": rhobeg_cobyla,
            "rhoend_cobyla": rhoend_cobyla,
            "seed": seed}
    # information not needed by the solver but potentially interesting 
    other = {"slopes": slopes,
             "constants": constants,
             "residual_stds": residual_stds,
             "prob_cat_year": prob_cat_year}
    
    return(x_ini, constraints, args, meta, other)
    

# we chose a probability for a catastrophic yields (1:risk). But if we 
# have multiple cluster the probability for a catastrophic year is different, 
# depending on the number of clusters and the number of clusters that have to
# be catastrophic for the year to be general
def RiskForCatastrophe(risk, num_clusters, num_cl_cat):
# risk: risk level for a catastrophic yields (1 : risk)
# num_clusters: number of clusters
# num_cl_cat: number of clusters that have to have catastrophic yields for a 
#             catastrophic year

    alpha = 1/risk
    res = 0
    for i in range(num_cl_cat, num_clusters + 1):
        res = res + comb(num_clusters, i) * (alpha**i) * \
                    (1-alpha)**(num_clusters-i)
                    
    print("Prob for catastrophic year: " + str(res), flush = True)    
    return(1/res)

# given risk level and parameters for yield distributions, generates the 
# catastrophes and corresponding yields
def YieldRealisations(yld_slopes, yld_constants, resid_std, year, N_c, \
                      risk, T_max, num_clusters, num_cl_cat, num_crops, \
                      yield_projection, returnyieldmean = False):  
# yld_slopes, yld_constants: slope and constant of linear trend in yield 
#                            averages, per crop & cluster  
# resid_std: standard deviation of residuals of yield trend per crop & cluster
# year: either year to use for yield generations for all years, or year to 
#       start with if using the trend
# N_c: number of realisations to generate
# risk: risk level for a catastrophic yields (1 : risk)
# onein_singlecluster: risk for catastrophe in a single cluster that 
#                      corresponds to overall risk onein_catyear   
# T_max: maximum number of years covered by the model. If no catastrophe 
#        happens before, the model ends after year T_max
# num_clutsers: number of clusters used
# num_cl_cat: number of clusters that have to have catastrophic yields for a 
#             catastrophic year
# num_crops: number of crops used
# yield_projection: type of yield generation. Either "fixed" if the same 
#                   distribution should be use for every year, or "trend" if
#                   the mean of the yield distribution should follow the linear
#                   trend
# returnyieldmean: as needed for VSS, if True only the yield means are returned
    
    # generating catastrophes 
    cat_clusters, terminal_years = CatastrophicYears(risk, \
                                    N_c, T_max, num_clusters, num_cl_cat)
    
    # generating yields according to catastrophes
    ylds = ProjectYields(yld_slopes, yld_constants, resid_std, year, N_c, \
                         cat_clusters, terminal_years, T_max, risk, \
                         num_clusters, num_crops, yield_projection, \
                         returnyieldmean)
    if returnyieldmean:
        return(ylds)
        
    ylds[ylds<0] = 0
    
    return(cat_clusters, terminal_years, ylds)

# generating the occurence of catastrophes
def CatastrophicYears(risk, N_c, T_max, num_clusters, \
                      num_cl_cat):
# for parameters see YieldRealisations
    
    # generating uniform random variables between 0 and 1 to define years
    # with catastrophic cluster (catastrohic if over threshold)
    threshold = 1 - (1/risk)
    cat_clusters = np.random.uniform(0, 1, [N_c, T_max, num_clusters])
    cat_clusters[cat_clusters > threshold] = 1
    cat_clusters[cat_clusters <= threshold] = 0
    
    # year is catastrophic if enough clusters are catastrophic
    cat_years = np.sum(cat_clusters, axis = 2)
    cat_years[cat_years < num_cl_cat] = 0
    cat_years[cat_years >= num_cl_cat] = 1
    
    # terminal year is first catastrophic year. If no catastrophic year before
    # T_max, we set terminal year to -1
    terminal_years = np.sum(cat_years, axis = 1)
    for i in range(0, N_c):
        if terminal_years[i] == 0:
            terminal_years[i] = -1
        else:
            terminal_years[i] = np.min(np.where(cat_years[i, :] == 1))
    return(cat_clusters, terminal_years)

# generate yield realizations according to the occurence of catastrophes
def ProjectYields(yld_slopes, yld_constants, resid_std, start_year, \
                  N_c, cat_clusters, terminal_years, T_max, onein_catyear, \
                  num_clusters, num_crops, yield_projection, \
                  returnyieldmean = False):
# cat_clusters: matrix of fromat [N_c, T_max, num_clusters] indicating whether 
#               a cluster in a certain year for a certain realization has 
#               catastrophic yields (1) or not (0)    
# terminal_years: array of length N_c giving the termination year for each
#                 realization (i.e. year of first catastrophe) with 0 for 
#                 catastrophe in first year, 1 for second year, etc. If no
#                 catastrophe happens before T_max, this is indicated by -1
# for other parameters see YieldRealisations
    
    # project means of yield distributions (either repeating fixed year or by
    # using trend)
    if yield_projection == "fixed": # for fixed we use the year before start
        year_rel = (start_year - 1) - 1981
        yld_means = yld_constants + year_rel * yld_slopes
        yld_means = np.repeat(yld_means[np.newaxis, :, :], T_max, axis = 0)
            
    elif yield_projection == "trend":
        year_rel = start_year - 1981
        years = np.transpose(np.tile(np.array(range(year_rel, year_rel + \
                                        T_max)), (num_clusters, num_crops, 1)))
        yld_means = yld_constants + years * yld_slopes
    
    # needed for VSS
    if returnyieldmean == True:
        return(yld_means)
    
    # initializing yield array
    ylds = np.empty([N_c, T_max, num_crops, num_clusters])
    ylds.fill(np.nan)
    
    # calculating quantile of standard normal distribution corresponding to 
    # the catastrophic yield quantile for truncnorm fct
    quantile_low = stats.norm.ppf(1/(onein_catyear), 0, 1)
    
    # generating yields: for catastrophic clusters from lower quantile, for
    # normal years from upper quantile. Dependence between crops, i.e. if 
    # cluster is catastrophic, both crops get yields from lower quantile of 
    # their distributions
    for run in range(0, N_c):
        if int(terminal_years[run]) == -1:
            term_year = T_max - 1   # as first year is year 0
        else:
            term_year = int(terminal_years[run])
        for t in range(0, term_year + 1):
            for cl in range(0, num_clusters):
                if cat_clusters[run, t, cl] == 1:
                    ylds[run, t, :, cl] = stats.truncnorm.rvs(- np.inf, \
                           quantile_low, yld_means[t, :, cl], resid_std[:, cl])
                if cat_clusters[run, t, cl] == 0:
                    ylds[run, t, :, cl] = stats.truncnorm.rvs(quantile_low, \
                                np.inf, yld_means[t, :, cl], resid_std[:, cl])
    
    return(ylds)  

# objective function: costs to minimize for robust solution
def ObjectiveFunctionMultipleYears(x, num_clusters, num_crops, N_c, \
                    cat_clusters, terminal_years, Y, costs, A, \
                    ini_fund, tax, prices, T_max, guaranteed_income, crop_cal, 
                    rhoF, rhoS, meta = False, meta_payouts = False): 
# x:                array of length T_max*num_crops*num_clusters
#                   gives allocation of area to each crop in each cluster
# num_clutsers:     number of clusters used
# num_crops:        number of crops used
# N_c:              number of yield realisations
# cat_clusters:     matrix of fromat [N_c, T_max, num_clusters] 
#                   indicating whether a cluster in a certain year for a 
#                   certain realization has catastrophic yields (1) or not (0)   
# terminal_years:   array of length N_c 
#                   give the termination year for each realization (i.e. year 
#                   of first catastrophe) with 0 for catastrophe in first year,
#                   1 for second year, etc. If no catastrophe happens before 
#                   T_max, this is indicated by -1
# Y:                matrix [N_c, T_max, num_crops, num_clusters], [t/ha]
#                   yield realisations per year, cluster & crop
# costs:            num_crops x num_clusters, [$/ha]
#                   gives cluster and crop specific costs of cultivation per ha
# rhoF:             food demand penalty. Has to be paied each year per kcal 
#                   of demand that is not covered by the production
# rhoS:             solvency penalty. Has to be paied adter final year per 
#                   negative dollar of final fund after payouts
# A:                total food demand (for all clusters) given in kcal
# ini_fund:         initial fund size
# tax:              tax rate farmers have to pay on profits if their cluster
#                   does not have catastrophic yields
# prices:           array of length num_crops
#                   farm gate price earned by farmers per tonne of crop
# T_max:            maximum number of years covered by the model. If no  
#                   catastrophe happens before, the model ends after year T_max
# guaranteed_income: araay of length num_clusters
#                   maximum amount of money in USD government pay for each 
#                   cluster in case of catastrophic yields
# crop_cal:         array of length num_crops, 
#                   energetic value of crops in kcal/t 
# meta:             if False only the expected costs are returned, if True some
#                   other information are returned as well

    if not meta:
        ObjectiveFunctionMultipleYears.Calls += 1
    
    # preparing x for all realizations
    x = np.reshape(x,[T_max, num_crops, num_clusters])
    X = np.repeat(x[np.newaxis, :, :, :], N_c, axis=0)
    for c in range(0, N_c):
        t_c = int(terminal_years[c])                   # catastrophic year
        if t_c > -1:
            X[c, (t_c + 1) : , :, :] = np.nan  # no crop area after catastrophe
            
    # Production
    prod = X * Y                             # nan for years after catastrophe
    kcal  = np.swapaxes(prod, 2, 3)
    kcal = kcal * crop_cal                   # relevant for realistic values
    
    # Shortcomings
    S = A - np.sum(kcal, axis = (2, 3))      # nan for years after catastrophe
    S[S < 0] = 0
    
    # fixed costs for cultivation of crops
    fixed_costs =  X * costs
    
    # Yearly profits
    P =  prod*prices - fixed_costs          # still per crop and cluster, 
                                            # nan for years after catastrophe
    P = np.sum(P, axis = 2)                 # now per cluster
    if meta:    # calculate expected income
        exp_income = np.nanmean(P, axis = 0) # per year and cluster
    P[P < 0] = 0
 
    # Payouts
    payouts = guaranteed_income - P  # as we set negative profit to zero,
                                     # government doesn't cover those
                                     # it only covers up to guaranteed income.
    payouts[(cat_clusters == 0) + (payouts < 0)] = 0
                # no payout if there is no catastrophe, even if guaranteed 
                # income is not reached
                # in the unlikely case where a cluster makes more than the
                # guaranteed income despite catastrophe, no negative payouts!
                      
    # Final fund
    ff = ini_fund + tax * np.nansum(P, axis = (1,2)) - \
                                            np.nansum(payouts, axis = (1,2))
    ff[ff > 0] = 0
    
    # expected total costs
    exp_tot_costs = np.mean(np.nansum(fixed_costs, axis = (1,2,3)) + \
                            rhoF * np.nansum(S, axis = 1) + rhoS * (- ff))
    
    if meta_payouts:
        fund = ini_fund + tax * np.nancumsum(np.nansum(P, axis = 2), axis = 1)
        return(payouts, fund)

    if meta:
        return(exp_tot_costs, 
               np.nansum(fixed_costs, axis = (1,2,3)), #  fixcosts (N_c)
               S, # shortcomings per realization and year
               exp_income, # expected income (T_max, k)
               P, # profits
               np.nanmean(S, axis = 0) , # yearly avg shortcoming (T_max)
               rhoF * S, # yearly food demand penalty (N_c x T_max)
               np.nanmean(rhoF * S, axis = 0), # yearly avg fd penalty (T_max)
               rhoS * (- ff), # solvency penalty (N_c)
               ini_fund + tax * np.nansum(P, axis = (1,2)) - \
                 np.nansum(payouts, axis = (1,2)), # final fund per realization
               payouts, # government payouts (N_c, T_max, k)
               np.nansum(fixed_costs, axis = (2,3)), #  fixcosts (N_c, T_max)
               )

    return(exp_tot_costs)    

# to get metainformation for final crop allocation after running model
def GetMetaMultipleYears(crop_alloc, args, rhoF, rhoS):
# crop_alloc: crop areas for all years, crops, and clusters
# args: arguments for the objective function (dictionary as given by 
#       SetParameters())
# rhoF: penalty for not reaching food demand
# rhoS: penalty for government not staying solvent after catastrophe
    
    # running the objective function with option meta = True to get 
    # intermediate results of the calculation
    exp_tot_costs, fix_costs, S, exp_incomes, P, exp_shortcomings, \
    fd_penalty, avg_fd_penalty, sol_penalty, final_fund, payouts, \
    yearly_fixed_costs = ObjectiveFunctionMultipleYears(crop_alloc, 
                                           args["k"], 
                                           args["num_crops"],
                                           args["N_c"], 
                                           args["cat_clusters"], 
                                           args["terminal_years"],
                                           args["ylds"], 
                                           args["costs"], 
                                           args["demand"],
                                           args["ini_fund"],
                                           args["tax"],
                                           args["prices"],
                                           args["T_max"],
                                           args["guaranteed_income"],
                                           args["crop_cal"], 
                                           rhoF, 
                                           rhoS,
                                           meta = True)
    
    # calculationg additional quantities
    prob_staying_solvent = np.sum(final_fund >= 0) /  args["N_c"]
    tmp = np.copy(S)
    tmp[tmp > 0] = 1
    prob_food_security = 1 - np.nanmean(tmp)
        
    # group information into a dictionary
    meta_sol = {"exp_tot_costs": exp_tot_costs,
                "fix_costs": fix_costs,
                "S": S,
                "exp_incomes": exp_incomes,
                "profits": P,
                "exp_shortcomings": exp_shortcomings,
                "fd_penalty": fd_penalty,
                "avg_fd_penalty": avg_fd_penalty,
                "sol_penalty": sol_penalty,
                "final_fund": final_fund,
                "prob_staying_solvent": prob_staying_solvent,
                "prob_food_security": prob_food_security,
                "payouts": payouts,
                "yearly_fixed_costs": yearly_fixed_costs}
    
    # reshae crop_alloc from a single vector to a matrix with time, crops and 
    # clusters as dimensions
    crop_alloc = np.reshape(crop_alloc, \
                                [args["T_max"], args["num_crops"], args["k"]])
    
    return(crop_alloc, meta_sol)

# defining the constraints on the crop areas
def DefineConstraints(k, num_crops, max_areas, T_max):
# k: number of clusters
# num_crops: number of crops
# max_areas: max. available agricultural area per cluster
# T_max: maximum number of years covered by the model
    
    # total used area in one cluster (for all crops) can't exceed the total 
    # area that is available for agriculture
    def const_cobyla1(x):
        x = np.reshape(x,[T_max, num_crops, k])
        x = np.nansum(x, axis = 1)
        res = max_areas - x
        res = res.flatten()
        return(res) 
        
    # crop areas can't be negative    
    def const_cobyla2(x):
        return(x)    
        
    constraints = [const_cobyla1, const_cobyla2]
    return(constraints)

# finding robust solution for crop allocation
def OptimizeMultipleYears(x_ini, constraints, args, meta, rhoF, rhoS):
    
# x_ini: initial guess for crop allocations
# constraints: constraint functions for crop allocation - no negative crop
#              areas and total area within a cluster below available agri-
#              cultural area in that cluster
# args: other parameters that are directly passed to objective function, 
#       given as dictionary by SetParameters()
# meta includes
#     rhobeg_cobyla: initial size of trust region for COBYLA solver
#     rhoend_cobyla: final size of trust region for COBYLA solver
#     seed: seed which is set before calling the solver for reproducebility
# rhoF: penalty for not reaching food demand
# rhoS: penalty for government not staying solvent after catastrophe
    
    
    t_b = tm.time()
    
    np.random.seed(250620)
    
    ObjectiveFunctionMultipleYears.Calls = 0
    
    # calling optimizer
    crop_alloc = opt.fmin_cobyla(func = ObjectiveFunctionMultipleYears, \
                                 x0 = x_ini, \
                                 cons = constraints, \
                                 args = (args["k"], 
                                         args["num_crops"],
                                         args["N_c"], 
                                         args["cat_clusters"], 
                                         args["terminal_years"],
                                         args["ylds"], 
                                         args["costs"], 
                                         args["demand"],
                                         args["ini_fund"],
                                         args["tax"],
                                         args["prices"],
                                         args["T_max"],
                                         args["guaranteed_income"],
                                         args["crop_cal"], 
                                         rhoF, 
                                         rhoS),
                                 consargs = (), \
                                 rhobeg = meta["rhobeg_cobyla"], \
                                 rhoend = meta["rhoend_cobyla"], \
                                 maxfun = 70000,
                                 disp = 3)
    
    # get meta information for final crop allocation
    crop_alloc, meta_sol = GetMetaMultipleYears(crop_alloc, args, rhoF, rhoS)
    
    t_e = tm.time()
    duration = t_e - t_b
    
    return(crop_alloc, meta_sol, duration) 
    
# finds correct penalty rhoF for a probability probF (while rhoS = 0)
def GetRhoF(settings, probF, rhoFini, x_ini = None, \
            meta = False, accuracy = 2):
# settings: dictionary of all settings given by DefaultSettingsExcept()
#           settings are eplained in StochasticOptimization.py
# probF: demanded probability of keeping the food demand constraint
# rhoFini: initial guess for the penalty which will give the correct 
#          probability for food security
# x_ini: initial guess of crop allocations for the solver
# meta: should meta information be returned?
# accuracy: to how many decimal diggits should the probability be correct?
#           accuracy is 2 for all runs except for calculating the expected 
#           income where we use 3     

    # setting parameters for the given settings and x_ini if given
    x_ini, const, args, meta_cobyla, other = \
                                    SetParameters(settings, x_ini)           
          
    # initializing meta informaiton                         
    if meta:
        probs = []
        rhoFs = []
        crop_allocs = []
        meta_sols = []
        
    # initialize values for search algorithm
    rhoFlastdown = 0
    rhoFlastup = 0
    
    # calculate initial guess
    crop_alloc, meta_sol, duration = \
            OptimizeMultipleYears(x_ini, const, args, meta_cobyla, rhoFini, 0)
    rhoFold = rhoFini
    print("rhoF: " + str(rhoFold) + \
          ", prob: " + str(meta_sol["prob_food_security"]) + \
          ", time: " + str(np.round(duration, 2)), flush=True)
    
    # save meta information
    if meta:
        probs.append(meta_sol["prob_food_security"])
        total_time = duration
        rhoFs.append(rhoFold)
        meta_sols.append(meta_sol)
        crop_allocs.append(crop_alloc)
            
    # we increase (or decrease) rhoF until we first pass the desired 
    # probability. Then we reduce the search intervall by half in each step
    # until the demanded accuracy is reached
    while np.round(meta_sol["prob_food_security"], accuracy) != probF:
        # find next guess
        if meta_sol["prob_food_security"] < probF:
            rhoFlastup = rhoFold
            if rhoFlastdown == 0:
                rhoFnew = rhoFold * 4
            else:
                rhoFnew = rhoFold + (rhoFlastdown - rhoFold)/2 
        else:
            rhoFlastdown = rhoFold
            if rhoFlastup == 0:
                rhoFnew = rhoFold / 4
            else:
                rhoFnew = rhoFold - (rhoFold - rhoFlastup) /2
        
        # solve model for guess
        crop_alloc, meta_sol, duration = OptimizeMultipleYears(x_ini, const, \
                                        args, meta_cobyla, rhoFnew, 0)
        
        # save meta information
        if meta:
            probs.append(meta_sol["prob_food_security"])
            total_time = total_time + duration
            crop_allocs.append(crop_alloc)
            rhoFs.append(rhoFold)
            meta_sols.append(meta_sol)
        
        rhoFold = rhoFnew
        print("rhoF: " + str(rhoFold) + \
              ", prob: " + str(meta_sol["prob_food_security"]) + \
              ", time: " + str(np.round(duration, 2)), flush=True)
    if meta:    
        return(rhoFs, crop_allocs, meta_sols, probs, total_time)
    else:
        return(rhoFold, crop_alloc, meta_sol)
            
# finds correct penalty rhoS for a probability probS (while rhoF = 0)
def GetRhoS(settings, probS, rhoSini, x_ini = None, meta = False):
# settings: dictionary of all settings given by DefaultSettingsExcept()
#           settings are eplained in StochasticOptimization.py
# probS: demanded probability of keeping the solvency constraint
# rhoSini: initial guess for the penalty which will give the correct 
#          probability for solvency
# x_ini: initial guess of crop allocations for the solver
# meta: should meta information be returned?

    # setting parameters for the given settings and x_ini if given
    x_ini, const, args, meta_cobyla, other = \
                                    SetParameters(settings, x_ini)      
          
    # initializing meta informaiton             
    if meta:
        probs = []
        rhoSs = []
        crop_allocs = []
        meta_sols = []
        
    # initialize values for search algorithm
    rhoSlastdown = 0
    rhoSlastup = 0
    
    # calculate initial guess
    crop_alloc, meta_sol, duration = \
         OptimizeMultipleYears(x_ini, const, args, meta_cobyla, 0, rhoSini)
    rhoSold = rhoSini
    print("rhoS: " + str(rhoSold) + \
          ", prob: " + str(meta_sol["prob_staying_solvent"]) + \
          ", time: " + str(np.round(duration, 2)), flush=True)
    
    # save meta information
    if meta:
        probs.append(meta_sol["prob_staying_solvent"])
        rhoSs.append(rhoSold)
        crop_allocs.append(crop_alloc)
        meta_sols.append(meta_sol)
        total_time = duration
    
    # we increase (or decrease) rhoS until we first pass the desired 
    # probability. Then we reduce the search intervall by half in each step
    # until the demanded accuracy is reached
    while np.round(meta_sol["prob_staying_solvent"], 2) != probS:
        # find next guess
        if meta_sol["prob_staying_solvent"] < probS:
            rhoSlastup = rhoSold
            if rhoSlastdown == 0:
                rhoSnew = rhoSold * 4
            else:
                rhoSnew = rhoSold + (rhoSlastdown - rhoSold)/2 
        else:
            rhoSlastdown = rhoSold
            if rhoSlastup == 0:
                rhoSnew = rhoSold / 4
            else:
                rhoSnew = rhoSold - (rhoSold - rhoSlastup) /2
            
        # solve model for guess
        crop_alloc, meta_sol, duration = OptimizeMultipleYears(x_ini, const, \
                                        args, meta_cobyla, 0, rhoSnew)
         
        # save meta information
        if meta:
            probs.append(meta_sol["prob_staying_solvent"])
            total_time = total_time + duration
            crop_allocs.append(crop_alloc)
            meta_sols.append(meta_sol)
            rhoSs.append(rhoSold)
    
        rhoSold = rhoSnew
        print("rhoS: " + str(rhoSold) + \
              ", prob: " + str(meta_sol["prob_staying_solvent"]) + \
              ", time: " + str(np.round(duration, 2)), flush=True)
    if meta:
        return(rhoSs, crop_allocs, meta_sols, probs, total_time)           
    else:
        return(rhoSold, crop_alloc, meta_sol)
            
# as penalties are calculated while the respective other penalty is set to zero
# they don't have to be recalculated for each different combination of 
# penalties. This function checks whether penalties have already been 
# calculated. If so, they are loaded, else calculated and saved
def GetPenalties(settings, probF, probS, rhoFini = 1e-3, rhoSini = 100):
# settings: dictionary of all settings given by DefaultSettingsExcept()
#           settings are eplained in StochasticOptimization.py
# probF: demanded probability of keeping the food demand constraint
# probS: demanded probability of keeping the solvency constraint
# rhoFini: initial guess for the penalty which will give the correct 
#          probability for reaching food demand
# rhoSini: initial guess for the penalty which will give the correct 
#          probability for solvency
    
    # all settings that affect the calculation of rhoF
    if settings["k"] == 1:
        SettingsAffectingRhoF = "k" + str(settings["k"]) + \
                     "num_cl_cat" + str(settings["num_cl_cat"]) + \
                     "num_crops" + str(settings["num_crops"]) + \
                     "yield_projection" + str(settings["yield_projection"]) + \
                     "yield_year" + str(settings["yield_year"]) + \
                     "stilised" + str(settings["stilised"]) + \
                     "pop_scenario" + str(settings["pop_scenario"]) +  \
                     "risk" + str(settings["risk"]) + \
                     "N_c" + str(settings["N_c"]) + \
                     "T_max" + str(settings["T_max"]) 
    else:
        SettingsAffectingRhoF = "k" + str(settings["k"]) + \
                     "num_crops" + str(settings["num_crops"]) + \
                     "yield_projection" + str(settings["yield_projection"]) + \
                     "yield_year" + str(settings["yield_year"]) + \
                     "stilised" + str(settings["stilised"]) + \
                     "pop_scenario" + str(settings["pop_scenario"]) +  \
                     "N_c" + str(settings["N_c"]) + \
                     "T_max" + str(settings["T_max"]) 
                          
    # all settings that affect the calculation of rhoS
    SettingsAffectingRhoS = "k" + str(settings["k"]) + \
                     "num_cl_cat" + str(settings["num_cl_cat"]) + \
                     "num_crops" + str(settings["num_crops"]) + \
                     "yield_projection" + str(settings["yield_projection"]) + \
                     "yield_year" + str(settings["yield_year"]) + \
                     "stilised" + str(settings["stilised"]) + \
                     "pop_scenario" + str(settings["pop_scenario"]) +  \
                     "risk" + str(settings["risk"]) + \
                     "N_c" + str(settings["N_c"]) + \
                     "T_max" + str(settings["T_max"]) + \
                     "tax" + str(settings["tax"]) + \
                     "perc_guaranteed" + str(settings["perc_guaranteed"])
                     
    # get lists of settings for which at least one penalty has already been
    # calculated
    with open("StoOptMultipleYears/ProbToPenalty/ListRhoF.txt", "rb") as fp:    
        list_rhoFs = pickle.load(fp)
    with open("StoOptMultipleYears/ProbToPenalty/ListRhoS.txt", "rb") as fp:    
        list_rhoSs = pickle.load(fp)
        
    if probF == 0:
        print("Setting rhoF to zero")
        rhoF = 0
    else:   
        # check is rhoF for this setting has been calculated
        if SettingsAffectingRhoF in list_rhoFs:
            # if so open calculated penalties
            with open("StoOptMultipleYears/ProbToPenalty/rhoFs/" + \
                                  SettingsAffectingRhoF + ".txt", "rb") as fp:    
                rhoFs = pickle.load(fp)  
            # if the right one has been calculated...
            if probF in rhoFs.keys():
                print("Fetching rhoF", flush=True)
                # ... load it
                rhoF = rhoFs[probF]
            # if not...
            else:
                print("Calculating rohF (new version)", flush=True)
                # ... calculate it
                rhoF, crop_alloc, meta_sol = GetRhoF(settings, probF, rhoFini)
                # add it to dictionary of calculated rhoFs
                rhoFs[probF] = rhoF
            # save calculated rhoFs
            with open("StoOptMultipleYears/ProbToPenalty/rhoFs/" + \
                                  SettingsAffectingRhoF + ".txt", "wb") as fp:    
                pickle.dump(rhoFs, fp)
        
        # if no rhoFs for this settings have been calculated...       
        else:
            print("Calculating rohF (new version)", flush=True)
            # .. calculated rhoF
            rhoF, crop_alloc, meta_sol = GetRhoF(settings, probF, rhoFini)
            # initialize dictionary of rhoFs
            rhoFs = {probF: rhoF}
            # save rhoFs
            with open("StoOptMultipleYears/ProbToPenalty/rhoFs/" + \
                                  SettingsAffectingRhoF + ".txt", "wb") as fp:    
                pickle.dump(rhoFs, fp)
            # add settings to list of settings for which rhoFs have been 
            # calculated (reload list in case other server changed something
            # in the meantime!)
            with open("StoOptMultipleYears/ProbToPenalty/" + \
                                                  "ListRhoF.txt","rb") as fp:    
                list_rhoFs = pickle.load(fp)
            list_rhoFs.append(SettingsAffectingRhoF)
            with open("StoOptMultipleYears/ProbToPenalty/" + \
                                                  "ListRhoF.txt","wb") as fp:    
                pickle.dump(list_rhoFs, fp)   
                
    if probS == 0:
        print("Setting rhoS to zero")
        rhoS = 0
    else:               
        # Analogous for rhoS:       
        if SettingsAffectingRhoS in list_rhoSs:
            with open("StoOptMultipleYears/ProbToPenalty/rhoSs/" + \
                                  SettingsAffectingRhoS + ".txt", "rb") as fp:    
                rhoSs = pickle.load(fp)  
            if probS in rhoSs.keys():
                print("Fetching rhoS", flush=True)
                rhoS = rhoSs[probS]
            else:
                print("Calculating rhoS (new version)", flush=True)
                rhoS, crop_alloc, meta_sol = GetRhoS(settings, probS, rhoSini)
                rhoSs[probS] = rhoS
            with open("StoOptMultipleYears/ProbToPenalty/rhoSs/" + \
                                  SettingsAffectingRhoS + ".txt", "wb") as fp:    
                pickle.dump(rhoSs, fp)
                
        else:
            print("Calculating rhoS (new version)", flush=True)
            rhoS, crop_alloc, meta_sol = GetRhoS(settings, probS, rhoSini)
            rhoSs = {probS: rhoS}
            with open("StoOptMultipleYears/ProbToPenalty/rhoSs/" + \
                                  SettingsAffectingRhoS + ".txt", "wb") as fp:    
                pickle.dump(rhoSs, fp)
                
            with open("StoOptMultipleYears/ProbToPenalty/" + \
                                                  "ListRhoS.txt","rb") as fp:    
                list_rhoSs = pickle.load(fp)
            list_rhoSs.append(SettingsAffectingRhoS)
            with open("StoOptMultipleYears/ProbToPenalty/" + \
                                                  "ListRhoS.txt","wb") as fp:    
                pickle.dump(list_rhoSs, fp)   
                
    return(rhoF, rhoS)
    
    
# function wrapping all of the above: if the model is to be run from scratch
# (i.e. not for specific penalties but for probabilities), this is the only
# function that needs to be called. 
# If specific penalties are to be used, the following combination of functions 
# has to be run:
#       settings = DefaultSettingsExcept(**kwargs)
#       x_ini, const, args, meta_cobyla, other = SetParameters(settings)  
#       crop_alloc, meta_sol, duration = \
#           OptimizeMultipleYears(x_ini, const, args, meta_cobyla, rhoF, rhoS)
def OptimizeFoodSecurityProblem(probF, probS, rhoFini = 1e-3, rhoSini = 100, \
                                                                    **kwargs):
# settings: dictionary of all settings given by DefaultSettingsExcept()
#           settings are eplained in StochasticOptimization.py
# probF: demanded probability of keeping the food demand constraint
# probS: demanded probability of keeping the solvency constraint
# rhoFini: initial guess for the penalty which will give the correct 
#          probability for reaching food demand
# rhoSini: initial guess for the penalty which will give the correct 
#          probability for solvency
# **kwargs: keyword arguments passed to DefaultSettingsExcept. Should be all 
#           settings that are not supposed to be on their default value 

    # create dictionary of all settings (includes calculating or loading the
    # correct expected income)
    settings = DefaultSettingsExcept(**kwargs)
    # get the penalties corresponding the given probabilities
    rhoF, rhoS = GetPenalties(settings, probF, probS, rhoFini, rhoSini)
    print("Getting parameters", flush = True)
    # get parameters for the given settings
    x_ini, const, args, meta_cobyla, other = SetParameters(settings)   
    print("Running Model", flush = True)
    # run the optimizer
    crop_alloc, meta_sol, duration = \
            OptimizeMultipleYears(x_ini, const, args, meta_cobyla, rhoF, rhoS)
    print("Time: " + str(np.round(duration, 2)), flush=True)
    return(crop_alloc, meta_sol, rhoF, rhoS, settings, args)            
                  
    
    
# %% #################### VALUE OF THE STOCHASTIC SOLUTION ####################            
            
# function to calculate a scenario based solution for caomparison to quantify
# the performance of the stochastic optimization model
def DetSolution(**kwargs):
    # get settings    
    settings = DefaultSettingsExcept(**kwargs)
    yield_means = SetParameters(settings, returnyieldmean=True)
    x_ini, constraints, args, meta, other = SetParameters(settings, \
                                                          wo_yields = True)
    
    # for each year, sort crops according to performance in kcal/$ and fill 
    # cluster areas up until demand is reached
    x = np.zeros([args["T_max"], args["num_crops"], args["k"]])
    for t in range(0, settings["T_max"]):
        yld_tmp = yield_means[t,:,:]
        # get yields in kcal per ha
        yields_kcal = (yld_tmp.transpose()*args["crop_cal"]).transpose()
        # calculate production per dollar for each crop in each cluster
        kcal_per_dollar =  yields_kcal/args["costs"]
        # find order of clutsers according to performacnce of the respective 
        # best crop
        cluster_order = np.flip(np.argsort(np.max(kcal_per_dollar, axis = 0)))
        # variable for termination of algorithm
        not_done = True
        i = 0
        # food demand not yet covered
        remaining_demand = args["demand"][t]
        while not_done == True:
            # if all clusters are used but demand not yet met this approach 
            # does not work
            if i >= args["k"]:
                print("Geht nicht")
                not_done = False
                continue
            # else get next best cluster
            cl = cluster_order[i]
            # with the correct crop
            cr = np.where(kcal_per_dollar[:,cl] ==  \
                          np.max(kcal_per_dollar[:,cl]))[0][0]
            # calculate possible production using all area of that cluster 
            # for this crop
            possible_production = args["max_areas"][cl]*yields_kcal[cr, cl]
            # if this is more than still needed, use only the area needed 
            # and set not_done = False to terminate the algorithm
            if possible_production >= remaining_demand:
                x[t, cr, cl] = remaining_demand/yields_kcal[cr, cl]
                not_done = False
            # else use whole area and calculate the still remaining demand
            else:
                x[t, cr, cl] = args["max_areas"][cl]
                remaining_demand = remaining_demand - possible_production
                i = i+1
    
    return(x, args)
    
    
# function to get expected total costs using a specific crop allocation (used
# for VSS but does not return VSS directly)    
def VSS(crop_alloc, probF, probS, **kwargs):
    # create dictionary of all settings (includes calculating or loading the
    # correct expected income)
    settings = DefaultSettingsExcept(**kwargs)
    # get the penalties corresponding the given probabilities
    rhoF, rhoS = GetPenalties(settings, probF, probS)
    # get parameters for the given settings
    x_ini, const, args, meta_cobyla, other = SetParameters(settings)   
            
    # run stochastic objective function for resulting crop allocation
    print("Calculate expected costs for EV crop allocation")
    crop_alloc, meta_sol = GetMetaMultipleYears(crop_alloc.flatten(), args, \
                                                rhoF, rhoS)
    
    return(meta_sol, settings, rhoF, rhoS)
    