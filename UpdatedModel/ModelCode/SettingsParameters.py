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
import pandas as pd
import sys

from ModelCode.Auxiliary import printing

# %% ############# FUNCTIONS TO GET INPUT FOR FOOD SECURITY MODEL #############

def DefaultSettingsExcept(PenMet = "prob",
                          probF = 0.99,
                          probS = 0.95, 
                          rhoF = None,
                          rhoS = None,
                          k = 9,     
                          k_using = [3],
                          num_crops = 2,
                          yield_projection = "fixed",   
                          sim_start = 2017,
                          pop_scenario = "fixed",
                          risk = 0.05,                          
                          N = 10000, 
                          validation_size = None,
                          T = 20,
                          seed = 201120,
                          tax = 0.01,
                          perc_guaranteed = 0.9,
                          ini_fund = 0):     
    """
    Using the default for all settings not specified, this creates a 
    dictionary of all settings.

    Parameters
    ----------
    PenMet : "prob" or "penalties", optional
        "prob" if desired probabilities are given and penalties are to be 
        calculated accordingly. "penalties" if input penalties are to be used
        directly. The default is "prob".
    probF : float, optional
        demanded probability of keeping the food demand constraint (only 
        relevant if PenMet == "prob"). The default is 0.99.
    probS : float, optional
        demanded probability of keeping the solvency constraint (only 
        relevant if PenMet == "prob"). The default is 0.95.
    rhoF : float or None, optional 
        If PenMet == "penalties", this is the value that will be used for rhoF.
        if PenMet == "prob" and rhoF is None, a initial guess for rhoF will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for reaching 
        food demand. The default is None.
    rhoS : float or None, optional 
        If PenMet == "penalties", this is the value that will be used for rhoS.
        if PenMet == "prob" and rhoS is None, a initial guess for rhoS will 
        be calculated in GetPenalties, else this will be used as initial guess 
        for the penalty which will give the correct probability for solvency.
        The default is None.
    k : int, optional
        Number of clusters in which the area is to be devided. 
        The default is 9.
    k_using : "all" or a list of int i\in{1,...,k}, optional
        Specifies which of the clusters are to be considered in the model. 
        The default is the representative cluster [3].
    num_crops : int, optional
        The number of crops that are used. The default is 2.
    yield_projection : "fixed" or "trend", optional
        If "fixed", the yield distribtuions of the year prior to the first
        year of simulation are used for all years. If "trend", the mean of 
        the yield distributions follows the linear trend.
        The default is "fixed".
    sim_start : int, optional
        The first year of the simulation. The default is 2017.
    pop_scenario : str, optional
        Specifies which population scenario should be used. "fixed" uses the
        population of the year prior to the first year of the simulation for
        all years. The other options are 'Medium', 'High', 'Low', 
        'ConstantFertility', 'InstantReplacement', 'ZeroMigration', 
        'ConstantMortality', 'NoChange' and 'Momentum', referring to different
        UN_WPP population scenarios. All scenarios have the same estimates up 
        to (including) 2019, scenariospecific predictions start from 2020
        The default is "fixed".
    risk : int, optional
        The risk level that is covered by the government. Eg. if risk is 0.05,
        yields in the lower 5% quantile of the yield distributions will be 
        considered as catastrophic. The default is 5%.
    N : int, optional
        Number of yield samples to be used to approximate the expected value
        in the original objective function. The default is 10000.
    validation_size : None or int, optional
        if not None, the objevtice function will be re-evaluated for 
        validation with a higher sample size as given by this parameter. 
        The default is None.
    T : int, optional
        Number of years to cover in the simulation. The default is 25.
    seed : int, optional
        Seed to allow for reproduction of the results. The default is 201120.
    tax : float, optional
        Tax rate to be paied on farmers profits. The default is 1%.
    perc_guaranteed : float, optional
        The percentage that determines how high the guaranteed income will be 
        depending on the expected income of farmers in a scenario excluding
        the government. The default is 90%.
    ini_fund : float
        The default is 0.
        
        
    Returns
    -------
    settings : dict
        A dictionary that includes all of the above settings.
    

    """

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
        sys.exit("A non-valid penalty method was chosen (PenMet must " + \
                 "be either \"prob\" or \"penalties\").")     
        
    # create dictionary of settings
    settings =  {"PenMet": PenMet,
                 "probF": probF,
                 "probS": probS,
                 "rhoF": rhoF,
                 "rhoS": rhoS,
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
                 "ini_fund": ini_fund}   
     

    # return dictionary of all settings
    return(settings)

def SetParameters(settings, AddInfo_CalcParameters,\
                  wo_yields = False, VSS = False, \
                  console_output = None, logs_on = None):
    """
    
    Based on the settings, this sets most parameters that are needed as
    input to the model.    
    
    Parameters
    ----------
    settings : dict
        Dictionary of settings as given by DefaultSettingsExcept().
    AddInfo_CalcParameters : dict 
        Additional information from calculatings expected income and penalties
        which are not needed as model input.        
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
        
        - k: number of clusters in which the area is to be devided.
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
        - import: import needed to reach the desired probability for food
          security in 10^12kcal
        - ini_fund: initial fund size in 10^9$
        - tax: tax rate to be paied on farmers profits
        - prices: np.array of size (num_crops,) giving farm gate prices 
          farmers earn in 10^9$/10^6t
        - T: number of years covered in model
        - expected_incomes: expected income in secnario without government and
          probF = 95% for each clutser.               
        - guaranteed_income: np.array of size (T, len(k_using)) giving 
          the income guaranteed by the government for each year and cluster
          in case of catastrophe in 10^9$
        - crop_cal : np.array of size (num_crops,) giving the calorie content
          of the crops in 10^12kcal/10^6t
        - max_areas: np.array of size (len(k_using),) giving the upper 
          limit of area available for agricultural cultivation in each
          cluster
    other : dict
        Dictionary giving some additional inforamtion
        
        - slopes: slopes of the yield trends
        - constants: constants of the yield trends
        - yld_means: average yields 
        - residual_stds: standard deviations of the residuals of the yield 
          trends
        - prob_cat_year: probability for a catastrophic years given the 
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
    expected_incomes = AddInfo_CalcParameters["expected_incomes"]
    
    if "needed_import" in AddInfo_CalcParameters.keys():
        n_import = AddInfo_CalcParameters["needed_import"]
        if n_import < 0:
            n_import = 0
    else:
        n_import = 0
    
# 1. get cluster information (clusters given by k-Medoids on SPEI data) 
    with open("InputData/Clusters/Clustering/kMediods" + \
                        str(k) + "_PearsonDistSPEI_ProfitableArea.txt", "rb") as fp:  
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
    total_pop_scen = total_pop[np.where(scenarios == "Medium")[0][0],:]
    total_pop_UN = total_pop_scen[2015-1950]
    cluster_pop = np.zeros(len(k_using))
    for i, cl in enumerate(k_using):
        cluster_pop[i] = np.nansum(gridded_pop[clusters == cl])
    total_pop_GPW = np.sum(cluster_pop)
    pop_ratio = total_pop_GPW/total_pop_UN
    
# 4. Country shares of area (for price calculation)
    with open("InputData/Prices/CountryCodesGridded.txt", "rb") as fp:    
        country_codes_gridded = pickle.load(fp)
    country_codes = pd.read_csv("InputData/Prices/CountryCodes.csv")
    with open("InputData/Other/MaskProfitableArea.txt", "rb") as fp:   
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

# 5. Per person/day demand       
    # we use data from a paper on food waste 
    # (https://doi.org/10.1371/journal.pone.0228369) 
    # it includes country specific vlues for daily energy requirement per 
    # person (covering five of the countries in West Africa), based on data
    # from 2003. the calculation of the energy requirement depends on 
    # country specific values on age/gender of the population, body weight, 
    # and Physical Avticity Level.
    # Using the area shares as weights, this results in an average demand
    # per person and day of 2952.48kcal in the area we use.
    ppdemand = pd.read_csv("InputData/Other/CaloricDemand.csv")                           
    ppdemand["Shares"] = 0
    for c in ppdemand["Country"]:
        ppdemand.loc[ppdemand["Country"] == c, "Shares"] = \
            country_codes.loc[country_codes["CountryName"] == c, "Shares"].values
    ppdemand["Shares"] = ppdemand["Shares"] / np.sum(ppdemand["Shares"])        
    ppdemand = np.sum(ppdemand["Demand"] * ppdemand["Shares"])
    
# 6. cultivation costs of crops
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
    costs = np.transpose(np.tile(np.array([643.44, 281.22]), \
                                                     (len(k_using), 1)))
    # in 10^9$/10^6ha
    costs = 1e-3 * costs 
        
# 7. Energy value of crops
    # https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/
    # beltsville-human-nutrition-research-center/methods-and-application-
    # of-food-composition-laboratory/mafcl-site-pages/sr11-sr28/
    # Rice: NDB_No 20450, "RICE,WHITE,MEDIUM-GRAIN,RAW,UNENR" [kcal/100g]
    kcal_rice = 360 * 10000             # [kcal/t]
    # Maize: NDB_No 20014, "CORN GRAIN,YEL" (kcal/100g)
    kcal_maize = 365 * 10000            # [kcal/t]
    crop_cal = np.array([kcal_rice, kcal_maize])
    # in 10^12kcal/10^6t
    crop_cal = 1e-6 * crop_cal
    
# 8. Food demand
    # based on the demand per person and day (ppdemand) and assuming no change
    # of per capita daily consumption we use UN population scenarios for West 
    # Africa and scale them down to the area we use, using the ratio from 2015 
    # (from gridded GPW data)
    with open("InputData/Population/" + \
                  "UN_PopTotal_Prospects_WesternAfrica.txt", "rb") as fp:    
        total_pop = pickle.load(fp)
        scenarios = pickle.load(fp)
    if pop_scenario == "fixed":
        total_pop_scen = total_pop[np.where(scenarios == "Medium")[0][0],:]
        demand = np.repeat(ppdemand*365*total_pop_scen[(sim_start-1)-1950],\
                   T) # for fixed we use pop of 2016 if run starts 2017
        demand = demand * pop_ratio
    else:
        total_pop_scen = total_pop[np.where(scenarios ==  \
                                                pop_scenario)[0][0],:]
        demand = ppdemand*365*total_pop_scen[(sim_start-1950): \
                                                (sim_start-1950+T)]
        demand = demand * pop_ratio
    # in 10^12 kcal
    demand = 1e-12 * demand

# 9. guaranteed income as share of expected income w/o government
    # if expected income is not given...
    guaranteed_income = np.repeat(expected_incomes[np.newaxis, :], T, axis=0)    
    guaranteed_income = perc_guaranteed * guaranteed_income                     
    # guaraneteed income per person assumed to be constant over time, 
    # therefore scale with population size
    if not pop_scenario == "fixed":
        total_pop_ratios = total_pop_scen / \
                                total_pop_scen[(sim_start-1)-1950]
        guaranteed_income = (guaranteed_income.swapaxes(0,1) * \
                            total_pop_ratios[(sim_start-1950): \
                                                (sim_start-1950+T)]) \
                            .swapaxes(0,1)
          
# 10. prices for selling crops, per crop and cluster
    with open("InputData//Prices/CountryAvgFarmGatePrices.txt", "rb") as fp:    
        country_avg_prices = pickle.load(fp)
    # Gambia is not included in our area
    country_avg_prices = country_avg_prices.drop(axis = 0, labels = [4])             
        
    price_maize = np.nansum(country_avg_prices["Maize"].values * \
                                            country_codes["Shares"].values)
    price_rice = np.nansum(country_avg_prices["Rice"].values * \
                          country_codes["Shares excl. Mauritania"].values)
    
    prices = np.transpose(np.tile(np.array([price_rice, \
                                            price_maize]), (len(k_using), 1)))
    # in 10^9$/10^6t
    prices = 1e-3 * prices 
    
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
    with open("InputData/YieldTrends/DetrYieldAvg_k" + \
                              str(k) + "_ProfitableArea.txt", "rb") as fp:   
         pickle.load(fp) # yields_avg 
         pickle.load(fp) # avg_pred
         pickle.load(fp) # residuals
         pickle.load(fp) # residual_means
         residual_stds = pickle.load(fp)
         pickle.load(fp) # fstat
         constants = pickle.load(fp)
         slopes = pickle.load(fp)
         pickle.load(fp) # crops
    residual_stds = residual_stds[:, [i - 1 for i in k_using]]
    constants = constants[:, [i - 1 for i in k_using]] 
    slopes = slopes[:, [i - 1 for i in k_using]]

    # get yield realizations:
    # what is the probability of a catastrophic year for given settings?
    printing("\nOverview on yield samples", console_output = console_output, logs_on = logs_on)
    prob_cat_year = RiskForCatastrophe(risk, len(k_using))
    printing("     Prob for catastrophic year: " + \
             str(np.round(prob_cat_year*100, 2)) + "%", \
             console_output = console_output, logs_on = logs_on)    
    # create realizations of presence of catastrophic yields and corresponding
    # yield distributions
    np.random.seed(seed)
    cat_clusters, terminal_years, ylds, yld_means = \
          YieldRealisations(slopes, constants, residual_stds, sim_start, \
                           N, risk, T, len(k_using), num_crops, \
                           yield_projection, VSS, wo_yields)
    # probability to not have a catastrophe
    no_cat = np.sum(terminal_years == -1) / N
    printing("     Share of samples without catastrophe: " + str(np.round(no_cat*100, 2)), \
              console_output = console_output, logs_on = logs_on) 
    # share of non-profitable crops
    share_rice_np = np.sum(ylds[:,:,0,:] < y_profit[0,:])/np.sum(~np.isnan(ylds[:,:,0,:]))
    printing("     Share of cases with rice yields too low to provide profit: " + \
             str(np.round(share_rice_np * 100, 2)), console_output = console_output, \
             logs_on = logs_on)
    share_maize_np = np.sum(ylds[:,:,1,:] < y_profit[1,:])/np.sum(~np.isnan(ylds[:,:,1,:]))
    printing("     Share of cases with maize yields too low to provide profit: " + \
             str(np.round(share_maize_np * 100, 2)), console_output = console_output, \
             logs_on = logs_on)
    # in average more profitable crop
    exp_profit = yld_means * prices - costs
    avg_time_profit = np.nanmean(exp_profit, axis = 0)
    more_profit = np.argmax(avg_time_profit, axis = 0) # per cluster
    printing("     On average more profit (per cluster): " + \
             str([crops[i] for i in more_profit]), \
             console_output = console_output, logs_on = logs_on)
    # in average more productive crop
    avg_time_production = np.nanmean(yld_means, axis = 0)
    more_food = np.argmax(avg_time_production, axis = 0)
    printing("     On average higher productivity (per cluster): " + \
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
            "import": n_import,
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
    
    population_information = {"total_pop_scen": total_pop_scen,
                              "pop_area_ratio2015": pop_ratio}
        
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
        The number of crops that are used.

    Returns
    -------
    prob : The probability for a year to be catastrophic.

    """
    res = 0
    for i in range(1, num_clusters + 1):
        res += comb(num_clusters, i) * (risk**i) * (1-risk)**(num_clusters-i)
             
    return(res)

def YieldRealisations(yld_slopes, yld_constants, resid_std, sim_start, N, \
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
        The first year of the simulation..
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
    cat_clusters, terminal_years = CatastrophicYears(risk, \
                                    N, T, num_clusters, VSS)
    
    # generating yields according to catastrophes
    if wo_yields:
        ylds = []
    else:
        ylds, yld_means = ProjectYields(yld_slopes, yld_constants, resid_std, sim_start, \
                             N, cat_clusters, terminal_years, T, risk, \
                             num_clusters, num_crops, yield_projection, \
                             VSS)
        
    # comparing with np.nans leads to annoying warnings, so we turn them off        
    np.seterr(invalid='ignore')
    ylds[ylds<0] = 0
    np.seterr(invalid='warn')
    
    return(cat_clusters, terminal_years, ylds, yld_means)

def CatastrophicYears(risk, N, T, num_clusters, VSS):
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
        in the original objective function. The default is 3500.
    T : int
        Number of years to cover in the simulation. The default is 25.
    k : int
        Number of clusters in which the area is to be devided. 
        The default is 1.
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

def ProjectYields(yld_slopes, yld_constants, resid_std, sim_start, \
                  N, cat_clusters, terminal_years, T, risk, \
                  num_clusters, num_crops, yield_projection, \
                  VSS = False):
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
        The first year of the simulation..
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

    Returns
    -------
    ylds : np.array of size (N, T, num_crops, len(k_using)) 
        yield samples in 10^6t/10^6ha according to the presence of 
        catastrophes
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
