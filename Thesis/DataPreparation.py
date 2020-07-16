# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:52:19 2020

@author: leip
"""

# %% IMPORTING NECESSARY PACKAGES AND SETTING WORKING DIRECTORY

import numpy as np
from os import chdir
import pickle
import pandas as pd
import matplotlib.pyplot as plt

chdir('/home/debbora/IIASA/FinalVersion')

import Functions_DataPreparation as OF

# %% General Definition

# define region corresponding to West Africa
lon_min = -19 
lon_max = 10.5 
lat_min = 3.0
lat_max = 18.5  

# %% 1) Preparing Drought Index data

# SPI:
# Data downloaded from: National Center for Atmospheric Research/University 
# Corporation for Atmospheric Research. 2013. Standardized Precipitation Index 
# (SPI) for Global Land Surface (1949-2012). Research Data Archive at the 
# National Center for Atmospheric Research, Computational and Information
# Systems Laboratory. 
# https://doi.org/10.5065/D6086397. 
# Accessed April 29th 2020
# Gridded monthly data in 1° resolution from Jan. 1949 to Dec. 2012

# SPEI:
# he Standardized Precipitation-Evapotranspiration Index (SPEI) was proposed 
# to overcome problems of the SPI (mainly not considering temperature) and the 
# sc-PDSI (mainly having only one timescale).
# The SPEIbase covers the period from January 1901 to December 2018 with 
# monthly frequency, with global coverage at 0.5° resolution. Values are 
# typically between -2.5 and 2.5 (exceeding probabilities ca. 0.006 and 0.994 
# respectively), theoretical limits are ±infinity.
# Source: Vicente-Serrano, Sergio M., Santiago Beguería, and Juan I. 
# López-Moreno. "A multiscalar drought index sensitive to global warming: the
# standardized precipitation evapotranspiration index." 
# Journal of climate 23.7 (2010): 1696-1718.
# Data downloaded from: http://hdl.handle.net/10261/202305
# Accessed April 29th 2020

# read SPEI and SPI data
OF.ReadAndSave_DroughtIndex("spei03", lon_min, lon_max, lat_min, lat_max)
OF.ReadAndSave_DroughtIndex("spei01", lon_min, lon_max, lat_min, lat_max)
# spi is in 1° resolution and only January 1949 to December 2012
OF.ReadAndSave_DroughtIndex("spi03", lon_min, lon_max, lat_min, lat_max)

# remove linear trend in timeseries per cell; a p-value below 0.05 is
# considered to be statistically significant
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "spei03_WA_filled.txt", "rb") as fp:    
    spei03_filled = pickle.load(fp)
    lats_spei03 = pickle.load(fp)
    lons_spei03 = pickle.load(fp)
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "mask_spei03_WA.txt", "rb") as fp:    
    mask_spei03 = pickle.load(fp)    
          
spei03_detrend, p_val_slopes, slopes, intercepts = \
                      OF.DetrendDataLinear(spei03_filled, mask_spei03)
np.sum(p_val_slopes <= 0.05)  # 3148     
np.sum(p_val_slopes > 0.05)   # 359
# for the vast majority (89.76%) the linear trend is significant (even though
# slopes are very small), therefore we will work with linear detrended SPEI 
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                         "spei03_WA_detrend.txt", "wb") as fp:    
    pickle.dump(spei03_detrend, fp, protocol = 2)    
    pickle.dump((p_val_slopes, slopes, intercepts), fp, protocol = 2)  
    pickle.dump(lats_spei03, fp, protocol = 2)
    pickle.dump(lons_spei03, fp, protocol = 2)
    
# %% 2) Preparing AgMIP yield data
    
# "The Global Gridded Crop Model Intercomparison phase 1 simulation dataset"
# by C. Müller et. al. (2019)
# data downloaded from links in the references of the paper
# Accessed between November 27th and December 4th 2019
# Global yearly gridded (resolution 0.5°) yield data for several crops and 
# data on additional variables depending on the model
    
# File names: 
    # [model]−[climate]−[clim.scenario]−[sim.scenario]−[variable]−[crop]−
    # [timestep]−[start−year]−[end−year].nc4
    # clim.scenario is always "hist", timestep always "annual", included for
    # consistency with ISIMIP file naming convention
    
# List of models:
    # CGMS-WODFOST
    # CLM-crop
    # EPIC-BOKU
    # EPIC-IIASA
    # EPIC-TAMU
    # GEPIC
    # LPJ-GUESS
    # LPJmL
    # ORCHIDEE-crop
    # pAPSIM
    # pDSSAT
    # PEGASUS
    # PEPIC
    # PRYSBI2
        
# List of possible climate datasets (not every model uses all of them):
# (not all models use full time range, e.g. GEPIC uses 1948-2008 for pgfv2)
    # agcfsr - AgCFSR [1980-2010]
    # agmerra - AgMERRA [1980-2010]
    # cfsr - CFSR reanalysis [1979-2012]
    # erai - ERA-I reanalysis [1979-2012]
    # grasp - GRASP [1961-2010]
    # gswp - GSWP [1901-2010]
    # princeton - Prineton GF [1948-2008]
    # pgfv2 - Princeton GF version 2 [1901-2012]
    # watch - WATCH (WFD) [1958-2001]
    # wfdei.cru - WFDELCRU [1979-2009]
    # wfdei.gpcc - WFDELGPCC [1979-2009]
    
# Irrigation scenarios:
    # firr - full irrigation
    # noirr - no irrigation

# List of possible variable (not every model has all of them):
    # yield  - Crop yields given in t/(ha*yr)
    # pirrww - Applied irrigation water in mm/yr (for firr scenarios)
    # biom   - Total Above ground biomass yield in t/(ha*yr)
    # aet    - Actual growing season evapotranspiration in mm/yr
    # plant-day - Actual planting day as day of year
    # anth-day - Days from planting to anthesis as days from planting
    # maty-day - Days from planting to maturity as days from planting
    # initr - Nitrogen application rate in kg/(ha*yr)
    # leach - Nitrogen leached in kg/(ha*yr)
    # sco2 - Soil carbon emissions in kg/ha
    # sn2o - Nitrous oxide emissions in kg/ha
    # gsprcp - Accumulated precipitation, planting to harvest in mm/yr
    # gsrsds - Growing season incoming solar in w/(m^2*yr)
    # sumt - Sum of daily mean temps, planting to harvest (deg C-days yr−1)
    
# List of crops:
    # Most model report at least the P1 crops (priority one crops):
    # maize, wheat, rice, soybean 
    # Some models report additional P2 crops (priority two crops) including:
    # barley, millet, rapeseed, rye, sorghum, sugar beet, sugar cane, cotton,
    # cassava, groundnut, field pea, sunflower, dry bean, potato, managed 
    # grassland
    
    
    
Model = ["EPIC-IIASA", "EPIC-Boku", "GEPIC", "pAPSIM"]
model = ["epic-iiasa", "epic-boku", "gepic", "papsim"]
climate = [["agmerra", "grasp"], ["grasp", "grasp"], ["pgfv2"], \
           ["agmerra", "grasp"]]
harm_scenario = [["default", "fullharm"], ["default", "fullharm"], \
                 ["default"], ["default", "default"]]
irri_scenarios = ["firr", "noirr"]
crops = [["Soy", "Maize", "Wheat", "Rice"], ["Soy", "Maize", "Wheat", "Rice"],\
         ["Soy", "Maize", "Wheat", "Rice"], ["Soy", "Maize", "Wheat"]]
crop_abbrvs = [["soy", "mai", "whe", "ric"],["soy", "mai", "whe", "ric"], \
               ["soy", "mai", "whe", "ric"], ["soy", "mai", "whe"]]
year_start = [[1980, 1961], [1961, 1961], [1948], [1980, 1961]]
year_end = [[2010, 2010], [2010, 2010], [2008], [2010, 2010]]
var_names = [[["yield", "plant-day", "aet", "gsprcp", "initr", "maty-day"], \
              ["yield", "plant-day", "gsprcp"]], \
             [["initr", "yield"], ["initr", "yield"]], \
             [["yield", "plant-day", "pirrww", "gsprcp", "initr"]], \
             [["yield", "plant-day", "pirrww", "gsprcp", "initr", "aet", \
               "gsrsds", "maty-day", "sumt"], ["yield", "plant-day", "pirrww",\
               "gsprcp", "initr", "aet", "sumt"]]]

# reading all versions used in regression analysis and saving them reduced to 
# West Africa in format we can directly use

for irri_scenario in irri_scenarios:
    for m in range(0, len(Model)):
        for scen in range(0, len(harm_scenario[m])):
                for cr in range(0, len(crops[m])):
                    for var in range(0, len(var_names[m][scen])):
                        OF.ReadAndSave_AgMIP(Model[m], model[m], \
                             climate[m][scen], harm_scenario[m][scen], \
                             irri_scenario, crops[m][cr], crop_abbrvs[m][cr], \
                             year_start[m][scen], year_end[m][scen], \
                             var_names[m][scen][var], lat_min, lon_min, \
                             lat_max, lon_max)    
             
# %% 3) Preparing GDHY yield data    

# Data downloaded from https://doi.pangaea.de/10.1594/PANGAEA.909132
# Annual time series data of 0.5-degree grid-cell yield estimates of major 
# crops (maize, rice, wheat, soybean) worldwide for the period 1981-2016 in 
# tonnes per hectare. Growing season categories are based on Sacks et al. 2010.
# Iizumi, Toshichika (2019): Global dataset of historical yields v1.2 and v1.3
# aligned version. PANGAEA, https://doi.org/10.1594/PANGAEA.909132, 
# Supplement to: Iizumi, Toshichika; Sakai, T (2020): The global dataset of 
# historical yields for major crops 1981–2016. Scientific Data, 7(1), 
# https://doi.org/10.1038
# Accessed April 26th 2020
                        
var_names = ["maize", "maize_major", "maize_second", "rice", "rice_major", \
             "rice_second", "soybean", "wheat", "wheat_spring", "wheat_winter"]

for v_name in var_names:
    # yields are saved per year, so we go through all years and save them in 
    # one array
    for year in range(1981, 2017):
        var_year, mask_year, lats_WA, lons_WA = OF.Read_GDHY(v_name, year, \
                                            lon_min, lon_max, lat_min, lat_max)
        if year == 1981:
            var = var_year
            mask = mask_year
        else:
            var  = np.dstack((var, var_year))
            mask = np.dstack((mask, mask_year))
    var_save = np.moveaxis(var, 2,0)
    mask_save = np.moveaxis(mask, 2, 0)
    mask_save = np.prod(mask_save, axis = 0)*(-1) +1 # chagne 0 and 1 so it 
                                                     # fits with other masks                       
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                              v_name + "_yld.txt", "wb") as fp:    
        pickle.dump(var_save, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                             v_name + "_mask.txt", "wb") as fp:    
        pickle.dump(mask_save, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
# %% 3) Preparing data from crop calender
        
# Crop Calendar of 2000, downloaded from https://nelson.wisc.edu/sage/data-and
# -models/crop-calendar-dataset/netCDF0-5degree.php
# Global gridded data (resolution 0.5°) on planting date, start of planting,
# end of planting, range of planting dates, harvest date, start of harvest, 
# end of harvest, range of harvest dates, days between planting and harvest
# Sacks, W.J., D. Deryng, J.A. Foley, and N. Ramankutty (2010). Crop
# planting dates: An analysis of global patterns. Global Ecology and
# Biogeography, 19: 607-620.
# Accessed January 13th 2020

# we use planting dates and harvest dates of following crops:     
crops = ["wheat_spring", "wheat_winter", "soybean", "rice_major", \
                                 "rice_second", "maize_major", "maize_second"]
for i in range(0, len(crops)):
    crop = crops[i]
    OF.ReadAndSave_CropCalendar(crop, lon_min, lon_max, lat_min, lat_max)

# %% Preparing AgMERRA data

# TODO do I need it? If so, what to do about too big arrays??
    
 
# %% Preparing CRU data (for absolute water deficit)

# TODO take out diurnal temperature range because not used?

# The CRU (Climatic Research Unit) data is saved with CEDA (Centre for Environ-
# mental Data Analysis).
# Downloaded from: http://wps-web1.ceda.ac.uk/submit/form?proc_id=Subsetter
# Used Dataset: CRU TS 4.03
# Accessed November 27th 2019
# Monthly gridded dataset of 0.5° resolution from January 1901 to December 2018
# Available variables are: precipitation, near-surface temperature, 
# near-surface temperature maximum, potential evapotranspiration, ground frost 
# frequency, cloud cover, wet day frequency, diurnal temperature range, vapour 
# pressure, near-surface temperature minimum
    
# reading data of existing variables
variables = ["Precipitation", "PET", "DiurnalTemp"]  
variables_abbrv = ["pre", "pet" , "dtr"]
for v in range(0, len(variables)):
    OF.ReadAndAgg_CRU(variables[v], variables_abbrv[v])
    
# creations water deficit data as difference between precipitation and 
# potential evapotranspiration (PET)   
with open("IntermediateResults/PreparedData/CRU/" + \
                                          "Precipitation_WA.txt", "rb") as fp:    
    pre = pickle.load(fp)
with open("IntermediateResults/PreparedData/CRU/" + \
                                                  "PET_WA.txt", "rb") as fp:    
    PET = pickle.load(fp)

WaterDeficit = pre-PET
WaterDeficit03 = np.zeros(WaterDeficit.shape)
[n_t, n_lat, n_lon] = WaterDeficit03.shape
for i in range(0, n_lat):
    for j in range(0, n_lon):
        WaterDeficit03[:, i, j] = np.array(pd.DataFrame(WaterDeficit[:, i, j])\
                           .rolling(window=3, center = False).mean()).flatten()
with open("IntermediateResults/PreparedData/CRU/" + \
                                           "WaterDeficit_WA.txt", "wb") as fp:    
    pickle.dump(WaterDeficit, fp)
with open("IntermediateResults/PreparedData/CRU/" + \
                                          "WaterDeficit03_WA.txt", "wb") as fp:    
    pickle.dump(WaterDeficit03, fp)


mask_WaterDeficit = np.zeros([n_lat, n_lon])
for t in range(0, n_t):
   for i in range(0, n_lat):
       for j in range(0, n_lon):        
           if ~(np.isnan(WaterDeficit[t, i, j])): 
               mask_WaterDeficit[i,j] = 1
with open("IntermediateResults/PreparedData/CRU/" + \
                                      "mask_WaterDeficit_WA.txt", "wb") as fp:    
     pickle.dump(mask_WaterDeficit, fp)    

WaterDeficit03_detrend, p_val_slopes, slopes, intercepts = \
                  OF.DetrendDataLinear(WaterDeficit03, mask_WaterDeficit)        
     
with open("IntermediateResults/PreparedData/CRU/" + \
                                 "WaterDeficit03_WA_detrend.txt", "wb") as fp:    
    pickle.dump(WaterDeficit03_detrend, fp)    
    pickle.dump((p_val_slopes, slopes, intercepts), fp)  
    
# %% Population data from UN World Population Prospects 
     
# Data Download: https://population.un.org/wpp/Download/
#                                               Probabilistic/Population/
# Accessed April 12th 2020
# Short description of Scenarios: https://population.un.org/wpp/
#                                               DefinitionOfProjectionVariants/
# Report: "World Population Prospects 2019: Methodology of the United Nations 
#          Population Estimates and Projections"
#  (https://population.un.org/wpp/Publications/Files/WPP2019_Methodology.pdf)
#
# Secnarios:  
# scenarioname (fertility - mortality - international migration)
# Low Fertility (low - normal - normal)
# Medium Fertility (medium - normal - normal)
# High Fertility (high - normal - normal)
# Constant Fertility (constant as of 2015-2020 - normal - normal)
# Instant-replacement Fertility (instant replacement as of 2020-2025 - normal 
#                                                                    - normal)
# Momentum (instant replacement as of 2020-2025 - constant as of 2015-2020 
#                                                      - zero as of 2020-2025)
# Constant Mortality (medium - constant as of 2015-2020 - normal)
# No change (constant as of 2015-2020 - constant as of 2015-2020 - normal)
# Zero-migration (medium - normal - zero as of 2020-2025)
#
# Data is presented in thousands
     
# read data
PopData = pd.read_csv("Data/Population/WPP2019_TotalPopulationBySex.csv")

# get different scenario names
scenarios = np.array(PopData["Variant"])
scenarios = scenarios[sorted(np.unique(scenarios, \
                                       return_index = True)[1])]
scenarioIDs = np.array(PopData["VarID"])
scenarioIDs = scenarioIDs[sorted(np.unique(scenarioIDs, \
                                           return_index = True)[1])]
scenarios = scenarios[scenarioIDs < 20]
scenarios[3] = "ConstantFertility"
scenarios[4] = "InstantReplacement"
scenarios[5] = "ZeroMigration"
scenarios[6] = "ConstantMortality"
scenarios[7] = "NoChange"

# reduce data to Western Africa
regions = np.unique(np.array(PopData["Location"]))
WhichRegion = "Western Africa"
WhichRegion_save = "WesternAfrica"
WhichValues = "PopTotal"
PopData_reg = PopData[PopData["Location"]== WhichRegion]

# estimates of past population numbers are saved in scenario 2 - Medium
past_estimates = np.zeros(71)
for t in range(1950, 2021):
    past_estimates[t-1950] = PopData_reg.loc[(PopData_reg.VarID==2) & \
                                    (PopData_reg.Time==t), WhichValues].values

total_pop = np.zeros([len(scenarios), 2101-1950])
for scen in range(0, len(scenarios)):
    total_pop[scen, 0:(2021-1950)] = past_estimates
    
for s in range(2, 11):
    for t in range(2021, 2101):
        total_pop[s-2, t-1950] = PopData_reg.loc[(PopData_reg.VarID == s) \
                                  & (PopData_reg.Time==t), WhichValues].values
        
total_pop = total_pop * 1000 # to have actual numbers
    
with open("IntermediateResults/PreparedData/Population/UN_" + \
          WhichValues + "_Prospects_" + WhichRegion_save + ".txt", "wb") as fp:    
    pickle.dump(total_pop, fp)
    pickle.dump(scenarios, fp)
    pickle.dump(np.array(range(1950, 2101)), fp)

# %% Gridded Populaion of the World (SEDAC)
    
# Data download: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population
#             -count-adjusted-to-2015-unwpp-country-totals-rev11/data-download    
# Accessed Juune 4th 2020
# Documentation: https://sedac.ciesin.columbia.edu/downloads/docs/
#                                       gpw-v4/gpw-v4-documentation-rev11.pdf

# Gridded data on population in 0.5 degree resolution, provides "estimates of 
# population count for the years 2000, 2005, 2010, 2015, and 2020, consistent 
# with national censuses and population registers with respect to relative 
# spatial distribution, but adjusted to match United Nations country totals.
    
OF.ReadAndReduce_GPW(lat_min, lon_min, lat_max, lon_max)    


# %% Farm gate prices for countries in West Africa

# Data download: http://www.fao.org/faostat/en/#data/PP
# "This sub-domain contains data on Agriculture Producer Prices. These are 
# prices received by farmers for primary crops, live animals and livestock 
# primary products as collected at the point of initial sale (prices paid at 
# the farm-gate). Annual data are provided from 1991, while mothly data from 
# January 2010 for 180 country and 212 product."

prices_raw = pd.read_csv("Data/Prices/FAOSTAT_data_6-11-2020.csv")
prices_raw = prices_raw[["Area","Item", "Year","Value"]]
countries = np.array(prices_raw["Area"])
countries = countries[sorted(np.unique(countries, return_index = True)[1])]
items = np.array(prices_raw["Item"])
items = items[sorted(np.unique(items, return_index = True)[1])]
years = np.array(prices_raw["Year"])
years = years[sorted(np.unique(years, return_index = True)[1])]
col = ["indianred", "navy"]
fig = plt.figure(figsize = (24, 13.5))
fig.subplots_adjust(wspace=0.25, hspace=0.5)
for c, country in enumerate(countries):
    fig.add_subplot(4,4,c+1)
    for idx, item in enumerate(items):
        plt.plot(prices_raw.loc[(prices_raw.Area==country) & \
                                (prices_raw.Item==item)]["Year"],
                 prices_raw.loc[(prices_raw.Area==country) & \
                           (prices_raw.Item==item)]["Value"], color = col[idx])
        plt.xlim([1990, 2020])
        if (c%4) == 0:
            plt.ylabel("Price in USD/t")
        if c >= len(countries) - 4:
            plt.xlabel("Years")
        plt.title(country)
plt.show()
plt.suptitle("Farm-gate prices in USD per tonne for maize (red)" + \
                                     " and rice (blue)") 
fig.savefig("Figures/StoOpt/prices_ts.png", \
                            bbox_inches = "tight", pad_inches = 0.5)

prices = np.empty([len(years), len(countries), len(items)])
prices.fill(np.nan)
for idx_c, country in enumerate(countries):
    for idx_i, item in enumerate(items):
        for idx_y, year in enumerate(years):
            if(len(prices_raw.loc[(prices_raw.Area==country) & \
                               (prices_raw.Item==item) & \
                               (prices_raw.Year==year)]["Value"].values)>0):
                prices[idx_y, idx_c, idx_i] = \
                    prices_raw.loc[(prices_raw.Area==country) & \
                                   (prices_raw.Item==item) & \
                                   (prices_raw.Year==year)]["Value"].values[0]
        
prices = np.nanmean(prices, axis = 0)

x = np.arange(len(countries))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (24, 13.5))
rects1 = ax.bar(x - width/2, prices[:,0], width, label='Maize', \
                                        color = col[0], alpha = 0.5)
rects2 = ax.bar(x + width/2, prices[:,1], width, label='Rice', \
                                        color = col[1],  alpha = 0.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average farm-gate prices in USD/t')
ax.set_title('Average farm-gate prices per country and crop')
ax.set_xticks(x)
ax.set_xticklabels(countries)
ax.legend()
plt.show()
fig.savefig("Figures/StoOpt/prices_CountryAvg.png", \
                            bbox_inches = "tight", pad_inches = 0.5)

prices = pd.DataFrame(prices)
prices.insert(0, 'Countries', countries)
prices.columns = ["Countries", "Maize", "Rice"]

with open("IntermediateResults/PreparedData/Prices/" + \
                              "CountryAvgFarmGatePrices.txt", "wb") as fp:    
    pickle.dump(prices, fp)


