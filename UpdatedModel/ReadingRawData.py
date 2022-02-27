# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:06:26 2021

@author: leip
"""

# %% IMPORTING NECESSARY PACKAGES AND SETTING WORKING DIRECTORY

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# import other modules
import numpy as np
import pickle
import pandas as pd

import ModelCode.DataPreparation as DP

if not os.path.isdir("ProcessedData"):
    os.mkdir("ProcessedData")
if not os.path.isdir("InputData/Visualization"):
    os.mkdir("InputData/Visualization")

print("Reading raw data ...", flush = True)

# %% 1. General Definition

print("... general settings", flush = True)

# define region corresponding to West Africa
lon_min = -19 
lon_max = 10.5 
lat_min = 3.0
lat_max = 18.5  

with open("InputData/Other/AreaExtent.txt", "wb") as fp:
    pickle.dump([lat_min, lat_max, lon_min, lon_max], fp)
    pickle.dump(["lat_min", "lat_max", "lon_min", "lon_max"], fp)
# creates InputData/Other/AreaExtent.txt
    
lats_WA = np.arange(lat_min, lat_max, 0.5) + 0.25    
lons_WA = np.arange(lon_min, lon_max, 0.5) + 0.25    

with open("InputData/Other/LatsLonsArea.txt", "wb") as fp:
    pickle.dump(lats_WA, fp)
    pickle.dump(lons_WA, fp)
# creates InputData/Other/LatsLonsArea.txt

# %% 2. SPEI (~20sec)
    
print("... SPEI", flush = True)

# The Standardized Precipitation-Evapotranspiration Index (SPEI) was proposed 
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

DP.ReadAndSave_SPEI03(lon_min, lon_max, lat_min, lat_max)
# creates ProcessedData/SPEI03_WA_masked.txt
#         ProcessedData/SPEI03_WA_filled.txt
#         ProcessedData/mask_SPEI03_WA.txt

# %% 3. Population data from UN World Population Prospects 
     
print("... UN World Population data", flush = True)

# Data Download: https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv
# Accessed April 12th 2020
# annual data, historic part covers 1950-2019, projections cover 2020-2100
# values per country
# Short description of Scenarios: https://population.un.org/wpp/DefinitionOfProjectionVariants/
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
  
DP.ReadAndSave_UNWPP(WhichRegion = "Western Africa", WhichValues = "PopTotal")
# creates InputData/Population/UN_PopTotal_Prospects_WestAfrica.txt

# %% 4. Gridded Populaion of the World (SEDAC)
    
print("... gridded population data", flush = True)

# Data download: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population
#             -count-adjusted-to-2015-unwpp-country-totals-rev11/data-download    
# Accessed Juune 4th 2020
# Documentation: https://sedac.ciesin.columbia.edu/downloads/docs/
#                                       gpw-v4/gpw-v4-documentation-rev11.pdf
# License: The GPW data collection is licensed under the Creative Commons 
#          Attribution 4.0 International License 
#          http://creativecommons.org/licenses/by/4.0

# Gridded data on population in 0.5 degree resolution, provides "estimates of 
# population count for the years 2000, 2005, 2010, 2015, and 2020, consistent 
# with national censuses and population registers with respect to relative 
# spatial distribution, but adjusted to match United Nations country totals.
    
DP.ReadAndReduce_GPW(lat_min, lon_min, lat_max, lon_max)    
# creates InputData/Population/GPW_WA.txt
#         InputData/Population/land_area.txt  
#         InputData/Prices/CountryCodesGridded.txt

# Mapping of SEDAC cuntry codes to country names:
country_codes = {"Code": np.array([204,854,120,384,226,270,288,324,
                                   624,430,466,478,562,566,686,694,768]),
                 "CountryName" : np.array(["Benin","Burkina Faso",
                                          "Cameroon","Cote d'Ivoire",
                                          "Equatorial Guinea","Gambia",
                                          "Ghana","Guinea","Guinea-Bissau",
                                          "Liberia","Mali","Mauritania",
                                          "Niger","Nigeria","Senegal",
                                          "Sierra Leone","Togo"])}
country_codes = pd.DataFrame(country_codes)
country_codes.to_csv("InputData/Prices/CountryCodes.csv", index = False)
# creates InputData/Prices/CountryCodes.csv

# %% 5. Preparing GDHY yield data    

print("... GDHY", flush = True)

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

DP.ReadAndSave_GDHY("maize_major", lon_min, lon_max, lat_min, lat_max)
# creates ProcessedData/maize_yld.txt
# creates ProcessedData/maize_mask.txt
DP.ReadAndSave_GDHY("rice_major", lon_min, lon_max, lat_min, lat_max)
# creates ProcessedData/rice_mask.txt
# creates ProcessedData/rice_mask.txt

# combine masks to get mask describing where both crops have data
with open("ProcessedData/rice_mask.txt", "rb") as fp:    
    rice_mask = pickle.load(fp)
with open("ProcessedData/maize_mask.txt", "rb") as fp:    
    maize_mask = pickle.load(fp)

yld_mask = rice_mask * maize_mask

print(np.sum(yld_mask == 1))
# 861

with open("ProcessedData/yld_mask.txt", "wb") as fp:    
    pickle.dump(yld_mask, fp)
# creates ProcessedData/yld_mask.txt

# %% 6. Farm gate prices for countries in West Africa

print("... farm gate prices", flush = True)

# Data download: http://www.fao.org/faostat/en/#data/PP
# countries: Benin, Burkina Faso, Cameroon, Cote d'Ivoire, Gambia, Ghana, 
# Guinea, Guinea-Bissau, Mali, Mauritania, Niger, Nigeria, Senegal,
# Sierra Leone, Togo 
# crops: Rice, paddy; Maize
# Variable: Producer Price (USD/tonne), annual values, 1991-2018
# "This sub-domain contains data on Agriculture Producer Prices. These are 
# prices received by farmers for primary crops, live animals and livestock 
# primary products as collected at the point of initial sale (prices paid at 
# the farm-gate). Annual data are provided from 1991, while mothly data from 
# January 2010 for 180 country and 212 product."

# We calculate average prices for each country using the given timeseries.
# Later, we calculate a single producer price per crop, by taking a
# weighted average of the country specific prices, using the country land 
# areas as weight.

DP.VisualizeAndPrepare_ProducerPrices()
# creates InputData/Prices/CountryAvgFarmGatePrices.txt
#         InputData/Visualization/ProducerPrices_CountryAvg.png
#         InputData/Visualization/ProducerPrices.png  

# %% 7. Crop Cultivation Costs

print("... crop cultivation costs", flush = True)

# Values from literature research 

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
costs = np.array([643.44, 281.22])

# in 10^9$/10^6ha
costs = 1e-3 * costs 

with open("InputData/Other/CultivationCosts.txt", "wb") as fp:
    pickle.dump(costs, fp)
# creates InputData/Other/CultivationCosts.txt


# %% 8. Energy value of crops

print("... energy value of crops", flush = True)

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

with open("InputData/Other/CalorieContentCrops.txt", "wb") as fp:
    pickle.dump(crop_cal, fp)
# creates InputData/Other/CalorieContentCrops.txt

# %% 9. Average calorie demand per person and day

print("... average calorie demand per person", flush = True)

# we use data from a paper on food waste 
# (https://doi.org/10.1371/journal.pone.0228369) 
# it includes country specific vlues for daily energy requirement per 
# person (covering five of the countries in West Africa), based on data
# from 2003. the calculation of the energy requirement depends on 
# country specific values on age/gender of the population, body weight, 
# and Physical Avticity Level.
# Using the area shares as weights, this results in an average demand
# per person and day of 2952.48kcal in the area we use.

CaloricDemand = {"Country": np.array(["Burkina Faso", "Cote d'Ivoire",
                                      "Ghana", "Mali", "Senegal"]),
                 "Demand" : np.array([3046.54, 3050.81, 2495.38, 3175.53, 3056.10])}
CaloricDemand = pd.DataFrame(CaloricDemand)
CaloricDemand.to_csv("ProcessedData/CountryCaloricDemand.csv", index = False)
# creates ProcessedData/CountryCaloricDemand.csv
                