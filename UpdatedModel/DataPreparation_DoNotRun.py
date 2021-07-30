# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 20:05:27 2021

@author: leip
"""


# %% IMPORTING NECESSARY PACKAGES AND SETTING WORKING DIRECTORY

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import numpy as np
from os import chdir
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl as xl

import ModelCode.DataPreparation as DP

# %% General Definition

# define region corresponding to West Africa
lon_min = -19 
lon_max = 10.5 
lat_min = 3.0
lat_max = 18.5  

# %% Population data from UN World Population Prospects 
     
# Data Download: https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv
# Accessed April 12th 2020
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
    
DP.ReadAndReduce_GPW(lat_min, lon_min, lat_max, lon_max)    

# %% Farm gate prices for countries in West Africa

# Data download: http://www.fao.org/faostat/en/#data/PP
# countries: Benin, Burkina Faso, Cameroon, Cote d'Ivoire, Gambia, Ghana, 
# Guinea, Guinea-Bissau, Mali, Mauritania, Niger, Nigeria, Senegal,
# Sierra Leone, Togo 
# crops: Rice, paddy; Maize
# Variable: Producer Price (USD/tonne), annual values, all years
# "This sub-domain contains data on Agriculture Producer Prices. These are 
# prices received by farmers for primary crops, live animals and livestock 
# primary products as collected at the point of initial sale (prices paid at 
# the farm-gate). Annual data are provided from 1991, while mothly data from 
# January 2010 for 180 country and 212 product."

# We calculate average prices for each country using the given timeseries.
# In the model, we calculate a single producer price per crop, by taking a
# weighted average of the country specific prices, using the country land 
# areas as weight.

DP.VisualizeAndPrepare_ProducerPrices()

# %% Preparing GDHY yield data    

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

DP.ReadAndSave_GDHY("maize_major", lat_min, lon_min, lat_max, lon_max)
DP.ReadAndSave_GDHY("rice_major", lat_min, lon_min, lat_max, lon_max)

# TODO make mask: combine profitable area masks with spei mask (BadCluster.py)
# -> MaskProfitableArea (?)

# %% SPEI
    
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


# TODO calc distance with reduce area to profitable cells! -> PearsonDistSPEI03.txt
DP.CalcPearsonDist()

# TODO kMedoids (BadCluster.py) -> kMedoidsX_PearsonDistSPEI_ProfitableArea.txt for X num cluster

# TODO manual set up of adjacency matrix for 9 cluster -> k9AdjacencyMAtrix.txt

# TODO yield trends (BadCluster.py) -> DetrYieldAvg_kX_ProfitableArea for X in 1, ..., 9

