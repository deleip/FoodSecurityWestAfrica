# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:51:51 2020

@author: leip
"""

# %% IMPORTING NECESSARY PACKAGES 

import numpy as np
import numpy.ma as np_ma
import statsmodels.api as sm
import pickle
import netCDF4
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import random

from ModelCode.PlotMaps import MapValues

# ---------------------- FUNCTIONS FOR ReadingRawData.py ----------------------

# %% 1. SPEI data

def ReadAndSave_SPEI03(lon_min, lon_max, lat_min, lat_max):
    """
    Function to read SPEI and save it in a format useable for the
    later following analysis
    

    Parameters
    ----------
    lon_min : float
        Latitude defining upper border of area.
    lon_max : float
        Longitude defining right border of area
    lat_min : float
        Latitude defining lower border of area.
    lat_max : float
        Longitude defining left border of area..

    Returns
    -------
    None.

    """
    
    # read data
    f = netCDF4.Dataset('RawData/SPEI03.nc') 
                  # opening connection to netCDF file [lon, lat, time, spei]
    time = f.variables['time'][:]   # days since 1900-1-1 (len = 1380)
    lats = f.variables['lat'][:]    # degrees north (len = 360)
    lons = f.variables['lon'][:]    # degrees east (len = 720)
    data = f.variables['spei'][:]   # z-values (shape = [1380, 360, 720]    
    f.close()                       # closing connection to netCDF file
    
    # define longitudes/latitudes corresponding to West Africa
    lons_WA = lons[(lons>=lon_min) & (lons<=lon_max)]      
    lats_WA = lats[(lats>=lat_min) & (lats<=lat_max)]
    
    # reduce SPEI data to region
    data_WA = data[:,((lats>=lat_min) & (lats<=lat_max)).data,:] \
                                 [:,:,((lons>=lon_min) & (lons<=lon_max)).data]
    data_WA.set_fill_value(value = np.nan)   # to get full matrix, we fill  
                                             # masked elements with NAN
                                             # (instead of original filling 
                                             # value 1e30)
    data_WA_filled = data_WA.filled()
    
    # save reduced datasets
    with open("ProcessedData/SPEI03_WA_masked.txt", "wb") as fp:    
        pickle.dump(data_WA, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("ProcessedData/SPEI03_WA_filled.txt", "wb") as fp:    
        pickle.dump(data_WA_filled, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    MapValues(data_WA_filled[-1,:,:], title = "SPEI03 (most recent date)", file = "InputData/Visualization/SPEI03_WA")
        
    # Mask: cells that are masked in every timestep should be excluded in
    # further analysis, therefore we define an object "mask" with value 1 
    # for cells with data for at least one timestep and value 0 for the rest 
    # (ocean)
    mask_WA = np.zeros([len(lats_WA),len(lons_WA)] )
    for t in range(0, len(time)):
       for i in range(0, len(lats_WA)):
           for j in range(0, len(lons_WA)):        
               if data_WA.mask[t, i, j] == False: 
                   mask_WA[i,j] = 1
    with open("ProcessedData/mask_SPEI03_WA.txt", "wb") as fp:    
        pickle.dump(mask_WA, fp)    
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    MapValues(mask_WA, title = "SPEI03 area covered", file = "InputData/Visualization/mask_SPEI03_WA", cmap = False)
    
    return(None)

# %% 2. UN world population prospect (UNWPP) data

def ReadAndSave_UNWPP(WhichRegion = "West Africa",
                      WhichValues = "PopTotal"):
    """
    Function that reads UN World Population Prospect and brings it into format
    used by the model

    Parameters
    ----------
    WhichRegion : str, optional
        Region for which data should be prepared. The default is "West Africa".
    WhichValues : str, optional
        Variable that should be prepared. The default is "PopTotal".

    Returns
    -------
    None.

    """
    # read data 
    PopData = pd.read_csv("RawData/WPP2019_TotalPopulationBySex.csv")   
    
    # get different scenario names
    scenarios = np.array(PopData["Variant"])
    scenarios = scenarios[sorted(np.unique(scenarios, \
                                           return_index = True)[1])]
    scenarioIDs = np.array(PopData["VarID"])
    scenarioIDs = scenarioIDs[sorted(np.unique(scenarioIDs, \
                                               return_index = True)[1])]
    scenarios = scenarios[scenarioIDs < 20]
    
    scenarios = np.array(["".join([x.capitalize() for x in y.split(" ")]) for y in scenarios]) 
    
    # reduce data to Western Africa
    WhichRegion_save =  "".join(WhichRegion.split(" "))
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
        
    with open("InputData/Population/UN_" + WhichValues + "_Prospects_" + \
              WhichRegion_save + ".txt", "wb") as fp:    
        pickle.dump(total_pop, fp)
        pickle.dump(scenarios, fp)
        pickle.dump(np.array(range(1950, 2101)), fp)
    
    return(None)

# %% 3. GPW data

def ReadAndReduce_GPW(lat_min, lon_min, lat_max, lon_max):
    """
    Function to read, prepare and save GPW data

    Parameters
    ----------
    lat_min : float
        Latitude defining lower border of area.
    lon_min : float
        Latitude defining upper border of area.
    lat_max : float
        Longitude defining left border of area.
    lon_max : float
        Longitude defining right border of area.

    Returns
    -------
    None.

    """
    
    f = netCDF4.Dataset("RawData/gpw_v4_population_count" + \
                                                 "_adjusted_rev11_30_min.nc")
    lons = f.variables["longitude"][:]  # degrees east (len = 720)
    lats = f.variables["latitude"][:]   # degrees north (len = 360)
    data = f.variables["UN WPP-Adjusted Population Count, " + \
            "v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes"][:]
    f.close()
    
    # First dimension is raster with following categories:
    # 0) UN WPP-Adjusted Population Count, v4.11 (2000)
    # 1) UN WPP-Adjusted Population Count, v4.11 (2005)
    # 2) UN WPP-Adjusted Population Count, v4.11 (2010)
    # 3) UN WPP-Adjusted Population Count, v4.11 (2015)
    # 4) UN WPP-Adjusted Population Count, v4.11 (2020)
    # 5) Data Quality Indicators, v4.11 (2010): Data Context
    # 6) Data Quality Indicators, v4.11 (2010): Mean Administrative Unit Area
    # 7) Data Quality Indicators, v4.11 (2010): Water Mask
    # 8) Land and Water Area, v4.11 (2010): Land Area
    # 9) Land and Water Area, v4.11 (2010): Water Area
    # 10) National Identifier Grid, v4.11 (2010): National Identifier Grid
    # 11) National Identifier Grid, v4.11 (2010): Data Code
    # 12) National Identifier Grid, v4.11 (2010): Input Data Year
    # 13) National Identifier Grid, v4.11 (2010): Input Data Level
    # 14) National Identifier Grid, v4.11 (2010): Input Sex Data Level
    # 15) National Identifier Grid, v4.11 (2010): Input Age Data Level
    # 16) National Identifier Grid, v4.11 (2010): Growth Rate Start Year
    # 17) National Identifier Grid, v4.11 (2010): Growth Rate End Year
    # 18) National Identifier Grid, v4.11 (2010): Growth Rate Administ. Level
    # 19) National Identifier Grid, v4.11 (2010): Year of Most Recent Census
    
    years = np.array([2000, 2005, 2010, 2015, 2020])
    
    # latitudes are ordered the other way round
    data = np.flip(data, axis = 1)
    lats = np.flip(lats)

    # reduce data to relevant region and relevant categories
    data_rel = data[0:5, ((lats>=lat_min) & (lats<=lat_max)).data,:] \
                                 [:,:,((lons>=lon_min) & (lons<=lon_max)).data]
    land_area = data[8, ((lats>=lat_min) & (lats<=lat_max)).data,:] \
                                 [:,((lons>=lon_min) & (lons<=lon_max)).data]
    country_codes = data[10, ((lats>=lat_min) & (lats<=lat_max)).data,:] \
                                 [:,((lons>=lon_min) & (lons<=lon_max)).data]
    
    # fill missing values wih NANs
    data_rel.set_fill_value(value = np.nan)
    data_rel_filled = data_rel.filled()
    land_area.set_fill_value(value = np.nan)
    land_area_filled = land_area.filled()
    country_codes.set_fill_value(value = np.nan)
    country_codes_filled = country_codes.filled()
    country_codes_filled[country_codes_filled == 32767] = np.nan
    
    
    with open("InputData/Population/GPW_WA.txt", "wb") as fp:    
        pickle.dump(data_rel_filled, fp)
        pickle.dump(years, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    with open("InputData/Population/land_area.txt", "wb") as fp:    
        pickle.dump(land_area_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    with open("InputData/Prices/CountryCodesGridded.txt", "wb") as fp:    
        pickle.dump(country_codes_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
        
    return(None)


# %% 4. GDHY data

def ReadAndSave_GDHY(v_name, lon_min, lon_max, lat_min, lat_max):
    """
    Function to read and save GDHY data.

    Parameters
    ----------
    v_name : str
        Name of crop for which yields should be read.
    lon_min : float
        Latitude defining upper border of area.
    lon_max : float
        Longitude defining right border of area
    lat_min : float
        Latitude defining lower border of area.
    lat_max : float
        Longitude defining left border of area..

    Returns
    ------
    None.

    """
    
    def __Read_GDHY(v_name, year, lon_min, lon_max, lat_min, lat_max):
        f = netCDF4.Dataset("RawData/yields/" + v_name + "/yield_" + str(year) + ".nc4")
        lats = f.variables['lat'][:]    # degrees north (360)  
        lons = f.variables['lon'][:]    # degrees east (720)
        var = f.variables["var"][:]     # yield in t/ha (360,720)     
        f.close() 
        
        # rearrange in right format    
        var_lonpos = var[:,lons <180]
        var_lonneg = var[:,lons>180]
        var_changed = np_ma.column_stack([var_lonneg, var_lonpos])
        lons = np.arange(-179.75, 180, 0.5) 
        
        # define longitudes/latitudes corresponding to West Africa
        lons_WA = lons[(lons>=lon_min) & (lons<=lon_max)]  
        lats_WA = lats[(lats>=lat_min) & (lats<=lat_max)]
        
        # reduce to region of West Africa and set missing values to NAN
        # shifted by one cell in both directions
        var_rel = var_changed[((lats>=(lat_min+0.5)) & (lats<=(lat_max+0.5))),:] \
                            [:,((lons>=(lon_min+0.5)) & (lons<=(lon_max+0.5)))]
        var_rel.set_fill_value(value = np.nan)
        var_rel_filled = var_rel.filled()    
        mask_rel = var_rel.mask
        
        return(var_rel_filled, mask_rel, lats_WA, lons_WA)
    
    # yields are saved per year, so we go through all years and save them in 
    # one array
    for year in range(1981, 2017):
        var_year, mask_year, lats_WA, lons_WA = __Read_GDHY(v_name, year, \
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
                                      
                                                     
    with open("ProcessedData/" + v_name.split("_")[0] + "_yld.txt", "wb") as fp:    
        pickle.dump(var_save, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("ProcessedData/" + v_name.split("_")[0] + "_mask.txt", "wb") as fp:    
        pickle.dump(mask_save, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    return(None)


# %% 5. Producer prices

def VisualizeAndPrepare_ProducerPrices():
    """
    Function reading producer prices for countries in West Africa, viasualizes 
    and saes in format used in input data preparation.

    Returns
    -------
    None.

    """
    
    # reading raw price data and 
    prices_raw = pd.read_csv("RawData/FAOSTAT_data_6-11-2020.csv")
    prices_raw = prices_raw[["Area","Item", "Year","Value"]]
    countries = np.array(prices_raw["Area"])
    countries = countries[sorted(np.unique(countries, return_index = True)[1])]
    items = np.array(prices_raw["Item"])
    items = items[sorted(np.unique(items, return_index = True)[1])]
    years = np.array(prices_raw["Year"])
    years = years[sorted(np.unique(years, return_index = True)[1])]
    
    # visualize raw data
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
    fig.savefig("InputData/Visualization/ProducerPrices.png", \
                                bbox_inches = "tight", pad_inches = 0.5)
    plt.close()
     
    # write prices into 3d array
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
            
    # calucalte regional average
    prices = np.nanmean(prices, axis = 0)
    
    # visualize resulting average prices
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
    fig.savefig("InputData/Visualization/ProducerPrices_CountryAvg.png", \
                                bbox_inches = "tight", pad_inches = 0.5)
    plt.close()
    
    # make data frame out of results
    prices = pd.DataFrame(prices)
    prices.insert(0, 'Countries', countries)
    prices.columns = ["Countries", "Maize", "Rice"]
    
    # save resulting prices
    with open("InputData/Prices/CountryAvgFarmGatePrices.txt", "wb") as fp:    
        pickle.dump(prices, fp)

    return(None)


# ----------------- FUNCTIONS USED IN InputDataCalculations.py ----------------

# %% 1. Calculates profitable areas based on yield data, cultivation costs, prices

def ProfitableAreas():
    """
    Function that creates a mask showing for which cells at least one of the 
    crops has profitable yields (according to cellwise regression evaluated for
    the baseyear 2016)
    """
    
    # function for cellwise linear regression
    def __DetrendDataLinear(data, mask):
        # initializing arrays
        data_detrend = data.copy()
        p_val_slopes = np.zeros(np.shape(mask))
        slopes = np.zeros(np.shape(mask))
        intercepts = np.zeros(np.shape(mask))
        
        # detrending each cell seperately
        [num_lat, num_lon] = mask.shape
        for lat in range(0, num_lat):
            for lon in range(0, num_lon):    
                # if data is masked, set NAN as results
                if mask[lat, lon] == 0:
                    p_val_slopes[lat, lon] = np.nan
                    slopes[lat, lon] = np.nan
                    intercepts[lat, lon] = np.nan
                    continue     
                Y = data[:, lat, lon]
                X = np.arange(0, len(Y)).reshape((-1, 1))
                X = sm.add_constant(X)
                model = sm.OLS(Y, X, missing='drop').fit()
                trend = X.dot(model.params)
                data_detrend[:, lat, lon] = data_detrend[:, lat, lon] - trend
                p_val_slopes[lat, lon] = model.pvalues[1]
                slopes[lat, lon] = model.params[1]
                intercepts[lat, lon] = model.params[0]
                
        return(data_detrend, p_val_slopes, slopes, intercepts)  
    
    # inout data
    with open("ProcessedData/rice_yld.txt", "rb") as fp:    
        rice_yld = pickle.load(fp)
    with open("ProcessedData/maize_yld.txt", "rb") as fp:    
        maize_yld = pickle.load(fp)
    
    # masks
    with open("ProcessedData/rice_mask.txt", "rb") as fp:    
        rice_mask = pickle.load(fp)
    with open("ProcessedData/maize_mask.txt", "rb") as fp:    
        maize_mask = pickle.load(fp)
     
    # index of year 2016 (as 1981 is the first for which we have yield data)
    year_rel = (2017 - 1) - 1981
    
    # get expected yields per cell for 2016
    data_detrend, p_val_slopes, slopes, intercepts = __DetrendDataLinear(rice_yld, rice_mask)
    exp_yld_rice = intercepts + year_rel * slopes
    data_detrend, p_val_slopes, slopes, intercepts = __DetrendDataLinear(maize_yld, maize_mask)
    exp_yld_maize = intercepts + year_rel * slopes
    
    # fig = plt.figure(figsize = (24, 13.5))
    # ax = fig.add_subplot(1,2,1)
    # c = OF2.MapValues(exp_yld_rice, lats_WA, lons_WA, vmin = 0, vmax = 4, ax = ax, title = "rice")
    # ax = fig.add_subplot(1,2,2)
    # c = OF2.MapValues(exp_yld_maize, lats_WA, lons_WA, vmin = 0, vmax = 4, ax = ax, title = "maize")
    # cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    # cbar = fig.colorbar(c, cax = cb_ax)     
    # fig.savefig("../ForPublication/Clustering/Area/exp_ylds.png", bbox_inches = "tight", pad_inches = 1)
    
    
    # calculate threshold
    prices = CalcAvgProducerPrices(rice_mask, maize_mask)
    with open("InputData/Other/CultivationCosts.txt", "rb") as fp:
        costs = pickle.load(fp)
    threshold = costs/prices
    
    # cells where both crops are not profitable (overage) are excluded
    rice_expProfitable = exp_yld_rice.copy()
    maize_expProfitable = exp_yld_maize.copy() 
    rice_expProfitable[~(exp_yld_rice > threshold[0]) & ~(exp_yld_maize > threshold[1])] = 0
    rice_expProfitable[rice_expProfitable > 0] = 1
    rice_expProfitable[np.isnan(rice_expProfitable)] = 0
    maize_expProfitable[~(exp_yld_maize > threshold[1]) & ~(exp_yld_rice > threshold[0])] = 0
    maize_expProfitable[maize_expProfitable > 0] = 1
    maize_expProfitable[np.isnan(maize_expProfitable)] = 0
    
    
    # fig = plt.figure(figsize = figsize)
    # ax = fig.add_subplot(1,2,1)
    # c = OF2.MapValues(rice_expProfitable, lats_WA, lons_WA, vmin = 0, vmax = 4, ax = ax, title = "rice")
    # ax = fig.add_subplot(1,2,2)
    # c = OF2.MapValues(maize_expProfitable, lats_WA, lons_WA, vmin = 0, vmax = 4, ax = ax, title = "maize")
    # cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    # cbar = fig.colorbar(c, cax = cb_ax)     
    # cbar.set_ticks([1.87144197, 1.05603763, 0, 1.5, 3])   
    # cbar.set_ticklabels(["rice", "maize", str(0), str(1.5), str(3)])
    # fig.savefig("../ForPublication/Clustering/Area/exp_ylds_profitable.png", bbox_inches = "tight", pad_inches = 1)
    
    mask_profitable = rice_expProfitable + maize_expProfitable
    mask_profitable[mask_profitable > 1] = 1

    with open("ProcessedData/MaskProfitableArea.txt", "wb") as fp:    
        pickle.dump(mask_profitable, fp)

    return(None)


# %% 2. Calculates average producer prices (weighted with land area per country)

def CalcAvgProducerPrices(rice_mask, maize_mask):
    """
    Function calculates average producer prices, by averaging the given 
    country values using using the country land areas as weight. The country
    prices values are based on FAO timeseries (see
    VisualizeAndPrepare_ProducerPrices).

    Parameters
    ----------
    rice_mask : np.array
        Specifiying which cells have rice yield data.
    maize_mask : np.array
        Specifiying which cells have maize yield data.

    Returns
    -------
    None.

    """
    
    ## Country shares of area for price calculation
    with open("InputData/Prices/CountryCodesGridded.txt", "rb") as fp:    
        country_codes_gridded = pickle.load(fp)
    country_codes_gridded_rice = country_codes_gridded.copy()
    country_codes_gridded_maize = country_codes_gridded.copy()
    country_codes_gridded_rice[rice_mask == 0] = np.nan
    country_codes_gridded_maize[maize_mask == 0] = np.nan
        
    country_codes = pd.read_csv("InputData/Prices/CountryCodes.csv")    
    
    total_land_cells_rice = np.sum(rice_mask == 1)
    total_land_cells_maize = np.sum(maize_mask == 1)
    
     # Liberia has no price data
    liberia_landcells_rice = np.sum(country_codes_gridded_rice == country_codes.loc \
                       [(country_codes.CountryName=="Liberia")]["Code"].values)
    liberia_landcells_maize = np.sum(country_codes_gridded_maize == country_codes.loc \
                       [(country_codes.CountryName=="Liberia")]["Code"].values)
    total_cells_for_rice = total_land_cells_rice - liberia_landcells_rice 
    total_cells_for_maize = total_land_cells_maize - liberia_landcells_maize 
                    
    # Mauritania has no rice price data
    mauritania_landcells = np.sum(country_codes_gridded_rice == country_codes.loc \
                    [(country_codes.CountryName=="Mauritania")]["Code"].values)
    total_cells_for_rice = total_cells_for_rice - mauritania_landcells 
                            
    country_codes["Shares rice"] = 0
    country_codes["Shares maize"] = 0
    
    # calculate shares for rice
    country_codes_gridded_rice[country_codes_gridded_rice == country_codes.loc \
             [(country_codes.CountryName=="Liberia")]["Code"].values] = np.nan
    country_codes_gridded_rice[country_codes_gridded_rice == country_codes.loc \
             [(country_codes.CountryName=="Mauritania")]["Code"].values] = np.nan
    for idx, c in enumerate(country_codes["Code"].values):
        country_codes.iloc[idx, 2] = \
                    np.sum(country_codes_gridded_rice == c)/total_cells_for_rice
    
    # calculate shares for maize
    country_codes_gridded_maize[country_codes_gridded_maize == country_codes.loc \
             [(country_codes.CountryName=="Liberia")]["Code"].values] = np.nan
    for idx, c in enumerate(country_codes["Code"].values):
        country_codes.iloc[idx, 3] = \
            np.sum(country_codes_gridded_maize == c)/total_cells_for_maize
            
    # removing countries with no share as they don't show up in prices df below
    country_codes = country_codes.drop(axis = 0, labels = [4, 5, 9])         

    ## apply shares to farm gat prices
    with open("InputData/Prices/CountryAvgFarmGatePrices.txt", "rb") as fp:    
        country_avg_prices = pickle.load(fp)
        
    # Gambia is not included in our area
    country_avg_prices = country_avg_prices.drop(axis = 0, labels = [4])             
    # weighted average (using area share as weight)    
    price_maize = np.nansum(country_avg_prices["Maize"].values * \
                                            country_codes["Shares maize"].values)
    price_rice = np.nansum(country_avg_prices["Rice"].values * \
                          country_codes["Shares rice"].values)
    
    prices = np.array([price_rice, price_maize])
    
    # in 10^9$/10^6t
    prices = 1e-3 * prices 

    return(prices)

# %% 3. Average caloric demand (with area as weight)

def AvgCaloricDemand():
    """
    Calculates average caloric demand per person per day based on country 
    values (weighted with areas).

    Returns
    -------
    None.

    """
    
    # read data
    with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
        mask_profitable = pickle.load(fp)                         

    with open("InputData/Prices/CountryCodesGridded.txt", "rb") as fp:    
        country_codes_gridded = pickle.load(fp)
    country_codes_gridded[mask_profitable == 0] = np.nan
        
    country_codes = pd.read_csv("InputData/Prices/CountryCodes.csv")    
    
    CaloricDemand = pd.read_csv("ProcessedData/CountryCaloricDemand.csv")  

    # calculate average
    CaloricDemand["LandCells"] = 0
    for c in CaloricDemand["Country"]:
        code = country_codes.loc[country_codes["CountryName"] == c, "Code"].values
        CaloricDemand.loc[CaloricDemand["Country"] == c, "LandCells"] = \
            np.sum(country_codes_gridded == code)
    CaloricDemand["Shares"] = CaloricDemand["LandCells"] / np.sum(CaloricDemand["LandCells"])        
    avgCaloricDemand = np.sum(CaloricDemand["Demand"] * CaloricDemand["Shares"])
    
    # save resulting value
    with open("InputData/Other/AvgCaloricDemand.txt", "wb") as fp:
        pickle.dump(avgCaloricDemand, fp)
    
    return(None)

# %% 4. Calculates pearson distance (based on SPEI) between cells

def CalcPearsonDist(mask):
    """
    - Calculate Pearson Correlation between all grid cells using a given time 
    window, saved in form [lat cell 1][lon cell 1][lat cell 2, lon cell 2]
    - Calculate Pearson distance out of given correlation:
        d(x,y) = sqrt(0.5*(1-corr(x,y))) 

    Returns
    -------
    None.

    """
    
    with open("ProcessedData/spei03_WA_filled.txt", "rb") as fp:
        data = pickle.load(fp)   
    
    corr = []
    dist = []
    [n_t, num_lat, num_lon] = data.shape
    
    # setting timewindow
    tw_start = 0
    tw_end = n_t
    
    #going thorugh all cell combinations
    for lat1 in range(0, num_lat):
        corr_lon = []
        dist_lon = []
        for lon1 in range(0, num_lon):
            corr_tmp = np.empty([num_lat, num_lon])
            corr_tmp.fill(np.nan)
            dist_tmp = np.empty([num_lat, num_lon])
            dist_tmp.fill(np.nan)
            if mask[lat1, lon1] == 0:
                corr_lon.append(corr_tmp) # no corr for ocean
                dist_lon.append(dist_tmp) # no distance for ocean
                continue
            X = data[tw_start:tw_end, lat1, lon1]   # get timeseries of cell 1
            for lat2 in range(0, num_lat):
                for lon2 in range(0, num_lon):
                    if mask[lat2, lon2] == 0:
                        continue       # no corr for ocean
                    Y = data[tw_start:tw_end, lat2, lon2]  # timeseries cell 2
                    
                    use = np.logical_and(~np.isnan(X), ~np.isnan(Y)) 
                    if np.sum(use) > 1:
                        corr_tmp[lat2,lon2] = stats.pearsonr(X[use], \
                                                                 Y[use])[0]
                        dist_tmp[lat2,lon2] = np.sqrt(0.5*(1 - \
                                                      corr_tmp[lat2,lon2]))
            corr_lon.append(corr_tmp)
            dist_lon.append(dist_tmp)
        corr.append(corr_lon)    
        dist.append(dist_lon)
        
    # saving results
    with open("InputData/Other/PearsonCorrSPEI03.txt", "wb") as fp:    
        pickle.dump(corr, fp)    
    with open("InputData/Other/PearsonDistSPEI03.txt", "wb") as fp:    
        pickle.dump(dist, fp)    
        
    return(None)    

# %% 5. Functions to find best number of clusters

def MedoidMedoidDistd(medoids, dist):
    """
    Calculates the distance (based on similarity of SPEI) between each pair
    of medoids.

    Parameters
    ----------
    medoids : list
        List giving indices of the cluster medoids.
    dist : np.array
        Giving distances between any two grid cells (based on SPEI).

    Returns
    -------
    None.

    """
    k = len(medoids)
    res = np.empty([k,k])
    res.fill(np.nan)
    # get distance to all medoids
    for i in range(0, k):
        for j in range(0, k):
            if i == j:
                continue
            res[i, j] = dist[medoids[i][0]][medoids[i][1]] \
                                                [medoids[j][0], medoids[j][1]]
    res_closest = np.zeros(k)
    # find closest medoid
    for i in range(0, k):
        res_closest[i] = np.nanmin(res[i])
    return(res, res_closest)

def MetricClustering(dist_within, dist_between, refX = 0, refY = 1):
    """
    Orders the clustering into differnet numbers of clusters based on their
    within cluster similarities and between cluster similarities. Optimal 
    would be distance_within = 0 (complete similarity) and distance_between = 1
    (complete dissimilarity). The metric qunatifies the quality of a clustering
    by calculating the Eudlidean distance between 
    (distance_within, distance_between) and the optimal point (0, 1), or if 
    specified, a different reference point (as the optimal value can never
    be reached in a real world).

    Parameters
    ----------
    dist_within : list
        Value descrinding the distance within clusters for different number of
        clusters.
    dist_between : list
        Value descrinding the distance between clusters for different number of
        clusters.
    refX : float, optional
        Reference value for distance within clusters. The default is 0.
    refY : floart, optional
        Reference value for distance between clusters. The default is 1.

    Returns
    -------
    m : np.array
        resulting metric values, ordered from best to worst.
    cl_order : np.array
        Tells number of clusters from best to worst.
    

    """
    # euclidean distance
    m = np.sqrt(((np.array(dist_within)-refX)**2) + \
                ((refY- np.array(dist_between))**2))
    # orderdifferent K accordin to performance
    order = np.argsort(m)
    # as the lowest number of clusters is 2 and ordering gives the index 
    # (starting with 0) we need to shift values
    cl_order = order + 2
    # sort results metric according to performance
    m = m[order]
    
    return(m, cl_order)

# %% 6. Calculate yield trends within clusters

def YldTrendsCluster(k):
    """
    Uses gridded yield data to calculate timeseries of average yields per 
    cluster. Then, a linear regression is calculated for these yield timeseries,
    and the standard deviation of the residuals is calculated. The results are
    saved.

    Parameters
    ----------
    k : int
        Number of clusters.

    Returns
    -------
    None.

    """
    
    # sub-function to calculate timeseries of average yields per cluster (call
    # for one crop at a time)
    def __ClusterAverage(data, cl, k):
        n_t = data.shape[0]
        res = np.empty([n_t, k])
        res.fill(np.nan)
        for t in range(0, n_t):
            for i in range(0, k):
                res[t, i] = np.nanmean(data[t, (cl == i + 1)])
        return(res) 

    # sub-function to calculate linear trends for each cluster-timeseries, 
    # and returns for each cluster:
    # regression values, residuals (ie.e. observations - regression vlaues),
    # the mean of the residuals (should be close to zero), standard 
    # deviation of the residuals, f statistics of the regression, regression
    # coefficients constant value and slope
    def __DetrendClusterAvgYlds(yields_avg, k, crops):
        len_ts = yields_avg.shape[0]
        # initializing results 
        avg_pred = np.empty([len_ts, len(crops), k]); avg_pred.fill(np.nan)
        residuals = np.empty([len_ts, len(crops), k]); residuals.fill(np.nan)
        residual_means = np.empty([len(crops), k]); residual_means.fill(np.nan)
        residual_stds = np.empty([len(crops), k]); residual_stds.fill(np.nan)
        fstat = np.empty([len(crops), k]); fstat.fill(np.nan)
        slopes = np.empty([len(crops), k]); slopes.fill(np.nan)
        constants = np.empty([len(crops), k]); constants.fill(np.nan)
        # detrending per cluster and crop
        for cr in range(0, len(crops)):
            for cl in range(0, k):
                # timeseries
                X = np.arange(0, len_ts).reshape((-1, 1))
                X = sm.add_constant(X)
                Y = yields_avg[:,cr,cl]
                if np.sum(~(np.isnan(Y))) > 0:
                    # regression
                    model = sm.OLS(Y, X, missing='drop')
                    result = model.fit()
                    # saving results
                    avg_pred[:,cr,cl] = result.predict(X)   
                    residuals[:,cr,cl] = Y - avg_pred[:,cr,cl]
                    residual_means[cr, cl] = np.nanmean(residuals[:,cr,cl])
                    residual_stds[cr, cl] = np.nanstd(residuals[:,cr,cl])
                    fstat[cr, cl] = result.f_pvalue
                    constants[cr, cl] = result.params[0]
                    slopes[cr, cl] = result.params[1] 
        return(avg_pred, residuals, residual_means, residual_stds, fstat, \
               constants, slopes)

    # load clustering for given k
    with open("InputData/Clusters/Clustering/kMediods" + \
                 str(k) + "_PearsonDistSPEI.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        
    # load yield data
    crops = ["rice", "maize"]
    yields = []
    for cr in crops:
        with open("ProcessedData/" + cr + "_yld.txt", "rb") as fp:    
            yld_tmp = pickle.load(fp)
        yields.append(yld_tmp)
        
    # get average yields per cluster
    num_years = yields[0].shape[0]
    yields_avg = np.empty([num_years, len(crops), k])
    yields_avg.fill(np.nan)
    
    for cr in range(0, len(crops)):
        yields_avg[:, cr, :] = __ClusterAverage(yields[cr], clusters, k)
                                               
    with open("ProcessedData/YieldAverages_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(yields_avg, fp)
        pickle.dump(crops, fp)
    
    # detrend cluster average yield timeseries
    avg_pred, residuals, residual_means, residual_stds, fstat, constants,\
        slopes = __DetrendClusterAvgYlds(yields_avg, k, crops)
        
    # save results
    years = np.array(range(1981, 2017))
    with open("InputData/YieldTrends/DetrYieldAvg_k" + \
                              str(k) + ".txt", "wb") as fp:    
        pickle.dump(yields_avg, fp)
        pickle.dump(avg_pred, fp)
        pickle.dump(residuals, fp)
        pickle.dump(residual_means, fp)
        pickle.dump(residual_stds, fp)
        pickle.dump(fstat, fp)
        pickle.dump(constants, fp)
        pickle.dump(slopes, fp)
        pickle.dump(crops, fp)
        pickle.dump(years, fp)
         
    return(None)

# --------------------------- k-Medoids Algorithm -----------------------------


# Definition of the k-Medoids algorithm:
# Step 1. k different objects are chosen as initial medoids by a greedy 
#         algorithm 
# Step 2. Each remaining object is associated with the medoid that is closest. 
#         The total costs are calculated as the sum over all squared distances 
#         between object and respective medoid.
# Step 3. For each pair of object and medoid, it is checked whether switching 
#         them (i.e. the normal object becoming medoid) would improve the 
#         clustering (i.e. decrease the total costs). After going through all
#         pairs, the switch yielding the biggest improvement is performed 
#         and step 3 is repeated. If none of the switches would yield an 
#         improvement the algorithm terminates.
    
# 0) Main part
def kMedoids(k, dist, mask, file, version = "", start_medoids = None, \
                      term = True, max_its = np.inf, seed = 3052020):
    """
    Definition of the k-Medoids algorithm:
    Step 1. k different objects are chosen as initial medoids by a greedy 
            algorithm 
    Step 2. Each remaining object is associated with the medoid that is closest. 
            The total costs are calculated as the sum over all squared distances 
            between object and respective medoid.
    Step 3. For each pair of object and medoid, it is checked whether switching 
            them (i.e. the normal object becoming medoid) would improve the 
            clustering (i.e. decrease the total costs). After going through all
            pairs, the switch yielding the biggest improvement is performed 
            and step 3 is repeated. If none of the switches would yield an 
            improvement the algorithm terminates.

    Parameters
    ----------
    k : int
        Number of clusters in which the full data is to be divided.
    dist : np.array
        Distance between any two locations, on which the clustering is based.
    mask : np.array
        Specifiying which cells are to be considered in clustering.
    file : str
        Name for saving results.
    version : str, optional
        Version name to be specified when saving results. The default is "".
    start_medoids : list, optional
        List of indices describing the start medoids. The default is None.
    term : booleand, optional
        Whether algorithm should run until it finds optimal solution (True) 
        or whether it should stop based on a different termination criterium, 
        i.e. maximum number of iterations (False). The default is True.
    max_its : int, optional
        Maximum number of iterations that should be tried before terminating
        the algorithm. The default is np.inf.
    seed : int, optional
        Seed to be set before getting intial medoids (to make the results
        reproducable). The default is 3052020.

    Returns
    -------
    None.

    """
    # initializing variables
    [num_lats, num_lons] = mask.shape
    terminated = False
    step = 0
    # Normally, the initial medoids are chosen thorugh a greedy algorithm, 
    # but if we wish to continue a run that had not yet terminated or for 
    # some other reason want to start with specific medoids these can be 
    # given to the function and will be used
    random.seed(seed) 
    if start_medoids == None:
        medoids = GetInitialMedoids(k, dist, mask, num_lats, num_lons)
    else:
        medoids = start_medoids
    # get best cluster for these medoids and save the results of this step
    cluster, cost = GetCluster(k, dist, num_lats, num_lons, medoids, mask)

    # Now we iterated until either the maximal number of iteration is reached 
    # or no improvment can be found
    while terminated == False:
        # going trough all possible switches and finding the best one
        new_cluster, new_cost, new_medoids = GetSwitch(k, dist, num_lats, \
                                                       num_lons, medoids, mask)
        # if the best possible switch actually improves the clusters we do the
        # switch and save the outcome of this step
        if new_cost < cost:
            cost, cluster, medoids = new_cost, new_cluster, new_medoids
            step += 1
            # checking if maximal number of iterations is reached
            if term == False and step > max_its:
                terminated = True                           
            continue
        # if best switch is not improving the clustering, we print a 
        # corresponding message and terminate the algorithm
        print("No improvement found")
        terminated = True
    # save results
    if seed == 3052020:
        title_seed = ""
    else:
        title_seed = "_seed" + str(seed)
        
    # shift cluster to go from 1 to k (instead of 0 to k-1)    
    cluster = cluster + 1    
    
    with open("InputData/Clusters/Clustering/" + version + \
                              "kMediods" + str(k) + "_" + file + \
                              title_seed + ".txt", "wb") as fp:    
        pickle.dump(cluster, fp)
        pickle.dump(cost, fp)
        pickle.dump(medoids, fp)
        
    return()
    
# 1) Greedy algorithm for initial medoids
def GetInitialMedoids(k, dist, mask, num_lats, num_lons):
    medoids = []
    # for each medoid take the cell the is the is the best choice at this point
    # given the other mediods that are already chosen. 
    for l in range(1, k+1):
        best_cost = 0
        # for each new medoid we check each possible cell 
        for i in range(0, num_lats):
            for j in range(0, num_lons):
                # ocean is not considered in the clustering
                if (mask[i, j] == 0) or ((i,j) in medoids):
                    continue
                # temporarily add this cell to medoids
                medoids_tmp = medoids.copy()
                medoids_tmp.append((i,j))
                # get new costs
                cluster, cost = \
                    GetCluster(l, dist, num_lats, num_lons, medoids_tmp, mask)
                # if best choice so far (or first cell checked) remember 
                if best_cost == 0:
                    best_cost = cost
                    best_medoid = (i,j)
                elif cost < best_cost:
                    best_cost = cost
                    best_medoid = (i,j)
        # add best choice to medoids
        medoids.append(best_medoid)
    return(medoids)

# 2) Subroutine to get clusters to given medoids:
def GetCluster(k, dist, num_lats, num_lons, medoids, mask):
    cluster = np.empty([num_lats, num_lons])
    cluster.fill(np.nan)
    cl_dist = np.empty([num_lats, num_lons])
    cl_dist.fill(np.nan)
    # loop over all grid cells
    for i in range(0, num_lats):
        for j in range(0, num_lons):
            # ocean is not considered in the clustering
            if mask[i, j] == 0:
                continue
            # we index cluster by the position of the respective medoid
            # a medoid obviously belongs to its own cluster
            if (i,j) in medoids:
                cluster[i,j] = medoids.index((i,j))
                cl_dist[i,j] = 0
                continue
            # initilizing the best distance with 2,  as 1 is the worst possible 
            # distance and we want something worse
            best_dist = 2
            # We check for each medoid how big the distance of the current grid
            # cell to that medoid is. If we found a better distance than the 
            # current best_dist we update it and remember the medoid index
            for [k,l] in medoids:
                dist_tmp = dist[i][j][k,l]
                if np.isnan(dist_tmp):
                    print("aeg", flush = True)
                if dist_tmp < best_dist:
                    best_dist = dist_tmp
                    best_med = medoids.index((k,l))
            # we then save the best distance and the corresponding cluster
            cluster[i,j] = best_med
            cl_dist[i,j] = best_dist
    # calculating the cost function: sum of all squared distances
    cost = np.nansum(cl_dist**2)
    return(cluster, cost)
                
# 3) Subroutine to get best change in medoids
def GetSwitch(k, dist, num_lats, num_lons, medoids, mask):
    new_cost = -1
    # loop over all grid cells
    for i in range(0, num_lats):
        for j in range(0, num_lons):
            # if the grid cell is ocean we don't want to switch as the ocean 
            # is a seperate region
            if mask[i, j] == 0:
                continue
            # if the grid cell already is a cluster a switch makes no sense
            if (i,j) in medoids:
                continue
            # for each of the medoids we check what a switch would result in
            for [k,l] in medoids:
                # switching the medoids
                medoids_tmp = medoids[:]
                medoids_tmp[medoids_tmp.index((k,l))] = (i,j)
                # getting the new cluster
                cluster_tmp, cost_tmp = GetCluster(k, dist, num_lats, \
                                                   num_lons, medoids_tmp, mask)
                # updating if we found a better switch (or if this was the 
                # first we tried)
                if cost_tmp < new_cost or new_cost == -1:
                    new_cluster = cluster_tmp
                    new_cost = cost_tmp
                    new_medoids = medoids_tmp
    # returning best switch found (even if it is no improvement to current 
    # situation - this is checked after)
    return(new_cluster, new_cost, new_medoids)




# --------------------------------- deprected --------------------------------



# def MapValues(values, lat_min, lat_max, lon_min, lon_max, lons_rel, title = "", vmin = None, vmax = None):
    
    
    # extent = [lon_min, lon_max, lat_min, lat_max]
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent(extent)

    # # lons = np.arange(lon_min, lon_max, 0.5)
    # # lats = np.arange(lat_min, lat_max, 0.5)
    # # lon2d, lat2d = np.meshgrid(lons, lats)
    
    # with open("InputData/Other/LatsLonsArea.txt", "rb") as fp:
    #     lats_WA = pickle.load(fp)
    #     lons_WA = pickle.load(fp)
        
    
    # plt.pcolormesh(lons_WA, lats_WA, clusters, shading='flat', transfoom = ccrs.PlateCarree())
    # ax.gridlines(draw_labels=True)
    
#     resol = '50m'  # use data at this scale
#     bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
#         name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
#     land = cartopy.feature.NaturalEarthFeature('physical', 'land', \
#         scale=resol, edgecolor='k', facecolor=cartopy.feature.COLORS['land'])
#     ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
#         scale=resol, edgecolor='none', facecolor=cartopy.feature.COLORS['water'])
#     lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', \
#         scale=resol, edgecolor='b', facecolor=cartopy.feature.COLORS['water'])
#     rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
#         scale=resol, edgecolor='b', facecolor='none')
    
#     ax.add_feature(land, facecolor='beige')
#     ax.add_feature(ocean, linewidth=0.2 )
#     ax.add_feature(lakes)
#     ax.add_feature(rivers, linewidth=0.5)
#     ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)
    
    
#     plt.show()
        

# def MapValues(values, lats_rel, lons_rel, \
#               title = "", vmin = None, vmax = None, ax = None):    
#     # initialize map
#     m = Basemap(llcrnrlon=lons_rel[0], llcrnrlat=lats_rel[0], \
#                 urcrnrlat=lats_rel[-1], urcrnrlon=lons_rel[-1], \
#                 resolution='l', projection='merc', \
#                 lat_0=lats_rel.mean(),lon_0=lons_rel.mean(), ax = ax)
    
#     lon, lat = np.meshgrid(lons_rel, lats_rel)
#     xi, yi = m(lon, lat)
    
#     # Plot Data
#     m.drawmapboundary(fill_color='azure')
#     c = m.pcolormesh(xi,yi,np.squeeze(values), cmap = 'jet_r', \
#                                           vmin = vmin, vmax = vmax)

#     # Add Grid Lines
#     m.drawparallels(np.arange(-80., 81., 10.), labels=[0,1,0,0], fontsize=8)
#     m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=8)
#     # Add Coastlines, States, and Country Boundaries
#     m.drawcoastlines(linewidth=1.1)
#     m.drawstates(linewidth=1.1)
#     m.drawcountries(linewidth=1.1)
#     m.drawrivers(linewidth=0.5, color='blue')
#     # Add Title
#     if ax:
#         ax.set_title(title)
#     else:
#         plt.title(title)
#     plt.show()
#     return(c)



