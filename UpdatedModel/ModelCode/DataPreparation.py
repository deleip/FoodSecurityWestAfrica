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
from scipy.spatial import distance

# %% DEFINING FUNCTIONS

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

# 7) 
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
    
    
    with open("InputData/Population/GPW_WAtest.txt", "wb") as fp:    
        pickle.dump(data_rel_filled, fp)
        pickle.dump(years, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    with open("InputData/Population/land_areatest.txt", "wb") as fp:    
        pickle.dump(land_area_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    with open("InputData/Prices/CountryCodesGriddedtest.txt", "wb") as fp:    
        pickle.dump(country_codes_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
        
    return(None)

def VisualizeAndPrepare_ProducerPrices():
    """
    Function reading producer prices for countries in West Africa, viasualizes 
    and saes in format used by the model.

    Returns
    -------
    None.

    """
    prices_raw = pd.read_csv("RawData/FAOSTAT_data_6-11-2020.csv")
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
    fig.savefig("InputData/Visualization/ProducerPrices.png", \
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
    fig.savefig("InputData/Visualization/ProducerPrices_CountryAvg.png", \
                                bbox_inches = "tight", pad_inches = 0.5)
    
    prices = pd.DataFrame(prices)
    prices.insert(0, 'Countries', countries)
    prices.columns = ["Countries", "Maize", "Rice"]
    
    with open("InputData/Prices/CountryAvgFarmGatePrices.txt", "wb") as fp:    
        pickle.dump(prices, fp)

    return(None)

def ReadAndSave_SPEI03(lon_min, lon_max, lat_min, lat_max):
    """
    Function to read SPEI and save it in a format useable for the
    later following analysis
    

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
    with open("RawData/ProcessedData/" + 
                                          "SPEI03_WA_masked.txt", "wb") as fp:    
        pickle.dump(data_WA, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("RawData/ProcessedData/" + \
                                      "SPEI03_WA_filled.txt", "wb") as fp:    
        pickle.dump(data_WA_filled, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
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
    with open("RawData/ProcessedData/mask_" + \
                                         "SPEI03_WA.txt", "wb") as fp:    
        pickle.dump(mask_WA, fp)    
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    return(None)

def CalcPearsonDist():
    """
    - Calculate Pearson Correlation between all grid cells using a given time 
    window, saved in form [lat cell 1][lon cell 1][lat cell 2, lon cell 2]
    - Calculate Pearson distance out of given correlation:
        d(x,y) = sqrt(0.5*(1-corr(x,y))) 

    Returns
    -------
    None.

    """
    
    with open("RawData/ProcessedData/spei03_WA_filled.txt", "rb") as fp:
        data = pickle.load(fp)   
        
    with open("RawData/ProcessedData/mask_spei03_WA.txt", "rb") as fp:    
        mask = pickle.load(fp)   
    
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
    with open("InputData/Other/PearsonDistSPEI03resr.txt", "wb") as fp:    
        pickle.dump(dist, fp)    
        
    return(None)    


# 4) A function to read GDHY data and change to format useable for the later 
# following analysis    
def ReadAndSave_GDHY(v_name, lon_min, lon_max, lat_min, lat_max):
    
    def _Read_GDHY(v_name, year, lon_min, lon_max, lat_min, lat_max):
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
        var_year, mask_year, lats_WA, lons_WA = _Read_GDHY(v_name, year, \
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
                                                     
    with open("RawData/ProcessedData/" + v_name.split("_")[0] + "_yld.txt", "wb") as fp:    
        pickle.dump(var_save, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("RawData/ProcessedData/" + v_name.split("_")[0] + "_mask.txt", "wb") as fp:    
        pickle.dump(mask_save, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    return(None)



# 2) A function subtracting linear trend (using OLS) per cell from gridded data 
# (using mask to decide which cells to use). Returns gridded detrended data, 
# p-value of the slopes, and the intercepts and slopes of the linear trend
def DetrendDataLinear(data, mask):
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
    
    
# 3) A function to read data from AgMIP models and save them in a format 
# useable for the later following analysis
def ReadAndSave_AgMIP(Model, model, climate, harm_scenario, irri_scenario, \
                      crop, crop_abbrv, year_start, year_end, var_name, \
                      lat_min, lon_min, lat_max, lon_max):
    timewindow = str(year_start) + "_" + str(year_end)
    filename_yld = "/" + Model + "." + crop + "/" + climate + "/" + model + \
                    "_" + climate + "_hist_" + harm_scenario + "_" + \
                    irri_scenario + "_" + var_name + "_" + crop_abbrv + \
                    "_annual_" + timewindow + ".nc4"
    data_type1 = model + "_" + climate + "_" + harm_scenario      
    crop_irr =  crop_abbrv + "_" + irri_scenario   
    
    f = netCDF4.Dataset("Data" + filename_yld)
#    time = f.variables['time'][:]   # growing seasons since 1980-01-01
    lats = f.variables['lat'][:]    # degrees north (360)
    lons = f.variables['lon'][:]    # degrees east (720)
    var = f.variables[var_name + "_" + crop_abbrv][:] # t per ha and yr     
    f.close()         
    
    # turn into form of spei
    lats = np.flip(lats)
    var = np.flip(var, axis = 1)

    # define longitudes/latitudes corresponding to West Africa
    lons_WA = lons[(lons>=lon_min) & (lons<=lon_max)]      
    lats_WA = lats[(lats>=lat_min) & (lats<=lat_max)]
    
    # reduce to region of WA
    var_rel = var[:,((lats>=(lat_min)) & (lats<=(lat_max))).data,:] \
                              [:,:,((lons>=lon_min) & (lons<=lon_max)).data] 
                              
    # fill masked elements with NAN
    var_rel.set_fill_value(value = np.nan)
    var_rel_filled = var_rel.filled()
    
    # save prepared data
    with open("IntermediateResults/PreparedData/AgMIP/" + \
            data_type1 + "_" + var_name + "_" + crop_irr + ".txt", "wb") as fp:    
        pickle.dump(var_rel_filled, fp) 
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    # different AgMIP models have different cells with missing data, so for 
    # each we make a new mask
    [T, num_lats, num_lons] = var_rel.shape
    if (var_name == "yield") & (irri_scenario == "noirr"):
        mask = np.zeros([num_lats, num_lons])
        for t in range(0, T):
           for i in range(0, num_lats):
               for j in range(0, num_lons):        
                   if var_rel.mask[t, i, j] == False: 
                       mask[i,j] = 1
        with open("IntermediateResults/PreparedData/AgMIP/" + \
                                      data_type1 + "_mask.txt", "wb") as fp:    
            pickle.dump(mask, fp)
            pickle.dump(lats_WA, fp)
            pickle.dump(lons_WA, fp)
   
    return()
    

    
# 5) A function to read data from crop calender and change to format useable 
# for the later following analysis     
def ReadAndSave_CropCalendar(crop, lon_min, lon_max, lat_min, lat_max):   
 
    f = netCDF4.Dataset("Data/CropCalendar/" + crop + ".crop.calendar.fill.nc")
    lats = f.variables['latitude'][:]       # degrees north (31)
    lons = f.variables['longitude'][:]      # degrees east (59)
    plant = f.variables['plant'][:]         # day of year
    harvest = f.variables['harvest'][:]     # day of year     

    # rearrange in right format
    plant_flipped = np.flip(plant, axis = 0)
    harvest_flipped = np.flip(harvest, axis = 0)
    lats = np.flip(lats)

    # define longitudes/latitudes corresponding to West Africa
    lons_WA = lons[(lons>=lon_min) & (lons<=lon_max)]      
    lats_WA = lats[(lats>=lat_min) & (lats<=lat_max)]
    
    # reduce to region of West Africa
    plant_rel = plant_flipped[((lats>=(lat_min+0.5)) & (lats<=(lat_max+0.5))).data,:] \
                                   [:,((lons>=lon_min+0.5) & (lons<=lon_max+0.5)).data]   
    harvest_rel = harvest_flipped[((lats>=(lat_min+0.5)) & (lats<=(lat_max+0.5))).\
                            data,:][:,((lons>=lon_min+0.5) & (lons<=lon_max+0.5)).data]   
    mask_plant = plant_rel.mask
    mask_harvest = harvest_rel.mask
    
    
    plant_rel.set_fill_value(value = np.nan)
    plant_rel_filled = plant_rel.filled()    
    harvest_rel.set_fill_value(value = np.nan)
    harvest_rel_filled = harvest_rel.filled()    
    
    # save data
    with open("IntermediateResults/PreparedData/CropCalendar/" + \
                                      crop + "_plant.txt", "wb") as fp:    
        pickle.dump(plant_rel_filled, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("IntermediateResults/PreparedData/CropCalendar/" + \
                                      crop + "_harvest.txt", "wb") as fp:    
        pickle.dump(harvest_rel_filled, fp)    
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("IntermediateResults/PreparedData/CropCalendar/" + \
                                      crop + "_plant_mask.txt", "wb") as fp:    
        pickle.dump(mask_plant, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
    with open("IntermediateResults/PreparedData/CropCalendar/" + \
                                      crop + "_harvest_mask.txt", "wb") as fp:    
        pickle.dump(mask_harvest, fp)
        pickle.dump(lats_WA, fp)
        pickle.dump(lons_WA, fp)
        
    return()
    

# TODO doesn't work on my laptop as array is too big... is it necessary (just
# used for one of the many discarded regression tries)    
def ReadAndAgg_AgMERRA(var, year, lon_min, lon_max, lat_min, lat_max):

    f = netCDF4.Dataset("Data/AgMERRA/" + var + "/AgMERRA_" + \
                                                str(year) + "_" + var + ".nc4")
#    time = f.variables['time'][:]          # days since start of year (366)
    lats = f.variables['latitude'][:]       # degrees north (31)
    lons = f.variables['longitude'][:]      # degrees east (59)
    data = f.variables[var][:]       
    f.close()
    
    # reducing to region of West Africa 
    data_lonpos = data[:,:,lons <180]
    data_lonneg = data[:,:,lons>180]
    data_changed = np_ma.dstack([data_lonneg, data_lonpos])
    data_changed = np.flip(data_changed, axis = 1)
    lons = np.arange(-179.875, 180, 0.25)
    lats = np.flip(lats)
    data_rel = data_changed[:,((lats>=lat_min) & (lats<=lat_max)),:] \
                                    [:,:,((lons>=lon_min) & (lons<=lon_max))]
    
    # aggregating to 0.5 degree resolution
    [n_t, n_lat, n_lon] = data_rel.shape
    data_rel_agg = np.zeros([n_t,int(n_lat/2), int(n_lon/2)])
    for t in range(0, n_t):
        for lat in range(0, int(n_lat/2)):
            for lon in range(0, int(n_lon/2)):
                x = np.nanmin(data_rel[t, (2*lat):(2*lat+2), \
                                                           (2*lon):(2*lon+2)])
                if np.isnan(float(x)):
                    data_rel_agg[t, lat, lon] = np.nan
                else:
                    data_rel_agg[t, lat, lon] = np.nanmean(data_rel[t, \
                                        (2*lat):(2*lat+2), (2*lon):(2*lon+2)])
                     
    return(data_rel_agg)
    
# 6) Function to read, prepare and save CRU data    
def ReadAndAgg_CRU(var, var_abbrv):
    
    # reading and saving data
    f = netCDF4.Dataset("Data/CRU/" + var + ".nc")
#    time = f.variables['time'][:]   # days since 1900-1-1 (1416)
    lats = f.variables['lat'][:]    # degrees north (31)
    lons = f.variables['lon'][:]    # degrees east (59)
    data = f.variables[var_abbrv][:]     # (1416, 31, 59)  
    f.close()
    data.set_fill_value(value = np.nan)
    data_filled = data.filled()  
    with open("IntermediateResults/PreparedData/CRU/" + \
                                          var + "_WA.txt", "wb") as fp:    
        pickle.dump(data_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    
    # creating and savgin mask
    mask_data = np.zeros([len(lats),len(lons)])
    [n_t, n_lat, n_lon] = data.shape
    for t in range(0, n_t):
       for i in range(0, n_lat):
           for j in range(0, n_lon):        
               if data.mask[t, i, j] == False: 
                   mask_data[i,j] = 1
    with open("IntermediateResults/PreparedData/CRU/mask_" + \
                                          var + "_WA.txt", "wb") as fp:    
        pickle.dump(mask_data, fp)   
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)  
        
    # calculating and saving 3 month averages
    data03_WA = np.zeros(data_filled.shape)
    for i in range(0, len(lats)):
        for j in range(0, len(lons)):
            if mask_data[i, j] == 0:
                continue     
            data03_WA[:, i, j] = np.array(pd.DataFrame(data_filled[:, i, j]).\
                            rolling(window=3, center = False).mean()).flatten()
    with open("IntermediateResults/PreparedData/CRU/" + \
                                          var + "03_WA.txt", "wb") as fp:    
        pickle.dump(data03_WA, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    
    # detrendind 3 month averages
    data_detrend, p_val_slopes, slopes, intercepts = \
                                        DetrendDataLinear(data03_WA, mask_data)
    with open("IntermediateResults/PreparedData/CRU/" + \
                                       var + "03_WA_detrend.txt", "wb") as fp:    
        pickle.dump(data_detrend, fp)    
        pickle.dump((p_val_slopes, slopes, intercepts), fp)  
    return()
    
