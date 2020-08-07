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

# %% DEFINING FUNCTIONS

# 1) A function to read SPEI or SPI and save them in a format useable for the
# later following analysis
def ReadAndSave_DroughtIndex(var, lon_min, lon_max, lat_min, lat_max):
    # read data
    f = netCDF4.Dataset('Data/DroughtIndicators/' + var + '.nc') 
                  # opening connection to netCDF file [lon, lat, time, sp(e)i]
    time = f.variables['time'][:]   # days since 1900-1-1 (len = 1380)
    lats = f.variables['lat'][:]    # degrees north (len = 360)
    lons = f.variables['lon'][:]    # degrees east (len = 720)
    if var[0:4] == "spei":
        data = f.variables['spei'][:]   # z-values (shape = [1380, 360, 720]            
    elif var == "spi03":
        data = f.variables['spi3'][:]
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
    with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                      var + "_WA_masked.txt", "wb") as fp:    
        pickle.dump(data_WA, fp, protocol = 2)
        pickle.dump(lats_WA, fp, protocol = 2)
        pickle.dump(lons_WA, fp, protocol = 2)
    with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                      var + "_WA_filled.txt", "wb") as fp:    
        pickle.dump(data_WA_filled, fp, protocol = 2)
        pickle.dump(lats_WA, fp, protocol = 2)
        pickle.dump(lons_WA, fp, protocol = 2)
        
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
    with open("IntermediateResults/PreparedData/DroughtIndicators/mask_" + \
                                          var + "_WA.txt", "wb") as fp:    
        pickle.dump(mask_WA, fp, protocol = 2)    
        pickle.dump(lats_WA, fp, protocol = 2)
        pickle.dump(lons_WA, fp, protocol = 2)
        
    return()

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
    
# 4) A function to read GDHY data and change to format useable for the later 
# following analysis    
def Read_GDHY(var, year, lon_min, lon_max, lat_min, lat_max):
    
    f = netCDF4.Dataset("Data/GDHY_v2_v3/" + var + "/yield_" + \
                                                            str(year) + ".nc4")
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
    
# 5) A function to read data from crop calender and save them in a format 
# useable for the later following analysis     
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
    plant_rel = plant_flipped[((lats>=(lat_min+0.5)) & \
                         (lats<=(lat_max+0.5))).data,:] \
                         [:,((lons>=lon_min+0.5) & (lons<=lon_max+0.5)).data]   
    harvest_rel = harvest_flipped[((lats>=(lat_min+0.5)) & \
                           (lats<=(lat_max+0.5))).data,:] \
                           [:,((lons>=lon_min+0.5) & \
                           (lons<=lon_max+0.5)).data]   
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
    
# 5) A function to read and aggregate data from AgMERRA and change to format 
# useable for the later following analysis     
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
    data_filled = data.filled()  # fill missing values with NAN
    
    # save data
    with open("IntermediateResults/PreparedData/CRU/" + \
                                          var + "_WA.txt", "wb") as fp:    
        pickle.dump(data_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    
    # creating and saving mask
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
    
    # detrendind and saving 3 month averages
    data_detrend, p_val_slopes, slopes, intercepts = \
                                        DetrendDataLinear(data03_WA, mask_data)
    with open("IntermediateResults/PreparedData/CRU/" + \
                                       var + "03_WA_detrend.txt", "wb") as fp:    
        pickle.dump(data_detrend, fp)    
        pickle.dump((p_val_slopes, slopes, intercepts), fp)  
    return()
    
# 7) Function to read, prepare and save GPW data
def ReadAndReduce_GPW(lat_min, lon_min, lat_max, lon_max):
    f = netCDF4.Dataset("Data/Population/gpw_v4_population_count" + \
                                                "_adjusted_rev11_30_min.nc")
    lons = f.variables["longitude"][:]  # degrees east (len = 720)
    lats = f.variables["latitude"][:]   # degrees north (len = 360)
    data = f.variables["UN WPP-Adjusted Population Count, " + \
            "v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes"][:]
    f.close()
    
    # First dimension is raster with following categories:
    # 1) UN WPP-Adjusted Population Count, v4.11 (2000)
    # 2) UN WPP-Adjusted Population Count, v4.11 (2005)
    # 3) UN WPP-Adjusted Population Count, v4.11 (2010)
    # 4) UN WPP-Adjusted Population Count, v4.11 (2015)
    # 5) UN WPP-Adjusted Population Count, v4.11 (2020)
    # 6) Data Quality Indicators, v4.11 (2010): Data Context
    # 7) Data Quality Indicators, v4.11 (2010): Mean Administrative Unit Area
    # 8) Data Quality Indicators, v4.11 (2010): Water Mask
    # 9) Land and Water Area, v4.11 (2010): Land Area
    # 10) Land and Water Area, v4.11 (2010): Water Area
    # 11) National Identifier Grid, v4.11 (2010): National Identifier Grid
    # 12) National Identifier Grid, v4.11 (2010): Data Code
    # 13) National Identifier Grid, v4.11 (2010): Input Data Year
    # 14) National Identifier Grid, v4.11 (2010): Input Data Level
    # 15) National Identifier Grid, v4.11 (2010): Input Sex Data Level
    # 16) National Identifier Grid, v4.11 (2010): Input Age Data Level
    # 17) National Identifier Grid, v4.11 (2010): Growth Rate Start Year
    # 18) National Identifier Grid, v4.11 (2010): Growth Rate End Year
    # 19) National Identifier Grid, v4.11 (2010): Growth Rate Administ. Level
    # 20) National Identifier Grid, v4.11 (2010): Year of Most Recent Census
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
    
    # save data
    with open("IntermediateResults/PreparedData/Population/" + \
                                          "GPW_WA.txt", "wb") as fp:    
        pickle.dump(data_rel_filled, fp)
        pickle.dump(years, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    with open("IntermediateResults/PreparedData/Population/" + \
                                          "land_area.txt", "wb") as fp:    
        pickle.dump(land_area_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    with open("IntermediateResults/PreparedData/Prices/" + \
                                  "CountryCodesGridded.txt", "wb") as fp:    
        pickle.dump(country_codes_filled, fp)
        pickle.dump(lats, fp)
        pickle.dump(lons, fp)
    return()