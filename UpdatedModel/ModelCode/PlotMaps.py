# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:56:22 2022

@author: leip
"""

import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
from cartopy.feature import NaturalEarthFeature as cfeatNat

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# %%

def MapValues(values, lat_min = 3.25, lat_max = 18.25, lon_min = -18.75, lon_max = 10.25,
              vmin = None, vmax = None, title = "", file = None, cmap = True, close_plt = True,
              clusters = False):
    
    values[values == 0] = np.nan
    
    # extent of area
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = ccrs.PlateCarree())
        
    # add gridlines
    gls = ax.gridlines(draw_labels = True, color = "lightgray", crs = ccrs.PlateCarree())
    gls.xlabels_top = False
    gls.ylabels_right = False
    gls.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gls.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gls.xlabel_style = {"color": "gray"}
    gls.ylabel_style = {"color": "gray"}
    
    # get data for features at higher resolution
    resol = '50m'  # use data at this scale
    boder = cfeatNat(category='cultural', 
        name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    lakes = cfeatNat('physical', 'lakes', \
        scale=resol, edgecolor='skyblue', facecolor=cartopy.feature.COLORS['water'])
    rivers = cfeatNat('physical', 'rivers_lake_centerlines', \
        scale=resol, edgecolor='skyblue', facecolor='none')
        
    # add coastline, color ocean and land
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    
    # plot gridded data and add colorbar
    if clusters is True:
        cmap = mpl.cm.Paired
        bounds = np.arange(0.5, int(np.nanmax(values)) + 1, 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N) 
    else:
        cmap = "jet_r"
    im = ax.imshow(values, cmap = cmap, origin = "lower", extent = extent,
                   transform = ccrs.PlateCarree(), vmin = vmin, vmax = vmax)
    if cmap is True:
        if clusters is True:
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         orientation='horizontal',
                         ticks = range(1, int(np.nanmax(values)) + 1))
        else:
            plt.colorbar(im,
                         orientation='horizontal')
            
    
    # add borders, lakes and rivers
    ax.add_feature(lakes, alpha = 0.5)
    ax.add_feature(rivers, alpha = 0.5)
    ax.add_feature(boder, linestyle='--', edgecolor='k', alpha=1)
    
    # set title
    plt.title(title)
    
    # save if file name is provided
    if file is not None:
        fig.savefig(file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    if close_plt:
        plt.close()
    return()


def PltClusterMap(clusters, lat_min = 3.25, lat_max = 18.25,
                  lon_min = -18.75, lon_max = 10.25, title = "", 
                  file = None, cmap = True, close_plt = True ):
    
    # extent of area
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = ccrs.PlateCarree())
        
    # add gridlines
    gls = ax.gridlines(draw_labels = True, color = "lightgray", crs = ccrs.PlateCarree())
    gls.xlabels_top = False
    gls.ylabels_right = False
    gls.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gls.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gls.xlabel_style = {"color": "gray"}
    gls.ylabel_style = {"color": "gray"}
    
    # get data for features at higher resolution
    resol = '50m'  # use data at this scale
    boder = cfeatNat(category='cultural', 
        name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    lakes = cfeatNat('physical', 'lakes', \
        scale=resol, edgecolor='skyblue', facecolor=cartopy.feature.COLORS['water'])
    rivers = cfeatNat('physical', 'rivers_lake_centerlines', \
        scale=resol, edgecolor='skyblue', facecolor='none')
        
    # add coastline, color ocean and land
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    
    # plot gridded data and add colorbar
    cmap = mpl.cm.Paired
    bounds = np.arange(0.5, 10, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(clusters, cmap = cmap, origin = "lower", extent = extent, transform = ccrs.PlateCarree() )
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 orientation='horizontal',
                 ticks = range(1, 10))
    
    # add borders, lakes and rivers
    ax.add_feature(lakes, alpha = 0.5)
    ax.add_feature(rivers, alpha = 0.5)
    ax.add_feature(boder, linestyle='--', edgecolor='k', alpha=1)
    
    # set title
    plt.title(title)
    
    # save if file name is provided
    if file is not None:
        fig.savefig(file + ".jpg", bbox_inches = "tight", pad_inches = 1)
        
    if close_plt:
        plt.close()
    return()
