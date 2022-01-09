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
import pickle
from cartopy.feature import NaturalEarthFeature as cfeatNat

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# %%

def MapValues(values, lat_min = 3.25, lat_max = 18.25, lon_min = -18.75, lon_max = 10.25,
              vmin = None, vmax = None, title = "", file = None, plot_cmap = True, close_plt = True):
    """
    Function to plot gridded data on top of a map, including country borders, 
    lakes and rivers.

    Parameters
    ----------
    values : np.array
        Gridded data that should be plotted.
    lat_min : float, optional
        Minimum latitdue of the provided data. The default is 3.25.
    lat_max : float, optional
        Maximum latitude of the provided data. The default is 18.25.
    lon_min : float, optional
        Minimum longitude of the provided data. The default is -18.75.
    lon_max : float, optional
        Maximum longitude of the procided data. The default is 10.25.
    vmin : float, optional
        Minimum value to be included on the color-scale. The default is None.
    vmax : float, optional
        Maximum value to be included on the color-scale. The default is None.
    title : str, optional
        Title of the plot. The default is "".
    file : str, optional
        Filename in case the plot should be saved. If None, the plot is not 
        aved. The default is None.
    plot_cmap : boolean, optional
        Whether the colorar should be added. The default is True.
    close_plt : boolean, optional
        Whether the plotting window should be closed after plotting.
        The default is True.

    Returns
    -------
    None.

    """
    
    values = values.astype(float)
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
    im = ax.imshow(values, cmap = "jet_r", origin = "lower", extent = extent,
                   transform = ccrs.PlateCarree(), vmin = vmin, vmax = vmax)
    if plot_cmap is True:
        plt.colorbar(im, orientation='horizontal')
            
    
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

def PlotClusterGroups(grouping = None, k = 9, lat_min = 3.25, lat_max = 18.25, 
                      lon_min = -18.75, lon_max = 10.25,
                      title = "", file = None, plot_cmap = True, 
                      close_plt = True):
    
    """
    Function to plot cluster on a map. If a grouing is provided, clusters of
    the same group are plotted in different shades of the same color.
    
    Parameters
    ----------
    groupnig : list of tuples or None
        Lists the cluster groups as tuples. If none, clusters are  not grouped.
    k : int
        Number of clusters the area is divided into.
    lat_min : float, optional
        Minimum latitdue of the provided data. The default is 3.25.
    lat_max : float, optional
        Maximum latitude of the provided data. The default is 18.25.
    lon_min : float, optional
        Minimum longitude of the provided data. The default is -18.75.
    lon_max : float, optional
        Maximum longitude of the procided data. The default is 10.25.
    title : str, optional
        Title of the plot. The default is "".
    file : str, optional
        Filename in case the plot should be saved. If None, the plot is not 
        aved. The default is None.
    plot_cmap : boolean, optional
        Whether the colorar should be added. The default is True.
    close_plt : boolean, optional
        Whether the plotting window should be closed after plotting.
        The default is True.

    Returns
    -------
    None.

    """
    
    if grouping is None:
        grouping = []
        for i in range(1, k + 1):
            grouping.append((i, ))
    
    ## set up plot
    
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
    
    
    ## make colormap
    def _adjustLightness(color, amount=0.5):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
        
    basecolors = ["dimgray", "firebrick", "darkorange", 
                  "forestgreen", "darkturquoise", "steelblue",
                  "blue", "darkviolet", "gold",
                  "peru", "yellow", "lightskyblue",
                  "pink", "plum", "cornflowerblue",
                  "indianred", "salmon", "khaki",
                  "palegreen", "tan"]
    
    colors = [np.nan] * k
    for idx1, tu in enumerate(grouping):
        for idx2, t in enumerate(tu):
            colors[t - 1] = _adjustLightness(basecolors[idx1], 0.6 + idx2 * 0.15)
    
    cmap = mpl.colors.ListedColormap(colors)    
    bounds = np.arange(0.5, k + 1, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N) 
    
    ## plot data
    
    with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
             "_PearsonDistSPEI.txt", "rb") as fp:  
        clusters = pickle.load(fp) # clusters
    
    ax.imshow(clusters, cmap = cmap, origin = "lower", extent = extent,
                   transform = ccrs.PlateCarree()) 
    
    if plot_cmap:
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     orientation='horizontal',
                     ticks = range(1, k + 1))
        
    
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