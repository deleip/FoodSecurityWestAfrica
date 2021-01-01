#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:07:55 2021

@author: debbora
"""
import numpy as np
import pickle
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

from ModelCode.GeneralSettings import figsize

# %% ######################### GROUPING CLUSTERS ############################## 

def GroupingClusters(k = 9, size = 5, aim = "Similar", adjacent = True, title = None, figsize = figsize):
    
    with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
        distance = pickle.load(fp)  

    with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
                 "_PearsonDistSPEI_ProfitableArea.txt", "rb") as fp:  
        pickle.load(fp) # clusters
        pickle.load(fp) # costs
        medoids = pickle.load(fp)
    DistMedoids = MedoidMedoidDist(medoids, distance)
    
    with open("InputData/Clusters/AdjacencyMatrices/k" + str(k) + "AdjacencyMatrix.txt", "rb") as fp:
        AdjacencyMatrix = pickle.load(fp)
    
    clusters = list(range(0,k))
    
    BestCosts = None
    BestGrouping = None
    valid = 0
    
    for grouping in AllGroupings(clusters, size):
        if adjacent and not CheckAdjacency(clusters, grouping, AdjacencyMatrix):
            continue
        valid += 1
        TmpCosts = CostsGrouping(grouping, DistMedoids)
        BestGrouping, BestCosts = \
           UpdateGrouping(BestCosts, TmpCosts, BestGrouping, grouping, aim)
    
    ShiftedGrouping = []
    for gr in BestGrouping:
        ShiftedGrouping.append(tuple([i + 1 for i in list(gr)]))
               
    if title is not None:
        VisualizeClusterGroups(k, size, aim, ShiftedGrouping, title, figsize, \
                          fontsizet = 20, fontsizea = 16)
    
    if adjacent:
        ad = "Adj"
    else:
        ad = ""
            
    with open("InputData/Clusters/ClusterGroups/GroupingSize" + \
                              str(size) + aim + ad + ".txt", "wb") as fp:
        pickle.dump(ShiftedGrouping, fp)
               
    return(ShiftedGrouping, BestCosts, valid)
      
def AllGroupings(lst, num):
    if len(lst) < num:
        yield []
        return
    if len(lst) % num != 0:
        # Handle odd length list
        for i in it.combinations(lst, len(lst) % num):
            lst_tmp = lst.copy()
            for j in i:
                lst_tmp.remove(j)
            for result in AllGroupings(lst_tmp, num):
                yield [i] + result
    else:
        for i in it.combinations(lst[1:], num - 1):
            i = list(i)
            i.append(lst[0])
            i.sort()
            i = tuple(i)
            lst_tmp = lst.copy()
            for j in i:
                lst_tmp.remove(j)
            for rest in AllGroupings(lst_tmp, num):
                yield [i] + rest        
                
def CheckAdjacency(clusters, grouping, AdjacencyMatrix):
    for gr in grouping:
        if len(grouping) == 1:
            continue
        ClustersNot = clusters.copy()
        for i in gr:
            ClustersNot.remove(i)
        AdjacencyMatrixRed = np.delete(AdjacencyMatrix, ClustersNot, 0)
        AdjacencyMatrixRed = np.delete(AdjacencyMatrixRed, ClustersNot, 1)
        AdjacencyMatrixRed = np.linalg.matrix_power(AdjacencyMatrixRed, len(gr)-1)
        if np.sum(AdjacencyMatrixRed[0,:] == 0) > 0:
            return(False)
    return(True)

def CostsGrouping(grouping, dist):
    costs = 0
    for gr in grouping:
        if len(gr) == 1:
            continue
        for i in it.combinations(list(gr), 2):
            costs = costs + dist[i[0], i[1]]
    return(costs)

def UpdateGrouping(BestCosts, TmpCosts, BestGrouping, grouping, aim):
    if BestCosts is None:
        return(grouping, TmpCosts)
    if aim == "Similar":
        if TmpCosts < BestCosts:
            return(grouping, TmpCosts)
        return(BestGrouping, BestCosts)
    elif aim == "Dissimilar":
        if TmpCosts > BestCosts:
            return(grouping, TmpCosts)
        return(BestGrouping, BestCosts)
            
def MedoidMedoidDist(medoids, dist):
    k = len(medoids)
    res = np.empty([k,k])
    res.fill(0)
    # get distance to all medoids
    for i in range(0, k):
        for j in range(0, k):
            if i == j:
                continue
            res[i, j] = dist[medoids[i][0]][medoids[i][1]] \
                                                [medoids[j][0], medoids[j][1]]
    return(res)

def VisualizeClusterGroups(k, size, aim, grouping, title, figsize, \
                          fontsizet = 20, fontsizea = 16):
    
    from mpl_toolkits.basemap import Basemap
        
    # shift as cisualization works with clusters 0, ..., k-1
    ShiftedGrouping = []
    for gr in grouping:
        ShiftedGrouping.append(tuple([i - 1 for i in list(gr)]))
    
    fig = plt.figure(figsize = figsize)
    
    # clusters
    with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
                 "_PearsonDistSPEI_ProfitableArea.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        
    # number of groups
    num_groups = len(ShiftedGrouping)
    
    # assignment to groups
    assign = np.empty(k)
    for idx, gr in enumerate(ShiftedGrouping):
        for j in gr:
            assign[j] = idx
    
    # getting longitudes and latitudes of region
    with open("InputData/Other/LatsLonsArea.txt", "rb") as fp:
        lats_WA = pickle.load(fp)
        lons_WA = pickle.load(fp)
        
    # initialize map
    m = Basemap(llcrnrlon=lons_WA[0], llcrnrlat=lats_WA[0], \
                urcrnrlat=lats_WA[-1], urcrnrlon=lons_WA[-1], \
                resolution='l', projection='merc', \
                lat_0=lats_WA.mean(),lon_0=lons_WA.mean())
    
    lon, lat = np.meshgrid(lons_WA, lats_WA)
    xi, yi = m(lon, lat)
    
    # Plot Data
    m.drawmapboundary(fill_color=(0.9745,0.9745,0.9857))
    cmap = cm.Paired._resample(num_groups)
    cmap = cmap(np.linspace(0,1,num_groups))
    NewMap = np.empty((0,4))
    for idx, gr in enumerate(ShiftedGrouping):
        l = len(gr)
        for j in range(0, l):
            NewColor = col.to_rgb(cmap[idx])
            NewColor = (0.6 + (j+1) * 0.35/(l+1)) * np.array(NewColor)
            NewColor = col.to_rgba(NewColor)
            NewMap = np.vstack([NewMap, NewColor])
    SortedColors = NewMap.copy()
    idx = 0
    for j in range(0,k):
        for i in [i for i, e in enumerate(assign) if e == j]:
            SortedColors[i] = NewMap[idx]
            idx +=1
            
    SortedColors = col.ListedColormap(SortedColors)
    
    c = m.pcolormesh(xi, yi, np.squeeze(clusters), cmap = SortedColors, \
                                             vmin = 0.5, vmax = k + 0.5)

    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], \
                            fontsize = fontsizea)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], \
                            fontsize = fontsizea)
    
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(linewidth=1.3, color = "dodgerblue")
    m.drawstates(linewidth=1.5)
    m.drawcountries(linewidth=1.5)
    m.drawrivers(linewidth=0.7, color='dodgerblue')
    
    # Add Title
    plt.title(title, fontsize = fontsizet, pad=20)
    plt.show()
    
    # add colorbar
    cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(c, cax = cb_ax)       
    cbar.set_ticks(range(1, k + 1))
    cbar.set_ticklabels(range(1, k + 1))
    
    # save figure
    fig.savefig("Figures/ClusterGroups/VisualizationGrouping_" + \
                "k" + str(k) + "s" + str(size) + aim + ".png", \
            bbox_inches = "tight", pad_inches = 1)
    return()
