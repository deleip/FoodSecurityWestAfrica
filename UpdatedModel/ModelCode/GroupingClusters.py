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

from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.SettingsParameters import SetParameters

# %% ######################### GROUPING CLUSTERS ############################## 

def GroupingClusters(k = 9, size = 5, aim = "Similar", adjacent = True, 
                     metric = "medoids", title = None, figsize = None):
    """
    Group the given clusters to groups of given size, according to the 
    medoid-to-medoid distances, either with the aim to group the most similar
    clusters or the most dissimilar clusters, and either forcing clusters in a
    group to be adjacent or not.

    Parameters
    ----------
    k : int, optional
        The number of clusters in which the area is devided. The default is 9.
    size : int, optional
        The size of the cluster groups. If k cannot be devided by the size, one
        group will be smaller. The default is 5.
    aim : str, optional
        Either "Dissimilar" or "Similar". The default is "Similar".
    adjacent : boolean, optional
        Whether clusters within a group need to be adjacent. The default is True.
    metric: str, optional
        Which metric should be used to rank the groupings. The default is 
        "medoids".
    title : str, optional
        Plot title to use. If None, clusters will not be plotted. The default is None.
    figsize : tuple, optional
        The figure size. If None, the default as defined in ModelCode/GeneralSettings is used.

    Returns
    -------
    ShiftedGrouping : list of tuples
        the grouping of clusters corresponding to the grouping type and size
    BestCosts : float
        the optimal costs (in terms of medoid to medoid distance), corresponds
        to the reported cluster grouping
    valid : int
        number of groupings that would be valid for the given size and adjacency

    """
    
    
    if figsize is None:
        from ModelCode.GeneralSettings import figsize
        
    # for metric "medoids" we want to minimize (or maximize) the distance 
    # between medoids of the same group
    if metric == "medoids":
        # distances between all grid cells
        with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
            distance = pickle.load(fp)  
    
        # load medoids of clusters for given number of clusters
        with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
                     "_PearsonDistSPEI.txt", "rb") as fp:  
            pickle.load(fp) # clusters
            pickle.load(fp) # costs
            medoids = pickle.load(fp)
        DistMedoids = _MedoidMedoidDist(medoids, distance)
        
    # if metric is equality, we look at the expected_surplus (or deficit) of 
    # each cluster relative to their food demand (assuming they use the full
    # area for the more productive crop). We then will group in a way that 
    # the aggregated surplus is distributed as (un-)equal as possible over the
    # groups.
    if metric == "equality":
        expected_surplus = []
        for cl in range(1, k + 1):
            settings = DefaultSettingsExcept(k_using = cl)
            args, yield_information, population_information = \
                SetParameters(settings, expected_incomes = None, VSS = True, 
                              console_output = False, logs_on = False)
            
            exp_yields = args["ylds"][0, 0, :, :] # t / ha
            exp_yields = np.transpose(exp_yields) * args["crop_cal"] # kcal / ha
            exp_yields = np.transpose(exp_yields) * args["max_areas"] # kcal / cluster
            exp_yields = np.max(exp_yields)        
            
            demand = args["demand"][0]
            
            expected_surplus.append(exp_yields - demand)
        expected_surplus = np.array(expected_surplus)
            
    
    # get adjacency matrix for given number of clusters
    with open("InputData/Clusters/AdjacencyMatrices/k" + str(k) + "AdjacencyMatrix.txt", "rb") as fp:
        AdjacencyMatrix = pickle.load(fp)
    
    clusters = list(range(0,k))
    
    BestCosts = None
    BestGrouping = None
    valid = 0
    
    # for each possible grouping ...
    for grouping in _AllGroupings(clusters, size):
        # ... check whether all groups are connected (if demanded) ...
        if adjacent and not _CheckAdjacency(clusters, grouping, AdjacencyMatrix):
            continue
        # ... and if valid check if the grouping is better than the current best
        valid += 1
        if metric == "medoids":
            TmpCosts = _CostsGroupingMedoids(grouping, DistMedoids)
        elif metric == "equality":
            TmpCosts = _costsGroupingEquality(grouping, expected_surplus)
        BestGrouping, BestCosts = \
           _UpdateGrouping(BestCosts, TmpCosts, BestGrouping, grouping, aim)
    
    # shift cluster numbers so we have clusters 1, ..., k 
    ShiftedGrouping = []
    for gr in BestGrouping:
        ShiftedGrouping.append(tuple([i + 1 for i in list(gr)]))
               
    # visualize resulting cluster groups
    if title is not None:
        VisualizeClusterGroups(k, size, aim, adjacent, ShiftedGrouping, title, figsize, \
                          fontsizet = 20, fontsizea = 16)
    
    if adjacent:
        ad = "Adj"
    else:
        ad = ""
            
    with open("InputData/Clusters/ClusterGroups/Grouping" + 
              metric.capitalize() + "Size" + \
                              str(size) + aim + ad + ".txt", "wb") as fp:
        pickle.dump(ShiftedGrouping, fp)
               
    return(ShiftedGrouping, BestCosts, valid)
      
def _AllGroupings(lst, num):
    """
    Create all possible groupings for the given clusters and custer size.

    Parameters
    ----------
    lst : list
         list(range(0, number of clusters))
    num : int,
        The size of the cluster groups.
        
    Yields
    ------
    list
        all possible grouping combinations.

    """
    if len(lst) < num:
        yield []
        return
    if len(lst) % num != 0:
        # Handle odd length list
        for i in it.combinations(lst, len(lst) % num):
            lst_tmp = lst.copy()
            for j in i:
                lst_tmp.remove(j)
            for result in _AllGroupings(lst_tmp, num):
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
            for rest in _AllGroupings(lst_tmp, num):
                yield [i] + rest        
                
def _CheckAdjacency(clusters, grouping, AdjacencyMatrix):
    """
    Checks if a given cluster grouping is made up of connected groups

    Parameters
    ----------
    clusters : list
        list(range(0, number of clusters))
    grouping : list
        given grouping.
    AdjacencyMatrix : np.array
        Adjacency matrix for the given clusters.

    Returns
    -------
    res : boolean
        Whether cluster grouping is made up of connected groups

    """
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

def _CostsGroupingMedoids(grouping, dist):
    """
    Calculates the cost (sum of medoid to medoid distances within cluster groups)
    of given grouping

    Parameters
    ----------
    grouping : list
        given cluster grouping.
    dist : np.array
        medoid to medoid distances.

    Returns
    -------
    costs : float
        sum of medoid to medoid distances within cluster groups.

    """
    costs = 0
    for gr in grouping:
        if len(gr) == 1:
            continue
        for i in it.combinations(list(gr), 2):
            costs = costs + dist[i[0], i[1]]
    return(costs)

def _costsGroupingEquality(grouping, expected_surplus):
    """
    Calculates the cost (based on difference between expected surplus of 
    different groups) of given grouping

    Parameters
    ----------
    grouping : list
        given cluster grouping.
    expected_surplus : np.array
        expected surplus in each cluster separately.

    Returns
    -------
    costs : float
        maximum difference between the aggregated surplus of two different 
        groups in this grouping
    """
    
    surplus = []
    for gr in grouping:
        gr = list(gr)
        tmp = np.sum(expected_surplus[gr])
        surplus.append(tmp)
       
    costs = 0
    for s1 in surplus:
        for s2 in surplus:
            costs += abs(s1 - s2)
            
    return(costs)

def _UpdateGrouping(BestCosts, TmpCosts, BestGrouping, grouping, aim):
    """
    Updates the so far best grouping if the new grouping is better.

    Parameters
    ----------
    BestCosts : float
        Costs of the best grouping checked so far.
    TmpCosts : float
        Costs of the newest grouping that was checked.
    BestGrouping : list
        The grouping with the best costs so far.
    grouping : list
         Newest grouping that was checked.
    aim : str
        Either "Dissimilar" or "Similar". Defines if the "best" costs means
        the highest or the lowest costs

    Returns
    -------
    BestGrouping : list
        Updated version of BestGrouping
    BestCosts : float
        Updated version of BestCosts

    """
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
            
def _MedoidMedoidDist(medoids, dist):
    """
    Creates matrix of medoid to medoid distances based on distances between
    all grid cells

    Parameters
    ----------
    medoids : list
        list of "coordinates" of the medoids.
    dist : list
        distance between each pair of grid cells. (list of list of np.arrays)

    Returns
    -------
    None.

    """
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

def VisualizeClusterGroups(k, size, aim, adjacent, grouping, title, figsize, \
                          fontsizet = 20, fontsizea = 16):
    """
    Visulaizes the cluster groups on a map

    Parameters
    ----------
    k : int, optional
        The number of clusters in which the area is devided. The default is 9.
    size : int, optional
        The size of the cluster groups. If k cannot be devided by the size, one
        group will be smaller. The default is 5.
    aim : str, optional
        Either "Dissimilar" or "Similar". The default is "Similar".
    adjacent : boolean, optional
        Whether clusters within a group need to be adjacent. The default is True.
    grouping : list of tuples
        Specifies the cluster groups.
    title : str, optional
        Plot title to use. If None, clusters will not be plotted. The default is None.
    figsize : tuple, optional
        The figure size. If None, the default as defined in ModelCode/GeneralSettings is used.
    fontsizet : int, optional
        Fontsize to use for the plot title. The default is 20.
    fontsizea : int, optional
        Fontsize to use for the logitudes and latitudes. The default is 16.

    Returns
    -------
    None.

    """
    
    from mpl_toolkits.basemap import Basemap
        
    # shift as cisualization works with clusters 0, ..., k-1
    ShiftedGrouping = []
    for gr in grouping:
        ShiftedGrouping.append(tuple([i - 1 for i in list(gr)]))
    
    fig = plt.figure(figsize = figsize)
    
    # clusters
    with open("InputData/Clusters/Clustering/kMediods" + str(k) + \
                 "_PearsonDistSPEI.txt", "rb") as fp:  
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
    
    # add adjacency to file name
    if adjacent:
        ad = "Adj"
    else:
        ad = ""
        
    # save figure
    fig.savefig("Figures/ClusterGroups/VisualizationGrouping_" + \
                "k" + str(k) + "s" + str(size) + aim + ad +".png", \
            bbox_inches = "tight", pad_inches = 1)
    return()
