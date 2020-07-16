#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:14:55 2020

@author: debbora
"""


# %% IMPORTING NECESSARY PACKAGES 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import stats
import pickle
import random
from scipy.spatial import distance
import matplotlib.cm as cm
import statsmodels.api as sm
import datetime as dt


# %% DEFINING FUNCTIONS

# ---------------------- CALCULATING DISTANCES --------------------------------

# 1) - Calculate Pearson Correlation between all grid cells using a given time 
#    window, saved in form [lat cell 1][lon cell 1][lat cell 2, lon cell 2]
#    - Calculate Pearson distance out of given correlation:
#    d(x,y) = sqrt(0.5*(1-corr(x,y))) 
def CalcPearson(data_in, mask, title, tw_start = None, tw_end = None, 
                DistType = "Pearson", cut = None, boolean = False):
    data = data_in.copy()
    title = DistType + "Dist" + "_" + title
    if boolean:        # just looking at pattern of extreme events
        data[data <= cut] = -np.inf   
        data[data > cut] = 0   
        data[data == -np.inf] = 1
        title = title + "_Boolean"
    corr = []
    dist = []
    [n_t, num_lat, num_lon] = data.shape
    # setting timewindow
    if  tw_start == None:
        tw_start = 0
    if tw_end == None:
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
                    if (cut == None) or (boolean): # use months for which 
                                                   # both cells have data
                        use = np.logical_and(~np.isnan(X), ~np.isnan(Y)) 
                    else: # using only months where extreme 
                          # event in at least one of the cells
                        use1 = np.logical_and(~np.isnan(X), ~np.isnan(Y)) 
                        use2 = np.logical_or(X < cut, Y < cut)
                        use = np.logical_and(use1, use2)
                    if np.sum(use) > 1:
                        if DistType == "Pearson":
                            corr_tmp[lat2,lon2] = stats.pearsonr(X[use], \
                                                                     Y[use])[0]
                            dist_tmp[lat2,lon2] = np.sqrt(0.5*(1 - \
                                                          corr_tmp[lat2,lon2]))
                        elif DistType == "Jaccard":
                            dist_tmp[lat2,lon2] = distance.jaccard(X[use], \
                                                                     Y[use])
                        elif DistType == "Dice":
                            dist_tmp[lat2,lon2] = distance.dice(X[use], \
                                                                     Y[use])
            corr_lon.append(corr_tmp)
            dist_lon.append(dist_tmp)
        corr.append(corr_lon)    
        dist.append(dist_lon)
    if cut != None:
        if cut < 0:
            title = title + "_CutNeg" + str(abs(cut))
        else:
            title = title + "_Cut" + str(cut)        
    # saving results
    if DistType == "Pearson":
        with open("IntermediateResults/Clustering/Distances/" + \
                            "PearsonCorr_" + title + "_p2.txt", "wb") as fp:    
            pickle.dump(corr, fp, protocol = 2)    
    with open("IntermediateResults/Clustering/Distances/" + \
                             title + "_p2.txt", "wb") as fp:    
        pickle.dump(dist, fp, protocol = 2)    
    return()    

# --------------------------- k-Medoids Algorithm -----------------------------


# Definition of the k-Medoids algorithm:
# Step 1. k different objects are randomly chosen as initial medoids 
#         (instead of using a greedy algo)
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
    # initializing variables
    cluster_all, cost_all, medoids_all = [], [], []
    [num_lats, num_lons] = mask.shape
    terminated = False
    step = 0
    # Normally, the initial medoids are chosen randomly, but if we wish to 
    # continue a run that had not yet terminated or for some other reason want 
    # to start with specific medoids these can be given to the function and 
    # will be used
    random.seed(seed) 
    if start_medoids == None:
        medoids = GetInitialMedoids(k, dist, mask, num_lats, num_lons)
#        medoids = []
#        count = 0
#        while count < k:
#            lat_tmp = random.randrange(0, num_lats)
#            lon_tmp = random.randrange(0, num_lons)
#            # checking that it's not ocean
#            if mask[lat_tmp, lon_tmp] == 0:
#                continue
#            # and that we didn't chose it already
#            if (lat_tmp, lon_tmp) in medoids:
#                continue
#            medoids.append((lat_tmp, lon_tmp))
#            count += 1
    else:
        medoids = start_medoids
    # get best cluster for these medoids and save the results of this step
    cluster, cost = GetCluster(k, dist, num_lats, num_lons, medoids, mask)
    cluster_all.append(cluster + 1) # medoids counted from 0 to k-1, but we 
                                    # want clusters to be 1 to k
    cost_all.append(cost)
    medoids_all.append(medoids)
    # Now we iterated until either the maximal number of iteration is reached 
    # or no improvment can be found
    while terminated == False:
        print(step)
        # going trough all possible switches and finding the best one
        new_cluster, new_cost, new_medoids = GetSwitch(k, dist, num_lats, \
                                                       num_lons, medoids, mask)
        # if the best possible switch actually improves the clusters we do the
        # switch and save the outcome of this step
        if new_cost < cost:
            cost, cluster, medoids = new_cost, new_cluster, new_medoids
            cluster_all.append(cluster + 1)
            cost_all.append(cost)
            medoids_all.append(medoids)
            step += 1
            # checking if maximal number of iterations is reached
            if term == False and step > max_its:
                terminated = True                           
            continue
        # if best switch is not improving the clustering, we print a 
        # corresponding message and terminate the algorithm
        print("No improvement found")
        terminated = True
    if seed == 3052020:
        title_seed = ""
    else:
        title_seed = "_seed" + str(seed)
    with open("IntermediateResults/Clustering/Clusters/" + version + \
                              "kMediods" + str(k) + "_" + file + \
                              title_seed + ".txt", "wb") as fp:    
        pickle.dump(cluster_all, fp)
        pickle.dump(cost_all, fp)
        pickle.dump(medoids_all, fp)
    return()
    
# 1) Greedy algorithm for initial medoids
def GetInitialMedoids(k, dist, mask, num_lats, num_lons):
    medoids = []
    for l in range(1, k+1):
        best_cost = 0
        for i in range(0, num_lats):
            for j in range(0, num_lons):
                # ocean is not considered in the clustering
                if (mask[i, j] == 0) or ((i,j) in medoids):
                    continue
                medoids_tmp = medoids.copy()
                medoids_tmp.append((i,j))
                cluster, cost = \
                    GetCluster(l, dist, num_lats, num_lons, medoids_tmp, mask)
                if best_cost == 0:
                    best_cost = cost
                    best_medoid = (i,j)
                elif cost < best_cost:
                    best_cost = cost
                    best_medoid = (i,j)
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


# ---------------------- Visualization of clusters ----------------------------
    
# 1) Function to visualize the cluster given by k-medoids on a map   
    
def VisualizeFinalCluster(clusters, medoids, lats_rel, lons_rel, 
               title, show_medoids = False, fontsizet = 12, fontsizea = 11):
    # number of cluster
    k = len(medoids[0])
    
    # initialize map
    m = Basemap(llcrnrlon=lons_rel[0], llcrnrlat=lats_rel[0], \
                urcrnrlat=lats_rel[-1], urcrnrlon=lons_rel[-1], \
                resolution='l', projection='merc', \
                lat_0=lats_rel.mean(),lon_0=lons_rel.mean())
    
    lon, lat = np.meshgrid(lons_rel, lats_rel)
    xi, yi = m(lon, lat)
    
    # Plot Data
    data = clusters[-1]
    m.drawmapboundary(fill_color='azure')
    cmap1 = cm.Paired._resample(k)
    if k == 1:
        cmap1 = "YlGn"
    c = m.pcolormesh(xi, yi, np.squeeze(data), cmap = cmap1, \
                                             vmin = 0.5, vmax = k + 0.5)
    # Plot medoids
    if show_medoids:
        draw_medoids = np.empty(data.shape)
        draw_medoids.fill(np.nan)
        for [i,j] in medoids[-1]:
            draw_medoids[i,j] = 1
        m.pcolormesh(xi, yi, np.squeeze(draw_medoids), cmap = "Greys_r", \
                                                         vmin = 0, vmax = 2)

    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], \
                            fontsize = fontsizea)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], \
                            fontsize = fontsizea)
    
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(linewidth=1.3)
    m.drawstates(linewidth=1.3)
    m.drawcountries(linewidth=1.3)
    m.drawrivers(linewidth=0.7, color='blue')

    # Add Title
    plt.title(title, fontsize = fontsizet, pad=20)
    plt.show()
    return(c)
    
# -------------- Visualization correlation within/between clusters ------------

# 1) Getting the distance/correlation of all grid cells to a specifc medoid
#    or if None chosen to the respective cluster medoid
    
def GetDistToMedoid(cluster, medoids, dist, medoid_num = None):
    [num_lats, num_lons] = cluster[0].shape
    res = np.empty([num_lats, num_lons])
    res.fill(np.nan)
    final_cluster = cluster[-1]
    final_medoids = medoids[-1]
    # dist to respective cluster medoid
    if medoid_num == None:
        for i in range(0, num_lats):
            for j in range(0, num_lons):
                if np.isnan(final_cluster[i, j]):
                    continue
                [med_i, med_j] = final_medoids[int(final_cluster[i, j] - 1)]
                res[i,j] = dist[med_i][med_j][i,j]
        return(res)       
    # dist to chosen medoid
    else:
        [med_i, med_j] = final_medoids[int(medoid_num - 1)]
        for i in range(0, num_lats):
            for j in range(0, num_lons):
                if np.isnan(final_cluster[i, j]):
                    continue
                res[i,j] = dist[med_i][med_j][i,j]
        return(res) 

# 2) Plot these distances/correlations on a map     
        
def PlotDistToMediod(cluster, medoids, dist, title, lats_rel, lons_rel,
                                     medoid_num = None, show_medoids = False):
    
    data = GetDistToMedoid(cluster, medoids, dist, medoid_num)
    
    # initialize map
    m = Basemap(llcrnrlon=lons_rel[0], llcrnrlat=lats_rel[0], \
                urcrnrlat=lats_rel[-1], urcrnrlon=lons_rel[-1], \
                resolution='l', projection='merc', \
                lat_0=lats_rel.mean(),lon_0=lons_rel.mean())

    lon, lat = np.meshgrid(lons_rel, lats_rel)    
    xi, yi = m(lon, lat)
    
    # Plot Data
    m.drawmapboundary(fill_color='azure')
    c = m.pcolormesh(xi, yi, np.squeeze(data), vmin = 0, vmax = 1, \
                                                              cmap = 'jet_r')
    
    # Plot medoids
    if show_medoids:
        draw_medoids = np.empty(data.shape)
        draw_medoids.fill(np.nan)
        for [i,j] in medoids[-1]:
            draw_medoids[i,j] = 1
        m.pcolormesh(xi, yi, np.squeeze(draw_medoids), cmap = "Greys_r", \
                                                         vmin = 0, vmax = 2)

    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=8)
    
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(linewidth=1.1)
    m.drawstates(linewidth=1.1)
    m.drawcountries(linewidth=1.1)
    m.drawrivers(linewidth=0.5, color='blue')

    # Add Title
    plt.title(title)
    plt.show()
    return(c)
    
# 3) Plot a histogram with the distribution of distance/correlation between all
#    grid cells within one specific cluster and its medoid and a chosen number 
#    of closets neighbouring medoids
    
def HistogramDistCluster(cluster, medoids, dist, cl_num, neighbour_num, Dist):
    k = len(medoids[-1])
    euc_dist = np.zeros(k)
    medoid_cl = medoids[-1][cl_num - 1]
    # euclidean distances between medoids to decide on "closest neighbors"
    for m in range(0, k):
       euc_dist[m] = distance.euclidean(medoid_cl, medoids[-1][m])
    # sorting clusters according to euclidean distance
    cl = np.array(range(0,k))
    cl = cl[euc_dist.argsort()]  
    legend =  []    
    # plotting histograms for the chosen number of clostest neighbors
    for i in cl[0:(neighbour_num + 1)]:       
        dist_tmp = GetDistToMedoid(cluster, medoids, dist, medoid_num = i + 1)
        dist_tmp = dist_tmp[cluster[-1] == cl_num]
        plt.hist(dist_tmp.flatten(), alpha = 0.5, bins = 60, \
                                     range = (0,1), density = True)
        legend.append("Medoid  " + str(i + 1))
    plt.legend(legend, loc='upper left', prop={'size': 8}, fancybox = True)
    if cl_num == 1:
        plt.title(Dist + "\n Cluster 1")
    else:
        plt.title("Cluster " + str(cl_num))
    return()

    
# -------------------- Finding optimal number of clusters  --------------------
    
# 1) Calculating the Davies-Bouldin-Index        
    
def DaviesBouldinIndex(cluster, medoids, dist):
    cluster = cluster[-1]
    medoids = medoids[-1]
    k = int(np.nanmax(cluster))
    avg_dist = np.zeros([k])
    med_dist = np.zeros([k, k])
    
    # calculating average distance from cells to the respective cluster mediod
    # (avg_dist) and the distance between the different medoids (med_dist)
    for i in range(0, k):
        [med1_lat, med1_lon] = medoids[i]
        avg_dist[i] = np.average(dist[med1_lat][med1_lon][cluster == i + 1])
        for j in range(0, k):
            [med2_lat, med2_lon] = medoids[j]
            med_dist[i, j] = dist[med1_lat][med1_lon][med2_lat, med2_lon]
    
    # for each medoid we take the worst ratio between average distance within 
    # that cluster plus average distance within another cluster to the distance 
    # between the mediods. These ratios are averaged over all clusters to get
    # the Davies-Bouldin-Index
    DB = 0
    for i in range(0, k):
        val = 0
        for j in range(0, k):
            if j == i:
                continue
            val_tmp = (avg_dist[i] + avg_dist[j])/med_dist[i, j]
            if val_tmp > val:
                val = val_tmp
        DB = DB + val
    DB = DB / k
    
    return(DB)

# 2) Method to find elbow of the Davies-Bouldin-Index for changing number 
#    of clusters
    
def elbow(values):
    l = len(values)
    s = []
    for i in range(1, l-1):
        s.append(values[i-1] - 3 * values[i] + 2 * values[i+1])
    el = s.index(max(s)) + 3
    return(el, s)  

# 3) Distances of one medoid to either all or the closest other medoid    
    
def MedoidMedoidDistd(medoids, dist):
    medoids = medoids[-1]
    k = len(medoids)
    res = np.empty([k,k])
    res.fill(np.nan)
    for i in range(0, k):
        for j in range(0, k):
            if i == j:
                continue
            res[i, j] = dist[medoids[i][0]][medoids[i][1]] \
                                                [medoids[j][0], medoids[j][1]]
    res_closest = np.zeros(k)
    for i in range(0, k):
        res_closest[i] = np.nanmin(res[i])
    return(res, res_closest)
    

# 4) For each cluster find minimum/average of average distance of all its 
#    clusters to another medoid
    
def CellsMedoidsDist(medoids, dist, cluster): 
    k = len(medoids[-1])
    res = np.empty([k,k])
    res.fill(np.nan)
    for i in range(0, k):
        dist_tmp = GetDistToMedoid(cluster, medoids, dist, medoid_num = i+1)
        for j in range(0, k):
            res[j, i] = np.nansum(dist_tmp[cluster[-1] == j])
    res_diag = np.diag(res).copy()
    np.fill_diagonal(res, np.nan)
    res_closest = np.zeros(k)
    for j in range(0, k):
        res_closest[j] = np.nanmin(res[j,:])
    return(res, res_diag, res_closest)

# distances as average between all cells of cluster A to medoid + all cells
# of cluster B to medoid A
def DistBetweenCluster(medoids, dist, cluster):
    k = len(medoids[-1])
    cm, cm_diag, cm_closest = CellsMedoidsDist(medoids, dist, cluster)
    cm_tr = np.transpose(cm)
    res = cm + cm_tr
    num_comb = DistanceCombinations(cluster)
    res = res/num_comb
    res_diag = np.diag(res).copy()
    np.fill_diagonal(res, np.nan)
    res_closest = np.zeros(k)
    for j in range(0, k):
        res_closest[j] = np.nanmin(res[j,:])
    return(res, res_diag, res_closest)
    
def DistanceCombinations(clusters):
    cluster = clusters[-1]
    k = int(np.nanmax(cluster))
    num_comb = np.zeros([k, k])
    for i in range(0, k):
        for j in range(0, k):
            num_comb = np.sum(cluster == (i+1)) + np.sum(cluster == (j+1))
    return(num_comb)
    
#def RelSizeOfClusters(clusters):
#    cluster = clusters[-1]
#    k = int(np.nanmax(cluster))
#    rel_areas = np.zeros(k)
#    land_cells = np.sum(~np.isnan(cluster))
#    for i in range(0, k):
#        rel_areas[i] = np.sum(cluster == (i + 1))/land_cells
#    return(rel_areas)
        

def MetricClustering(dist_within, dist_between, refX = 0, refY = 1):
    m = np.sqrt(((np.array(dist_within)-refX)**2) + \
                ((refY- np.array(dist_between))**2))
    order = np.argsort(m)
    cl_order = order + 2
    m = m[order]
    return(m, cl_order)
        

def VarianceOfCluster(data, clusters):
    len_ts = data.shape[0]
    cluster = clusters[-1]
    k = int(np.nanmax(cluster))
    variance = np.empty([len_ts, k, k]); variance.fill(np.nan)
    for cl1 in range(0, k):
        for cl2 in range(0, k):
            for t in range(0, len_ts):
                variance[t, cl1, cl2] = np.nanvar(data[t, \
                        ((cluster == (cl1 + 1)) + (cluster == (cl2 + 1)))])
    return(variance)
    
def ClusterMetricVariance(data, clusters):
    variance = VarianceOfCluster(data, clusters)
    variance = np.nanmean(variance, 0)
    m_within = np.diag(variance).copy()
    np.fill_diagonal(variance, np.nan)
    m_between_closest = np.nanmin(variance, 1)
    m_between_all = np.nanmean(variance, 1)
    return(m_within, m_between_closest, m_between_all)
    

# --------------------- preparing data for regression -------------------------

# 1) Main function to get data: reads from prepared AgMIP output files, and
#    prepares respective SPEI/wd data (depending on settings)
     
def RegressionDataAgMIP(model, climate, harm, irri, crop, var_names, \
                        detrend_yield, year_start, detrend_vars = True):
    model_setting = model + "_" + climate + "_" + harm + "_"     
    crop_irr =  crop + "_" + irri   
    
    variables = []
    
    # reading variables from AgMIP output (always yield, plant-day, others)
    for var_name in var_names:
        with open("IntermediateResults/PreparedData/AgMIP/" + model_setting + \
                              var_name + "_" + crop_irr +  ".txt", "rb") as fp:    
            variables.append(pickle.load(fp))    
    
    # different models have different missing cells...
    with open("IntermediateResults/PreparedData/AgMIP/" + \
                                     model_setting + "mask.txt", "rb") as fp:    
        mask = pickle.load(fp)  
                
    if detrend_yield:
        variables[0], p_val, sl, intc = DetrendDataLinear(variables[0], mask)
    if detrend_vars:
        for i in range(1, len(variables)):
            variables[i], p_val, sl, intc = DetrendDataLinear(variables[i], \
                                                                         mask)
        
    # loading SPEI and wd data (detrended or raw data depenting on setting)    
    if detrend_vars: 
        with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                "spei03_WA_detrend.txt", "rb") as fp:  
            spei = pickle.load(fp) 
        with open("IntermediateResults/PreparedData/CRU/" + \
                                "WaterDeficit03_WA_detrend.txt", "rb") as fp:  
            wd = pickle.load(fp) 
    else:
        with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                "spei03_WA_filled.txt", "rb") as fp:  
            spei = pickle.load(fp) 
        with open("IntermediateResults/PreparedData/CRU/" + \
                                "WaterDeficit03_WA.txt", "rb") as fp:  
            wd = pickle.load(fp) 
    
    # taking SPEI/wd values related to the model outputs:
    # either the lowest value in the respective year, or the 3-month SPEI/wd 
    # corrsponding to the growing season
    pltmths = PltdayToPltmths(variables[1], year_start)
    spei_annual_gs = GetAnnual(spei, pltmths, mask, 
                               year_start, "growing_season")
    spei_annual_lowest = GetAnnual(spei, pltmths, mask, 
                                   year_start, "annual_lowest")
    wd_annual_gs = GetAnnual(wd, pltmths, mask, 
                               year_start, "growing_season")
    wd_annual_lowest = GetAnnual(wd, pltmths, mask, 
                                   year_start, "annual_lowest")
 
    return(variables, mask, spei_annual_gs, \
           spei_annual_lowest, wd_annual_gs, wd_annual_lowest)
        
# 2) Helpfunction as AgMIP output gives planting time in days of year, but we 
#    need to know the month, as SPEI and wd are monthly.
    
def PltdayToPltmths(pltdays, year_start):
    pltmths = np.empty(pltdays.shape); pltmths.fill(np.nan)
    [n_t, n_lat, n_lon] = pltdays.shape
    for t in range(0, n_t):
        for i in range(0, n_lat):
            for j in range(0, n_lon):
                if np.isnan(pltdays[t, i, j]):
                    continue
                pltmths[t, i, j] = (dt.datetime(year_start + t, 1, 1) +\
                                     dt.timedelta(pltdays[t, i, j] - 1)).month
    return(pltmths)
    
def PltdayToPltmthsSingle(pltdays, mask):# no different plantdays for each year
    pltmths = np.empty(pltdays.shape); pltmths.fill(np.nan)
    [n_lat, n_lon] = pltdays.shape
    for i in range(0, n_lat):
        for j in range(0, n_lon):
            if mask[i,j] == 0:
                continue
            pltmths[i, j] = (dt.datetime(2000, 1, 1) + \
                               dt.timedelta(pltdays[i, j] - 1)).month
    return(pltmths)
    
# 3) Helpfunction to detrend AgMIP yield data (Same as for DataPreparation.py)
    
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

# 4) Get annual SPEI/wd data related to the AgMIP model output
    
def GetAnnual(data, pltmths, mask, year_start, relation_DI):
    [n_t, n_lat, n_lon] = pltmths.shape
    res = np.empty(pltmths.shape); res.fill(np.nan)
    for t in range(0, n_t):
        for i in range(0, n_lat):
            for j in range(0, n_lon):
                if relation_DI == "growing_season":
                    if np.isnan(pltmths[t, i, j]):
                        continue
                    res[t, i, j] = data[int(12 * (year_start - 1901 + t) + \
                                           (pltmths[t, i, j] - 1) + 2), i, j]
                if relation_DI == "annual_lowest":
                    res[t, i, j] = data[int(12 * (year_start - 1901 + t)) : \
                             int(12 * (year_start - 1901 + t + 1)), i, j].min()
    return(res)
    
def GetAnnualSingle(data, pltmths, mask, year_start, relation_DI, yld): 
    [n_t, n_lat, n_lon] = yld.shape
    res = np.empty(yld.shape); res.fill(np.nan)
    for t in range(0, n_t):
        for i in range(0, n_lat):
            for j in range(0, n_lon):
                if relation_DI == "growing_season":
                    if np.isnan(pltmths[i, j]):
                        continue
                    res[t, i, j] = data[int(12 * (year_start - 1901 + t) + \
                                           (pltmths[i, j] - 1) + 2), i, j]
                if relation_DI == "annual_lowest":
                    res[t, i, j] = data[int(12 * (year_start - 1901 + t)) : \
                             int(12 * (year_start - 1901 + t + 1)), i, j].min()
    return(res)
   
# ------------------------ visualizing data on map ----------------------------
    
def MapValues(values, lats_rel, lons_rel, \
              title = "", vmin = None, vmax = None, ax = None):    
    # initialize map
    m = Basemap(llcrnrlon=lons_rel[0], llcrnrlat=lats_rel[0], \
                urcrnrlat=lats_rel[-1], urcrnrlon=lons_rel[-1], \
                resolution='l', projection='merc', \
                lat_0=lats_rel.mean(),lon_0=lons_rel.mean(), ax = ax)
    
    lon, lat = np.meshgrid(lons_rel, lats_rel)
    xi, yi = m(lon, lat)
    
    # Plot Data
    m.drawmapboundary(fill_color='azure')
    c = m.pcolormesh(xi,yi,np.squeeze(values), cmap = 'jet_r', \
                                          vmin = vmin, vmax = vmax)

    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[0,1,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=8)
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(linewidth=1.1)
    m.drawstates(linewidth=1.1)
    m.drawcountries(linewidth=1.1)
    m.drawrivers(linewidth=0.5, color='blue')
    # Add Title
    if ax:
        ax.set_title(title)
    else:
        plt.title(title)
    plt.show()
    return(c)
    
def MapValuesCluster(clusters, values, lats_rel, lons_rel, \
              title = "", vmin = None, vmax = None, ax = None):    
    # create values per cell
    cluster = clusters[-1]
    values_new = np.empty(cluster.shape); values_new.fill(np.nan)
    for k in range(0, int(np.nanmax(cluster))):
        values_new[cluster == (k+1)] = values[k]
    
    # use original MapValues
    c = MapValues(values_new, lats_rel, lons_rel, title, vmin, vmax, ax)
    return(c)    

# -------------------------- linear regressions -------------------------------

def LinearRegressionCells(yld, mask, climate_var, other_vars, time = False):
    [n_lat, n_lon] = mask.shape
    
    # Number of independent variables
    num_ind_vars = len(other_vars) + 1  # plus constant 
    if climate_var != "None":
        num_ind_vars = num_ind_vars + 1     
    if time == True:
        num_ind_vars = num_ind_vars + 1 
    
    # initalizing results
    pvals = np.empty([n_lat, n_lon, num_ind_vars]); pvals.fill(np.nan) 
    preds = np.empty(yld.shape); preds.fill(np.nan)
    errors = np.empty(yld.shape); errors.fill(np.nan)
    fstat = np.empty(mask.shape); fstat.fill(np.nan)
    rsquared = np.empty(mask.shape); rsquared.fill(np.nan)
    
    # regression per cell
    for i in range(0, n_lat):
        for j in range(0, n_lon):
            if mask[i,j] == 0:
                continue
            # data
            y = yld[:, i, j]
            X = np.repeat(1,yld.shape[0])
            if time == True:
                X = np.column_stack((X, np.array(range(0,yld.shape[0]))))
            if climate_var != "None":
                X = np.column_stack((X, climate_var[:, i, j]))
            for v in range(0, len(other_vars)):
                X = np.column_stack((X, other_vars[v][:, i, j]))
            # regression
            if (np.sum(~np.isnan(y)) == 0):
                continue
            if (climate_var != "None") & \
                        (np.sum(~np.isnan(climate_var[:, i, j])) == 0):
                continue
            model = sm.OLS(y,X, missing = "drop")
            results = model.fit()
            predictions = results.predict(X)       
            err = y - predictions
            # saving results
            pvals[i, j, :] = results.pvalues
            fstat[i, j] = results.f_pvalue
            rsquared[i, j] = results.rsquared
            preds[:, i, j] = predictions      
            errors[:, i, j] = err    
    return(errors, pvals, fstat, preds, rsquared, yld)
 
def LinearRegressionClusterAverage(yld, clusters, mask, climate_var, \
                                  other_vars, time = False):
    cluster = clusters[-1]
    k = int(np.nanmax(cluster))
    n_t = yld.shape[0]
    
    # Number of independent variables
    num_ind_vars = len(other_vars) + 1  # plus constant 
    if climate_var != "None":
        num_ind_vars = num_ind_vars + 1     
    if time == True:
        num_ind_vars = num_ind_vars + 1 
        
    # initalizing results
    pvals = np.empty([k, num_ind_vars]); pvals.fill(np.nan)
    preds = np.empty([n_t, k]); preds.fill(np.nan) 
    errors = np.empty([n_t, k]); errors.fill(np.nan)
    fstat = np.empty(k);  fstat.fill(np.nan)
    rsquared = np.empty(k); rsquared.fill(np.nan)
    
    # get cluster averages
    yld_cl = ClusterAverage(yld, cluster, k, mask)
    if climate_var != "None":
        climate_var_cl = ClusterAverage(climate_var, cluster, k, mask)
    other_vars_cl = []
    for i in range(0, len(other_vars)):
        other_vars_cl.append(ClusterAverage(other_vars[i], cluster, k, mask))
        
    # regression per cluster
    for cl in range(0, k):
        # data
        y = yld_cl[:, cl]
        X = np.repeat(1, n_t)
        if time == True:
            X = np.column_stack((X, np.array(range(0, n_t))))
        if climate_var != "None":
            X = np.column_stack((X, climate_var_cl[:, cl]))
        for v in range(0, len(other_vars)):
            X = np.column_stack((X, other_vars_cl[v][:, cl]))
        # regression
        if np.sum(~np.isnan(y)) == 0:
            continue
        model = sm.OLS(y,X, missing = "drop")
        results = model.fit()
        predictions = results.predict(X)       
        err = y - predictions
        # saving results
        pvals[cl, :] = results.pvalues
        fstat[cl] = results.f_pvalue
        rsquared[cl] = results.rsquared
        preds[:, cl] = predictions      
        errors[:, cl] = err    
    return(errors, pvals, fstat, preds, rsquared, yld_cl)
    
def LinearRegressionClusterSample(yld, clusters, mask, climate_var, \
                                  other_vars, time = False):
    cluster = clusters[-1]
    k = int(np.nanmax(cluster))
    
    # Number of independent variables
    num_ind_vars = len(other_vars) + 1  # plus constant 
    if climate_var != "None":
        num_ind_vars = num_ind_vars + 1     
    if time == True:
        num_ind_vars = num_ind_vars + 1 
        
    # stacking data of clusters
    yld_stacked, const_stacked, time_stacked, \
    climate_stacked, other_vars_stacked = \
                ClusterIncreaseSample(yld, climate_var, other_vars, cluster, k)
    
    # initalizing results
    pvals = np.empty([k, num_ind_vars]); pvals.fill(np.nan)
    preds = np.empty(yld_stacked.shape); preds.fill(np.nan) 
    errors = np.empty(yld_stacked.shape); errors.fill(np.nan)
    fstat = np.empty(k);  fstat.fill(np.nan)
    rsquared = np.empty(k); rsquared.fill(np.nan)
    
    # regression per cluster
    for cl in range(0, k):
        # data
        y = yld_stacked[:, cl]
        X = const_stacked[:, cl]
        if time == True:
            X = np.column_stack((X, time_stacked[:,cl]))
        if climate_var != "None":
            X = np.column_stack((X, climate_stacked[:,cl]))
        for v in range(0, len(other_vars)):
            X = np.column_stack((X, other_vars_stacked[v][:, cl]))    
        # regression
        model = sm.OLS(y,X, missing = "drop")
        results = model.fit()
        predictions = results.predict(X)      
        err = y - predictions
        # saving results
        pvals[cl, :] = results.pvalues
        fstat[cl] = results.f_pvalue
        rsquared[cl] = results.rsquared
        preds[:, cl] = predictions      
        errors[:, cl] = err        

    return(errors, pvals, fstat, preds, rsquared, yld_stacked)
    
def ClusterIncreaseSample(yld, climate_var, other_vars, cluster, k):
    cluster_sizes = []
    [n_t, n_lat, n_lon] = yld.shape
    
    # getting maximum cluster size to initialize objects
    for cl in range(1, k + 1):
        cluster_sizes.append(np.sum(cluster == cl))
    max_cluster_size = max(cluster_sizes)
    
    # initialíze objects
    const_stacked = np.empty([max_cluster_size * n_t, k])
    const_stacked.fill(1)
    yld_stacked = np.empty([max_cluster_size * n_t, k])
    yld_stacked.fill(np.nan)
    time_stacked = np.empty([max_cluster_size * n_t, k])
    time_stacked.fill(np.nan)
    if climate_var != "None":
        climate_stacked = np.empty([max_cluster_size * n_t, k])
        climate_stacked.fill(np.nan)    
    else:
        climate_stacked = "None"
    other_vars_stacked = np.empty([max_cluster_size * n_t, k])
    other_vars_stacked.fill(np.nan) 
    other_vars_stacked = [other_vars_stacked]*len(other_vars)
    
    # stack all cells of each cluster
    for cl in range(0, k):
        which_cell = 0
        time_tmp = np.tile(range(0, n_t), cluster_sizes[cl])
        time_stacked[0:(n_t*cluster_sizes[cl]), cl] = time_tmp
        for lat in range(0, n_lat):
            for lon in range(0, n_lon):
                if cluster[lat, lon] == (cl + 1):
                    yld_stacked[(which_cell*n_t) : ((which_cell+1)*n_t), cl] \
                                                             = yld[:, lat, lon]
                    if climate_var != "None":
                        climate_stacked[(which_cell*n_t) : \
                           ((which_cell+1)*n_t), cl] = climate_var[:, lat, lon]
                    for var in range(0, len(other_vars)):
                        other_vars_stacked[var][(which_cell*n_t) : \
                       ((which_cell+1)*n_t), cl] = other_vars[var][:, lat, lon]                  
                    which_cell = which_cell + 1
    return(yld_stacked, const_stacked, time_stacked, \
           climate_stacked, other_vars_stacked)    
    
def ClusterAverage(data, cl, k, mask):
    n_t = data.shape[0]
    res = np.empty([n_t, k])
    res.fill(np.nan)
    for t in range(0, n_t):
        for i in range(0, k):
            res[t, i] = np.nanmean(data[t, (cl == (i + 1)) & (mask == 1)])
    return(res) 

def PlotRegressionResults(regtype, yld, mask_crop, masks_di, indices, \
             other_vars, time, crop_name, other_vars_title, vars_filename, \
             subfolder, di_names, figsize, lats_WA, lons_WA, clusters = None, \
             model = "GEPIC", scatter = True, resid = True, rsq = True, \
             fstat = True):
    
    # layout of plots
    num_subplots = len(indices)
    cols = int(np.ceil(np.sqrt(num_subplots)))
    if cols * (cols - 1) >= num_subplots:
        rows = int(cols - 1)
    else:
        rows = int(cols)

    # initialize plots
    if scatter:
        figscatter = plt.figure(figsize = figsize)
        figscatter.subplots_adjust(bottom=0.07, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.2)
    if resid:
        figresid = plt.figure(figsize = figsize)
        figresid.subplots_adjust(bottom=0.07, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.2)
    if rsq:
        figrsq = plt.figure(figsize = figsize)
        figrsq.subplots_adjust(bottom=0.07, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.2)
    if fstat:
        figfstat = plt.figure(figsize = figsize)
        figfstat.subplots_adjust(bottom=0.07, top=0.9, left=0.1, right=0.9,
                    wspace=0.2, hspace=0.2)
    
    # regression and plotting
    for di in range(0, len(indices)):
        # regression
        if regtype == "cellwise":
            errors, pvals, f_stat, preds, r_sq, orig_data = \
                        LinearRegressionCells(yld, mask_crop*masks_di[di], \
                                          indices[di], other_vars, time = time)
        if regtype == "clusteraverage":
            errors, pvals, f_stat, preds, r_sq, orig_data = \
                        LinearRegressionClusterAverage(yld, clusters, \
                                        mask_crop*masks_di[di], indices[di], \
                                        other_vars, time = time)
        if regtype == "clustersample":
            errors, pvals, f_stat, preds, r_sq, orig_data = \
                        LinearRegressionClusterSample(yld, clusters, \
                                        mask_crop*masks_di[di], indices[di], \
                                        other_vars, time = time)
        # plots
        if scatter:
            ax_scatter = figscatter.add_subplot(rows, cols, di + 1)
            ax_scatter.scatter(orig_data.flatten(), \
                               preds.flatten(), marker = ".")
            ax_scatter.set_xlabel(model + " yield in t/ha")
            ax_scatter.set_ylabel("Yield values of regression in t/ha")
            ax_scatter.set_title(di_names[di] + other_vars_title + \
                                                 " as independent variable")
        if resid:
            ax_resid = figresid.add_subplot(rows, cols, di + 1)
            ax_resid.hist(errors.flatten(), density = True, \
                          alpha = 0.7, bins = 100)
            ax_resid.set_xlabel("Residuals of regression")
            ax_resid.set_ylabel("Density")
            ax_resid.set_title(di_names[di] + other_vars_title + \
                                                 " as independent variable")
        if rsq:
            ax_rsq = figrsq.add_subplot(rows, cols, di + 1)
            if regtype == "cellwise":
                c_rsq = MapValues(r_sq, lats_WA, lons_WA, \
                            title = di_names[di] +  other_vars_title + \
                            " as independent variable", \
                            vmin = 0, vmax = 1, ax = ax_rsq)
            else:
                c_rsq = MapValuesCluster(clusters, r_sq, lats_WA, lons_WA, \
                            title = di_names[di] +  other_vars_title + \
                            " as independent variable", \
                            vmin = 0, vmax = 1, ax = ax_rsq)
        if fstat:
            ax_fstat = figfstat.add_subplot(rows, cols, di + 1)
            if regtype == "cellwise":
                c_fstat = MapValues(f_stat, lats_WA, lons_WA, title = \
                               di_names[di] +  other_vars_title + \
                               " as independent variable", \
                               vmin = 0, vmax = 1, ax = ax_fstat)
            else: 
                c_fstat = MapValuesCluster(clusters, f_stat, lats_WA, \
                               lons_WA, title = di_names[di] + \
                               other_vars_title + " as independent variable", \
                               vmin = 0, vmax = 1, ax = ax_fstat)
    # colorbars
    if rsq:
        cb_ax = figrsq.add_axes([0.93, 0.2, 0.02, 0.6])
        figrsq.colorbar(c_rsq, cax = cb_ax)    
    if fstat:
        cb_ax = figfstat.add_axes([0.93, 0.2, 0.02, 0.6])
        figfstat.colorbar(c_fstat, cax = cb_ax)    
        
    # suptitles and saving plots
    if scatter:
        figscatter.suptitle(model + " yields vs. " + regtype + " regression" \
                            + " values of " +  crop_name + " yields")
        figscatter.savefig("Figures/LinearRegressions/" + subfolder + "/" + \
                "scatter_"+  model + "_" +  crop_name + "_" + \
               vars_filename + ".png", bbox_inches = "tight", pad_inches = 0.5)  
    if resid:
        figresid.suptitle("Distribution of residuals of " + regtype + " " + \
                  "regression of " + model + " " + crop_name + " yields")
        figresid.savefig("Figures/LinearRegressions/" + subfolder + "/" + \
               "residuals_" + model + "_" +  crop_name + "_" + \
              vars_filename + ".png", bbox_inches = "tight", pad_inches = 0.5)  
    if rsq:
        figrsq.suptitle("R squared of " + regtype + " regression of " + \
                                model + " " + crop_name + " yields")
        figrsq.savefig("Figures/LinearRegressions/" + subfolder + "/" + \
               "rsquared_" +  model + "_" +  crop_name + "_" + \
              vars_filename + ".png", bbox_inches = "tight", pad_inches = 0.5)  
    if fstat:
        figfstat.suptitle("F statistics of " + regtype + " regression of " +  \
                          model + " " +  crop_name + " yields")
        figfstat.savefig("Figures/LinearRegressions/" + subfolder + "/" + \
               "fstat_" + model + "_" +  crop_name + "_" + \
              vars_filename + ".png", bbox_inches = "tight", pad_inches = 0.5)  
    return()

def ReduceData(data, mask, quantile):
    res = data.copy()
    [n_t, n_lat, n_lon] = data.shape
    for lat in range(0, n_lat):
        for lon in range(0, n_lon):
            if mask[lat, lon] == 0:
                continue
            threshold = np.nanquantile(data[:, lat, lon], quantile)
            res[data[:, lat, lon] > threshold, lat, lon] = np.nan
    return(res)
    
def VariablesGDHY(crops, masks_yield, ylds):
    
    variables = []
    variables_detr = []
    masks = []
    
    with open("IntermediateResults/PreparedData/DroughtIndicators/" +\
                                 "spei03_WA_filled.txt", "rb") as fp:    
        variables.append(pickle.load(fp))    
    with open("IntermediateResults/PreparedData/DroughtIndicators/" +\
                                 "spei03_WA_detrend.txt", "rb") as fp:    
        variables_detr.append(pickle.load(fp))   
    with open("IntermediateResults/PreparedData/DroughtIndicators/" +\
                                 "mask_spei03_WA.txt", "rb") as fp:    
        masks.append(pickle.load(fp))    
        
    for other in ["WaterDeficit", "Precipitation", "PET", "DiurnalTemp"]:
        with open("IntermediateResults/PreparedData/CRU/" +\
                                 other + "03_WA.txt", "rb") as fp:    
            variables.append(pickle.load(fp))  
        with open("IntermediateResults/PreparedData/CRU/" +\
                                 other + "03_WA_detrend.txt", "rb") as fp:    
            variables_detr.append(pickle.load(fp))   
        with open("IntermediateResults/PreparedData/CRU/" +\
                                "mask_" + other + "_WA.txt", "rb") as fp:    
            masks.append(pickle.load(fp))
            
    variables_gs = []
    variables_detr_gs = []
    for cr in range(0, len(crops)):
        variables_gs_cr = []
        variables_detr_gs_cr = []
        
        with open("IntermediateResults/PreparedData/CropCalendar/" + \
                                         crops[cr] + "_plant.txt", "rb") as fp:    
            plant = pickle.load(fp)    
        pltmths = PltdayToPltmthsSingle(plant, masks_yield[cr])
        
        for var in range(0, len(variables)):
            variables_gs_cr.append(GetAnnualSingle(variables[var], pltmths, \
                   masks_yield[cr], 1982, "growing_season", ylds[cr]))
            variables_detr_gs_cr.append(GetAnnualSingle(variables_detr[var], \
                   pltmths, masks_yield[cr], 1982, "growing_season", ylds[cr]))
            
        variables_gs.append(variables_gs_cr)    
        variables_detr_gs.append(variables_detr_gs_cr)    
            
    return(variables_gs, variables_detr_gs, masks)
          

def DetrendClusterAvgGDHY(yields_avg, k, crops):
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
    
    
def BootstrapResiduals(avg_pred, residuals, constants, slopes, num_bt, \
                        k, crops):
    len_ts = avg_pred.shape[0]
    # initializing results
    bt_residuals = np.empty([num_bt, len_ts, len(crops), k])
    bt_residuals.fill(np.nan)
    bt_slopes = np.empty([num_bt, len(crops), k]); bt_slopes.fill(np.nan)
    bt_constants = np.empty([num_bt, len(crops), k]); bt_constants.fill(np.nan)
    # saving original values as first sample
    bt_residuals[0, :, :, :] = residuals
    bt_slopes[0, :, :] = slopes
    bt_constants[0, :, :] = constants
    
    # bootstrapping and regression per cluster and crop
    for cr in range(0, len(crops)):
        for cl in range(0, k):
            for i in range(1, num_bt):
                # bootstrapping
                rand_resid = np.random.choice(residuals[:, cr, cl], len_ts)
                # timeseries
                Y_bt = avg_pred[:, cr, cl] + rand_resid
                X = np.arange(0, len_ts).reshape((-1, 1))
                X = sm.add_constant(X)
                # regression
                model = sm.OLS(Y_bt, X, missing='drop')
                result = model.fit()
                # saving results
                bt_residuals[i, :, cr, cl] = Y_bt - result.predict(X)
                bt_constants[i, cr, cl] = result.params[0]
                bt_slopes[i, cr, cl] = result.params[1] 
    
    return(bt_residuals, bt_slopes, bt_constants)

  
    