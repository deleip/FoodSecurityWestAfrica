# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:39:18 2020

@author: leip
"""

# %% IMPORTING NECESSARY PACKAGES AND SETTING WORKING DIRECTORY

import numpy as np
from os import chdir
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from scipy.stats import shapiro
import matplotlib.cm as cm
import seaborn as sns

chdir('/home/debbora/IIASA/FinalVersion')

import OutsourcedFunctions_Analysis as OF
        
# setting which plots should be shown
visualize = "AllPlots"
#visualize = "ThesisPlots"

figsize = (24, 13.5)

# getting longitudes and latitudes of region
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                         "spei03_WA_filled.txt", "rb") as fp:
    spei03 = pickle.load(fp)   
    lats_WA = pickle.load(fp)   
    lons_WA = pickle.load(fp)   

###############################################################################
# %% Clustering

# as using all cells as single regions is comuptationally infeasible, but using
# all of West Africa as one region is oversimplifying and does not allow to 
# analyse the effect of correlation we devide the area into cluster according
# to climate data (drought index SPEI).


# %% 1) Calculate distances on which to base the clustering

with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                         "spei03_WA_detrend.txt", "rb") as fp:
    spei03detr = pickle.load(fp)    
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                         "spei03_WA_filled.txt", "rb") as fp:
    spei03 = pickle.load(fp)   
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "mask_spei03_WA.txt", "rb") as fp:    
    mask_spei03 = pickle.load(fp)    
    

with open("IntermediateResults/PreparedData/CRU/" + \
                                      "WaterDeficit03_WA.txt", "rb") as fp:
    wd03 = pickle.load(fp)   
with open("IntermediateResults/PreparedData/CRU/" + \
                                      "mask_WaterDeficit_WA.txt", "rb") as fp:    
    mask_wd03 = pickle.load(fp)     
   
# Pearson corr and dist using full 3-month SPEI dataset
OF.CalcPearson(spei03, mask_spei03, "spei03")     
# Pearson corr and dist using full detrended 3-month SPEI dataset
OF.CalcPearson(spei03detr, mask_spei03, "spei03detr")
# Pearson corr and dist using the last 30 years of 3-month SPEI dataset
OF.CalcPearson(spei03[-(30*12):,:,:], mask_spei03, "spei03_30y")
# Pearson corr and dist using last 30 years of detrended 3-month SPEI dataset
OF.CalcPearson(spei03detr[-(30*12):,:,:], mask_spei03, "spei03detr_30y")
# Pearson corr and dist using 3-month SPEI dataset. For each pair of cells 
# only months with at least one cell below cut value are used
OF.CalcPearson(spei03, mask_spei03, "spei03", cut = -1)
OF.CalcPearson(spei03, mask_spei03, "spei03", cut = -2)
# Pearson corr and dist using a boolean 3-month SPEI dataset, just showing
# in which months each cell had an extreme event (defined by cut value)
OF.CalcPearson(spei03, mask_spei03, "spei03", cut = -1, boolean = True)
OF.CalcPearson(spei03, mask_spei03, "spei03", cut = -2, boolean = True)
# Pearson corr and dist using full 3-month water deficit dataset
OF.CalcPearson(wd03, mask_wd03, "wd03")
# Jaccard dist using boolean 3-month SPEI dataset, cut value -1
OF.CalcPearson(spei03, mask_spei03, "spei03", DistType = "Jaccard", \
                                                   cut = -1, boolean = True)    
# Dice dist using boolean 3-month SPEI dataset, cut value -1
OF.CalcPearson(spei03, mask_spei03, "spei03", DistType = "Dice", \
                                                   cut = -1, boolean = True)     

# %% 2) Running k-Mediods for different distances
# for k = 4, 8, 12 

DistFiles = ["PearsonDist_spei03",
             "PearsonDist_spei03detr", 
             "PearsonDist_spei03_30y",
             "PearsonDist_spei03detr_30y",
             "PearsonDist_spei03_CutNeg1", 
             "PearsonDist_spei03_CutNeg2",
             "PearsonDist_spei03_Boolean_CutNeg1", 
             "PearsonDist_spei03_Boolean_CutNeg2",
             "DiceDist_spei03_Boolean_CutNeg1",
             "JaccardDist_spei03_Boolean_CutNeg1",
             "PearsonDist_wd03"]

with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "mask_spei03_WA.txt", "rb") as fp:    
    mask_spei03 = pickle.load(fp)  
    
for file in range(0, len(DistFiles)-1):
    with open("IntermediateResults/Clustering/Distances/" + \
                                      DistFiles[file] + ".txt", "rb") as fp:    
        dist = pickle.load(fp)    
    for k in [4, 8, 12]:
        OF.kMedoids(k, dist, mask_spei03, DistFiles[file])
   
with open("IntermediateResults/PreparedData/CRU/" + \
                                     "mask_WaterDeficit_WA.txt", "rb") as fp:    
    mask_wd03 = pickle.load(fp)      
with open("IntermediateResults/Clustering/Distances/" + \
                                  DistFiles[-1] + ".txt", "rb") as fp:    
    dist = pickle.load(fp)    
for k in [4, 8, 12]:
    OF.kMedoids(k, dist, mask_wd03, DistFiles[-1])

# %% 3) Running k-Mediods for different seeds (PearsonDist_spei03)

#with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
#                                          "mask_spei03_WA.txt", "rb") as fp:    
#    mask_spei03 = pickle.load(fp)  
#with open("IntermediateResults/Clustering/Distances/" + \
#                                  "PearsonDist_spei03.txt", "rb") as fp:    
#    dist = pickle.load(fp)  
#
#for s in range(0, 4):
#    OF.kMedoids(8, dist, mask_spei03, "PearsonDist_spei03", seed = s)
    
# %% 4) Running k-Mediods for a bigger number of k (PearsonDist_spei03)
#       (and with different seeds to then choose best version)
    
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "mask_spei03_WA.txt", "rb") as fp:    
    mask_spei03 = pickle.load(fp)  
with open("IntermediateResults/Clustering/Distances/" + \
                                  "PearsonDist_spei03.txt", "rb") as fp:    
    dist = pickle.load(fp)    
    

ks = list(range(1, 21)); ks.remove(4); ks.remove(8); ks.remove(12)
for k in [19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 7, 6, 5, 3, 2, 1]:
    print("k = " + str(k))
    OF.kMedoids(k, dist, mask_spei03, "PearsonDist_spei03")
    
#ks = list(range(1, 21)); ks.remove(8)
#for k in ks:
#    print("k = " + str(k))
#    for s in range(0, 4):
#        print("seed = " + str(s))
#        OF.kMedoids(k, dist, mask_spei03, "PearsonDist_spei03", seed = s)
   
## saving best version as optimal clustering
#for k in range(1, 21):
#    with open("IntermediateResults/Clustering/Clusters/kMediods" + str(k) + \
#                                  "_PearsonDist_spei03.txt", "rb") as fp:  
#        clusters = pickle.load(fp)
#        costs = pickle.load(fp)
#        medoids = pickle.load(fp)
#    min_costs = costs[-1]
#    with open("IntermediateResults/Clustering/Clusters/kMediods" + str(k) + \
#                               "opt_PearsonDist_spei03.txt", "wb") as fp:  
#        pickle.dump(clusters, fp)
#        pickle.dump(costs, fp)
#        pickle.dump(medoids, fp)
#        pickle.dump(3052020, fp)
#    for s in range(0, 4):
#        with open("IntermediateResults/Clustering/Clusters/kMediods" + \
#                             str(k) + "_PearsonDist_spei03_seed" + \
#                                                  str(s) + ".txt", "rb") as fp:  
#            clusters = pickle.load(fp)
#            costs = pickle.load(fp)
#            medoids = pickle.load(fp)
#        if costs[-1] < min_costs:
#            min_costs = costs[-1]
#            with open("IntermediateResults/Clustering/Clusters/kMediods" + \
#                        str(k) + "opt_PearsonDist_spei03.txt", "wb") as fp:  
#                pickle.dump(clusters, fp)
#                pickle.dump(costs, fp)
#                pickle.dump(medoids, fp)
#                pickle.dump(s, fp)
    
# %% 4) Analysis of clustering
                
# - Comparing SPEI and wd data
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                         "spei03_WA_filled.txt", "rb") as fp:
    spei = pickle.load(fp)   
with open("IntermediateResults/PreparedData/CRU/" + \
                                         "WaterDeficit03_WA.txt", "rb") as fp:
    wd = pickle.load(fp)       

fig = plt.figure(figsize=figsize)
plt.scatter(wd, spei, marker = ".")
plt.xlabel("3-month water deficit (precipitation - PET)")
plt.ylabel("3-month SPEI")
plt.title("Scatter plot of water deficit and SPEI from " + \
                                             "1901 to 2018 for all grid cells")
fig.savefig("Figures/Clustering/Scatter_SPEI_wd.png",
                        bbox_inches = "tight", pad_inches = 0.5)  

# - Visualizing CLuster  
DistFiles = ["PearsonDist_spei03",
             "PearsonDist_spei03detr", 
             "PearsonDist_spei03_30y",
             "PearsonDist_spei03_CutNeg1", 
             "PearsonDist_spei03_CutNeg2",
             "PearsonDist_spei03_Boolean_CutNeg1",
             "DiceDist_spei03_Boolean_CutNeg1",
             "JaccardDist_spei03_Boolean_CutNeg1",
             "PearsonDist_wd03"]
        
for k in [4, 8, 12]:
    fig = plt.figure(k, figsize=figsize)
    fig.subplots_adjust(bottom=0.03, top=0.92, left=0.1, right=0.9,
                    wspace=0.25, hspace=0.1)
    for file in range(0,len(DistFiles)):
        with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                  str(k) + "_" + DistFiles[file] + ".txt", "rb") as fp:    
            clusters = pickle.load(fp)
            costs = pickle.load(fp)
            medoids = pickle.load(fp)
        fig.add_subplot(3,3,file+1)
        c = OF.VisualizeFinalCluster(clusters, medoids, lats_WA, lons_WA, \
                                     DistFiles[file])
            
    cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(c, cax = cb_ax)       
    cbar.set_ticks(range(1, k + 1))
    cbar.set_ticklabels(range(1, k + 1))
    plt.suptitle("Clusters for k = "+ str(k) + \
                 " using diffferent distances", fontsize = 24)
    fig.savefig("Figures/Clustering/kMediods" + str(k) + ".png",
                            bbox_inches = "tight", pad_inches = 0.5)    


# - Visualizing correlation within cluster (cells to respective medoid)
corr = []
with open("IntermediateResults/Clustering/Distances/" + \
                                "PearsonCorr_spei03.txt", "rb") as fp:    
    corr.append(pickle.load(fp))
with open("IntermediateResults/Clustering/Distances/" + \
                                "PearsonCorr_spei03_CutNeg1.txt", "rb") as fp:    
    corr.append(pickle.load(fp))
with open("IntermediateResults/Clustering/Distances/" + \
                                "PearsonCorr_wd03.txt", "rb") as fp:    
    corr.append(pickle.load(fp))

Dists = ["PearsonDist_spei03",
         "PearsonDist_spei03_CutNeg1",
         "PearsonDist_wd03"]

fig = plt.figure(figsize=figsize)
fig.subplots_adjust(bottom=0.03, top=0.97, left=0.1, right=0.9,
                wspace=0.25, hspace=-0.2)
for k in [4, 8, 12]:
    for file in range(0, len(Dists)):
        fig.add_subplot(3, 3, k/4 + file*3)
        with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                            str(k) + "_" + Dists[file] + ".txt", "rb") as fp:    
            cluster = pickle.load(fp)
            cost = pickle.load(fp)
            medoids = pickle.load(fp)
        c = OF.PlotDistToMediod(cluster, medoids, corr[file], Dists[file] + \
                    ", k = " + str(k), lats_WA, lons_WA, show_medoids = True)
cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = fig.colorbar(c, cax = cb_ax)    
fig.suptitle('Visualization of Pearson Correlation to respective medoids', \
                                                                fontsize = 24)
fig.savefig("Figures/Clustering/kMediods_CorrToRespectiveMedoid.png", \
                            bbox_inches = "tight", pad_inches = 0.5)

# spei03_CutNeg1 shows very low correlations, wd03 very high, while spei03
# is in the middle. This seems to be the best way, as we want to have low
# correlation between different clusters, i.e. correlation within clusters
# has to decrease towards the edges.

# - Visualizing distance between cluster (cells to medoids)
Dists = ["PearsonDist_spei03",
         "PearsonDist_spei03_CutNeg1",
         "PearsonDist_wd03"]
k = 4
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(bottom=0.07, top=0.85, left=0.1, right=0.9,
                wspace=0.3, hspace=0.6)
for f in range(0, len(Dists)):
    with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                        str(k) + "_" + Dists[f] + ".txt", "rb") as fp:  
        cluster = pickle.load(fp)
        cost = pickle.load(fp)
        medoids = pickle.load(fp)  
    with open("IntermediateResults/Clustering/Distances/" + \
                                   Dists[f] + ".txt", "rb") as fp: 
        dist = pickle.load(fp)
    for cl in range(1, k + 1):
        fig.add_subplot(k, len(Dists), (cl - 1) * len(Dists) + f + 1)
        OF.HistogramDistCluster(cluster, medoids, dist, cl, 3, Dists[f])
        if f == 0:
            plt.ylabel("Denisty of distribution")
    fig.suptitle("Visualization of distance from cells of a cluster " + \
                     "to different medoids", fontsize = 24)
    fig.savefig("Figures/Clustering/kMediods_DistToOtherMedoids.png", \
                bbox_inches = "tight", pad_inches = 0.5)

# Different distances to see how clusters can be influenced. Reducing datasets
# to extreme events to see if the pattern is very different. Looking at past
# 30 years to see if pattern changes over time.
# However, SPEI seems the most fitting data (compared to water deficit) as it
# takes into account the "normal" climate at a location, and is one of the
# "standard" drought indices. Clusters are supposed to reflect the correlation
# of extreme events, for which SPEI is more fitting. For yield regressions 
# (which don't focus on extreme events) wd-clusters could be working better. 
# An idea for the future would be to subcluster the SPEI-clusters (which are 
# needed for the dependence structure) according to wd data, and each of these
# subclusters gets its onw regression. For now we only use the SPEI clusters.
#  Even though we see a slight change when reducing
# the dataset to the last 30 years, using the all years gives a much bigger
# dataset to calculate the distances from, and also we will not change clusters
# in our projection over time, so it makes sense to use an "average" clustering
# over time. Boolean datasets (for Pearson distance as well as for both the
# boolean-specific distances Jaccard and Dice) give more information about
# tail dependence, but discard data about the strength of drouhgts and make 
# less sense keeping in mind that out model asumes full dependence within 
# (by using average yield values) and independence between different clusters 
# both for extreme events and normal years. 
# Our main interest is therefore to have high dependence within a cluster and 
# low dependennce between different cluster, which we analyze by looking at the
# Pearson Correlation Coeffieient.
# Going on we therefore use the complete spei03 dataset for clustering. We
# use the detrended version, as we want correlation to focus on the drought 
# patterns and not a linear trend present in both cells.
    
# - Visualizing influence of seeds
#    
#fig = plt.figure(figsize=figsize)
#fig.subplots_adjust(bottom=0.03, top=0.97, left=0.1, right=0.9,
#                wspace=0.25, hspace=-0.2)
#k = 8
#for s in range(0,4):
#    with open("IntermediateResults/Clustering/Clusters/kMediods" + str(k) + \
#             "_PearsonDist_spei03_seed" + str(s) + ".txt", "rb") as fp:    
#        clusters = pickle.load(fp)
#        costs = pickle.load(fp)
#        medoids = pickle.load(fp)
#    fig.add_subplot(2, 2, s+1)
#    c = OF.VisualizeFinalCluster(clusters, medoids, lats_WA, lons_WA, \
#                                             "Seed " + str(s))   
#cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
#cbar = fig.colorbar(c, cax = cb_ax)       
#cbar.set_ticks(range(1, k + 1))
#cbar.set_ticklabels(range(1, k + 1))
#plt.suptitle("Clusters for k = 8, PearsonDist_spei03 " + \
#                                 "with different seeds", fontsize = 24)
#fig.savefig("Figures/Clustering/kMediods_Seeds.png",
#                        bbox_inches = "tight", pad_inches = 0.5)    

# Depending on the randomly chosen initial medoids, the final clusters can 
# vary. In order to save computational time, we first worked with just one run
# of k-Medoids, but after deciding on PearsonDist_spei03detr as distance and 
# these cases were rerun with several different seeds and the version with
# lowest final costs was chosen as clustering

# %% 5) Optimal number of clusters

with open("IntermediateResults/Clustering/Distances/" + \
                                "PearsonDist_spei03.txt", "rb") as fp:    
    dist = pickle.load(fp) 


# a) Elbow-Method using Davies-Bouldin-Index: 
#    The elbow method tried to find the number of clusters after which the 
#    marginal gain of increasing the number of clusters drops, given by a dent 
#    in a graph showing som sort of cost function for each clustering. 
#    An index combining both inter-cluster similarity and intra-cluster 
#    dissimilarity is the Davies-Bouldin-Index.
    
DB = []
final_costs = []
for k in range(2,21): 
    with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                      str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)  
    DB.append(OF.DaviesBouldinIndex(clusters, medoids, dist))
    final_costs.append(costs[-1])
    
fig = plt.figure(figsize=figsize)
plt.plot(range(2,21), DB)
plt.ylabel("Davies-Bouldin-Index", fontsize = 20)
plt.xlabel("Number of clusters", fontsize = 20)
plt.title("Davies-Bouldin-Index", fontsize = 24)
plt.xticks(np.arange(2, 21, 2))
fig.savefig("Figures/Clustering/kMediods_DaviesBouldinIndex.png",
                                    bbox_inches = "tight", pad_inches = 0.5)    
# As the graph is not falling monotonously, the heighest elbow strengths are
# given to cluster numbers where the index increases, which is not the intended
# outcome...

# b) Comparing similarity within and between cluster on a scatter plot to find
#    the best trade-off. 
#    We look at the average correltaion of cells to the respective cluster 
#    medoid, compared to the average correlation between different medoids. 
#    When including all medoid combinations in the latter, the undesired effect  
#    of more similarity to neghboring clusters might be cancelld out by the  
#    higher number of far-away clusters which are uncorrelated. Therefore we  
#    then take only the closest (using Pearson Distance) neghboring cluster 
#    into account.

with open("IntermediateResults/Clustering/Distances/" + \
                                "PearsonDist_spei03.txt", "rb") as fp:    
    dist = pickle.load(fp) 

between_all = []
between_closest = []
within_cluster = []
kmax = 20

# visualize clusters and save distances
fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.03, top=0.97, left=0.1, right=0.9,
                wspace=0.25, hspace=-0.2)
for k in range(2, kmax + 1):
    with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                      str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    if k%2 == 0:
        fig.add_subplot(3,4, k/2)
        c = OF.VisualizeFinalCluster(clusters, medoids, \
                                        lats_WA, lons_WA, "k = " + str(k))

    all_dists, closest_dist = OF.MedoidMedoidDistd(medoids, dist)
    between_all.append(np.nanmean(all_dists))
    between_closest.append(np.nanmean(closest_dist))
    within_cluster.append(costs[-1]/(np.sum(~np.isnan(clusters[-1]))))

plt.suptitle("Visualization of clusters for different k")
fig.savefig("Figures/Clustering/VisualizationClusters.png",
                                    bbox_inches = "tight", pad_inches = 0.5)    

dists_between = [between_all, between_closest]
dists_within = [within_cluster, within_cluster]
title = ["Comparison using all clusters for raw SPEI", \
         "Comparison using closest clusters for raw SPEI"]

# plot distances
version = ["All", "Closest"]
for i in range(0,2):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,2,1) 
    ax.scatter(dists_within[i], dists_between[i],  c=range(2, kmax + 1))
    plt.title(title[i])
#    plt.xlim([0, 0.3])
#    plt.ylim([0.4, 0.7])
    plt.xlabel("Average distance within clusters")
    plt.ylabel("Average distance between clusters")
    for t, txt in enumerate(range(2, kmax + 1)):
        ax.annotate(txt, (dists_within[i][t] + 0.0013, \
                          dists_between[i][t] - 0.001)) 
    fig.add_subplot(1,2,2) 
    metric, cl_order = OF.MetricClustering(dists_within[i], dists_between[i], \
                                        refX = 0, refY = max(dists_between[i])) 
    plt.scatter(cl_order, metric)
    plt.xticks(range(2, 21))
    plt.title("Quantification of tradeoff")
    plt.xlabel("Number of clusters")
    plt.ylabel("Euclidean distance to (0, "+ \
                                  str(np.round(max(dists_between[i]), 2)) + \
                                  ") on scatter plot of distances")
    plt.suptitle("Tradeoff of distances within and between cluster")
    
    fig.savefig("Figures/Clustering/kMediods_ScatterInterVsIntraCluster" + \
                 version[i] + ".png", bbox_inches = "tight", pad_inches = 0.5)    

# for between_all the best k are the heighest k
# for between_closest the best k are: 2, 3, 8
        

# Problem: for few clusters they seem far apart, as the medoid is further away
#         because it is in the middle of the cluster. But there are still many 
#         close cells. Therefore we should do something depending on the cells!

# Idea: As distance between different clusters use dist of all cells of cluster 
#       A to medoid B + dist of all cells of cluster B to medoid A, devided by
#       the total amount of cells in cluster A and B (commented out as it 
#       worked less well with the euclidean distance as metric)
    
    
# c) Other idea: we want cells to behave similar within a cluster, i.e. small
#   variance. Hence similarity within cluster could be measured by varaince, 
#   and then compared to variance including cells of both cluster. As variance 
#   over time does not make sese, we would calculate variance per year and then
#   take the average over the timeseries to get a single value.
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                         "spei03_WA_filled.txt", "rb") as fp:
    spei = pickle.load(fp)   

between_all = []
between_closest = []
within_cluster = []
kmax = 20

# save distances
for k in range(2, kmax + 1):
    with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                      str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)

    m_within, m_closest, m_all = OF.ClusterMetricVariance(spei, clusters)
    between_all.append(np.nanmean(m_all))
    between_closest.append(np.nanmean(m_closest))
    within_cluster.append(np.nanmean(m_within))   

dists_between = [between_all, between_closest]
dists_within = [within_cluster, within_cluster]
title = ["Comparison using all clusters for raw SPEI", \
         "Comparison using closest clusters for raw SPEI"]

# plot distances
version = ["All", "Closest"]
for i in range(0,2):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,2,1) 
    ax.scatter(dists_within[i], dists_between[i],  c=range(2, kmax + 1))
    plt.title(title[i])
#    plt.xlim([0, 0.3])
#    plt.ylim([0.4, 0.7])
    plt.xlabel("Average distance within clusters")
    plt.ylabel("Average distance between clusters")
    for t, txt in enumerate(range(2, kmax + 1)):
        ax.annotate(txt, (dists_within[i][t] + 0.0013, \
                          dists_between[i][t] - 0.001)) 
    fig.add_subplot(1,2,2) 
    metric, cl_order = OF.MetricClustering(dists_within[i], dists_between[i], \
                                                               w0 = 1, w1 = 1) 
    plt.scatter(cl_order, metric)
    plt.xticks(range(2, 21))
    plt.title("Quantification of tradeoff")
    plt.xlabel("Number of clusters")
    plt.ylabel("Euclidean distance to (0, 0.7) on scatter plot of distances")
    plt.suptitle("Tradeoff of distances within and between cluster", \
                                                             fontsize = 24)
    
# doesn't work, gives linear plot...
    
# %% Analysis of relation between drought index and yield data

# we hoped to find an "easy" regression between a drought index and yield data
# in order to use the distribution of a drought index to sample yield

# 1) Using model output from the GGCMI phase 1 of the AgMIP project:
#    14 different GGCMS, 11 different climate datasets, 3 settings (default,
#    fullharm, noharm), two scenarios (full irrigation, no irrigation)
#    different output variables depending on model
#    We decided to focus on default settings (as these are the ones the models
#    were originally calibrated to, and therefore should give the best yield
#    outputs), and no irrigation (as irrigation is very rare in West Africa).

# a) Analysis of AgMIP yield data

# - are yield outputs from different models comparabel? (visual)
#   We focus on EPCI-IIASA with AgMERRA (1980-2010) dataset, EPIC-BOKU with 
#   grasp dataset (1961-2010), GEPIC with pgfv2 dataset (1948-2008) and 
#   pAPSIM with AgMERRA dataset (1980-2010).

models = ["epic-iiasa_agmerra_default", \
         "epic-boku_grasp_default", \
         "gepic_pgfv2_default", \
         "papsim_agmerra_default"]
crop_scenarios = ["_yield_whe_noirr", "_yield_mai_noirr", \
         "_yield_soy_noirr", "_yield_ric_noirr"]
crops = ["wheat", "maize", "soy", "rice"]

yield_data = []  # yield_data[model][crop]
# reducing all timeseries to 1980-2008
for m in range(0, len(models)):
    yield_data_tmp = []
    for cr in range(0, len(crop_scenarios)):
        if (m == 3) & (cr == 3):    # papsim does not have data for rice
            continue
        with open("IntermediateResults/PreparedData/AgMIP/" + \
                       models[m] + crop_scenarios[cr] + ".txt", "rb") as fp: 
            yields = pickle.load(fp)
            if (m == 0) or (m == 3):
                yields = yields[:(-2),:,:]
            if m == 1:
                yields = yields[19:(-2),:,:]
            if m == 2:
                yields = yields[32:,:,:]
            yield_data_tmp.append(yields)  
    yield_data.append(yield_data_tmp)

combinations = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
for cr in range(0, len(crops)):
    fig = plt.figure(figsize=figsize)
    if cr < 3:
        for c in range(0, len(combinations)):
            fig.add_subplot(2,3,c+1)
            plt.scatter(yield_data[combinations[c][0]][cr], \
                        yield_data[combinations[c][1]][cr], marker = ".")
            plt.xlabel("Yield in t/ha, " + models[combinations[c][0]])
            plt.ylabel("Yield in t/ha, " + models[combinations[c][1]])
        plt.suptitle("Comparison of different yield models for " + \
                                             crops[cr] + " (no irrigation)")  
        fig.savefig("Figures/LinearRegressions/AgMIP/YieldsScatter_" + \
                   crops[cr] + ".png", bbox_inches = "tight", pad_inches = 0.5)    
    else:
        for c in range(0, len(combinations)-3):
            fig.add_subplot(2,3,c+1)
            plt.scatter(yield_data[combinations[c][0]][cr], \
                        yield_data[combinations[c][1]][cr], marker = ".")
            plt.xlabel("Yield in t/ha, " + models[combinations[c][0]])
            plt.ylabel("Yield in t/ha, " + models[combinations[c][1]])
        plt.suptitle("Comparison of different yield models for " + \
                                             crops[cr] + " (no irrigation)")  
        fig.savefig("Figures/LinearRegressions/AgMIP/YieldsScatter_" + \
                   crops[cr] + ".png", bbox_inches = "tight", pad_inches = 0.5)    
        
# There is no visual relationship between yield outputs of different models.
# Each model seems to strongly depend on its own setups and calibration. 
# Nevertheless, we hope that by including other output variables from the 
# respective models as independent variables alonside the drought indicators 
# in the regression analysis, we could get a significant relationship. Even if 
# the yields are not completely accurate, this would give the relationship 
# needed for the proof of conept of the stochastic model. 
        
# - Visualization of data of chosen model
# For the first part of the regression analysis, we focus on GEPIC, as it 
# covers the longest time period, includes fertilizer application as 
# output variable, and covers the whole region of West Africa without missing
# cells.
        
startyear_gepic = 1948

yields_gepic = []
for cr in crop_scenarios:
    with open("IntermediateResults/PreparedData/AgMIP/" + \
                  "gepic_pgfv2_default" + cr + ".txt", "rb") as fp: 
        yields_gepic.append(pickle.load(fp))
        
time = [0, 20, 40, 59]
fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.07, top=0.9, left=0.1, right=0.9,
                wspace=0.2, hspace=0.2)
for t in range(0, len(time)):
    for cr in range(0, len(crops)):
        fig.add_subplot(4, 4, (t*4) + cr + 1)
        if t == 0:
            title = crops[cr]
        else:
            title = ""
        c = OF.MapValues(yields_gepic[cr][time[t],:,:], lats_WA, lons_WA, \
                                         title = title, vmin = 0, vmax = 5.5)
        if cr == 0:
            plt.ylabel(str(startyear_gepic + time[t]))
cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = fig.colorbar(c, cax = cb_ax)    
plt.suptitle("Yields given by " + models[2] + " (no irrigation)", fontsize=24)
fig.savefig("Figures/LinearRegressions/AgMIP/yields_" + models[2] + \
                "_noirr_.png", bbox_inches = "tight", pad_inches = 0.5)   


# b) Loading GEPIC data and drought indices for regressions

with open("IntermediateResults/PreparedData/CRU/" + \
                                     "mask_WaterDeficit_WA.txt", "rb") as fp:    
    mask_wd = pickle.load(fp) 
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "mask_spei03_WA.txt", "rb") as fp:    
    mask_spei = pickle.load(fp)  

model = "gepic"
climate = "pgfv2"
harm = "default" 
irri = "noirr"
var_names = ["yield", "plant-day", "pirrww" ,"initr", "gsprcp"]
year_start = 1948
crops = ["whe", "mai", "soy", "ric"]
crop_names = ["wheat", "maize", "soy", "rice"]

variables = []
masks_yield = []
variables_detr = []
masks_detr = []
indices = []
indices_detr = []

detrend_yield = False
detrend_vars = False
for crop in crops:
    # without detrending
    var, mask, spei_annual_gs, spei_annual_lowest, \
    wd_annual_gs, wd_annual_lowest = OF.RegressionDataAgMIP(model, climate, \
                                       harm, irri, crop, var_names, \
                                       False, year_start, False)
    variables.append(var)
    masks_yield.append(mask)
    indices.append([spei_annual_gs, spei_annual_lowest, \
                            wd_annual_gs, wd_annual_lowest])
    # with detrending
    var, mask, spei_annual_gs, spei_annual_lowest, \
    wd_annual_gs, wd_annual_lowest = OF.RegressionDataAgMIP(model, climate, \
                                       harm, irri, crop, var_names, \
                                       True, year_start, True)
    variables_detr.append(var)
    indices_detr.append([spei_annual_gs, spei_annual_lowest, \
                            wd_annual_gs, wd_annual_lowest])

di_names = ["SPEI in growing season", "Lowest SPEI value of the year", \
            "prec - PET in growing season", \
            "Lowest prec - PET value of the year"]
di_names_detr = ["Detr. SPEI in growing season", \
                 "Lowest detr. SPEI value of the year", \
                 "Detr. prec - PET value in growing season", \
                 "Lowest detr. prec - PET value of the year"]
masks_di = [mask_spei, mask_spei, mask_wd, mask_wd]
no_di = ["None", "None", "None", "None"] 
no_di_names = ["", "", "", ""]

with open("IntermediateResults/LinearRegression/" + \
                              "data_GEPIC_regressions.txt", "wb") as fp:    
    pickle.dump([variables, variables_detr, masks_yield, masks_di, \
                 indices, indices_detr, di_names, di_names_detr, \
                 no_di, no_di_names], fp)
     
# c) Different Regression combinations

with open("IntermediateResults/LinearRegression/" + \
                          "data_GEPIC_regressions.txt", "rb") as fp:    
    [variables, variables_detr, masks_yield, masks_di, \
     indices, indices_detr, di_names, di_names_detr, \
     no_di, no_di_names] = pickle.load(fp)
    
crops = ["whe", "mai", "soy", "ric"]
crop_names = ["wheat", "maize", "soy", "rice"]

# - Only drought index as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = indices[cr], other_vars = [], time = False, \
            crop_name = crop_names[cr], other_vars_title = "", \
            vars_filename = "di", subfolder = "AgMIP", di_names = di_names, \
            figsize = figsize, lats_WA = lats_WA, lons_WA = lons_WA, \
            model = "GEPIC", scatter = True, resid = True, rsq = True, \
            fstat = True)
    
# - Drought index and GEPIC output (pirrww, initr, gsprcp) as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = indices[cr], other_vars = variables[cr][2:], \
            time = False, crop_name = crop_names[cr], \
            other_vars_title = ", pirrww, initr and gsprcp", \
            vars_filename = "di_all", subfolder = "AgMIP", \
            di_names = di_names, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, model = "GEPIC", scatter = True, resid = True, \
            rsq = True, fstat = True)
    
# - Only GEPIC output (pirrww, initr, gsprcp) as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = no_di, other_vars = variables[cr][2:], time = False, \
            crop_name = crop_names[cr], \
            other_vars_title = "pirrww, initr and gsprcp", \
            vars_filename = "nodi_all", subfolder = "AgMIP", \
            di_names = no_di_names, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, model = "GEPIC", scatter = True, resid = True, \
            rsq = True, fstat = True)
    
 # - detr drought index and time as independent vars
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = indices_detr[cr], other_vars = [], time = True, \
            crop_name = crop_names[cr], other_vars_title = " and time", \
            vars_filename = "di_detr", subfolder = "AgMIP", \
            di_names = di_names_detr, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, model = "GEPIC", scatter = True, resid = True, \
            rsq = True, fstat = True)

# - Detr drought index and GEPIC output as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = indices_detr[cr], other_vars = variables_detr[cr][2:], \
            time = True, crop_name = crop_names[cr], \
            other_vars_title = ", detr. pirrww, initr, gsprcp and time", \
            vars_filename = "di_all_detr", subfolder = "AgMIP", \
            di_names = di_names_detr, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, model = "GEPIC", scatter = True, resid = True, \
            rsq = True, fstat = True)
    
# - Detr. GEPIC output (pirrww, initr, gsprcp) as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = no_di, other_vars = variables[cr][2:], time = True, \
            crop_name = crop_names[cr], \
            other_vars_title = "pirrww, initr, gsprcp and time", \
            vars_filename = "nodi_all_detr", subfolder = "AgMIP", \
            di_names = no_di_names, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, model = "GEPIC", scatter = True, resid = True, \
            rsq = True, fstat = True)
    
# Questions: 
# - is growing season or lowest yearly value better?
# - is water deficit or SPEI better?
# - can drought index improve regression from GEPIC outputs?
# - does any of the drought indices give a good regression? 
# - does detrending improve regressions? does it change any of the answers 
#   to the other questions?
# Values to use:
# - Scatter plot of original values and predictions as first visual impression
# - r suared of regression (explained variance)
# - Histogramm of residuals to check if normally distributed. All of them are,
#   therefore we can use:
# - pvalues of f-statistics of regression (regression significant?)
# Results:
# - drought indices for growing season give better r^2 values in some cases 
#   (no difference in others, lowest values is never better), and adding 
#   drought indices slightly improves regressions from only using GEPIC output 
#   variable. SPEI and prec - PET give very similar results.
# - in general the r^2 values are very low, best r^2 values (sometimes even 
#   quite high) only for the North of the region, probably because this alredy 
#   is desert-like and has consistently low yield values which are easy
#   to predict. The rest of the area has mostly values below 0.2
# - detrending slightly improves r^2 values
# - pvalues are low (i.e. regression significant) for the Northern areas 
#   (fitting to the good r^2 values), and seem random in other cells (i.e. 
#   some cells low some high wihtout spatial pattern)
# - exception: wheat gives low pvalues and higher r^2 values for a bigger area 
#   using drought index and GEPIC output variables (without detrending a bit 
#   better than with detrending).
    
# We will check if reducing timeseries to points with "extreme" values of the
# drought indices can imporve the regressions, as it would make sense that the
# relation to yield values is stronger in these years. In "normal" years the
# relation might be mutes by other effects we can't control for.
# As we will use clusters in the stochastic model, we also try two other
# spatial regression types: 
# 1) Using clusters to increase sample size (as we assume that within a cluster
#    the relation between yields and independent variables is the same in all 
#    cells)
# 2) Using cluster average timeseries for regression, as we might use a single
#    value per cell in the stochastic model anyway...
    
# Due to the results of the cellwise regression, we will use drought index 
# values of the growing season. 
    
# - Reduced drought index and GEPIC output as independent var
quantile = 0.5  # only using drier half of data
indices_red = []
for cr in range(0, len(crops)):
    indices_red_cr = []; indices_red_cr.append(indices[cr][0])
    indices_red_cr.append(OF.ReduceData(indices[cr][0], mask_spei,quantile))
    indices_red_cr.append(indices[cr][2])
    indices_red_cr.append(OF.ReduceData(indices[cr][2], mask_wd, quantile))
    indices_red.append(indices_red_cr)
di_names_red = ["SPEI in growing season", \
            "SPEI in growing season (lower 50%)", \
            "prec - PET in growing season", \
            "prec - PET in growing season (lower 50%)"]

for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = variables[cr][0], \
            mask_crop = masks_yield[cr], masks_di = masks_di, \
            indices = indices_red[cr], other_vars = variables[cr][2:], \
            time = False, crop_name = crop_names[cr], \
            other_vars_title = ", pirrww, initr and gsprcp", \
            vars_filename = "di_all_red", subfolder = "AgMIP", \
            di_names = di_names_red, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, clusters = None, model = "GEPIC", \
            scatter = True, resid = True, rsq = True, fstat = True)

# No big improvement. However, this also reduced the timeseries to only 30 
# years, which could be part of the reason. Therefore we next try with incresed
# samplesize by clustering
with open("IntermediateResults/Clustering/Clusters/kMediods3" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clustersample", \
            yld = variables[cr][0], mask_crop = masks_yield[cr], \
            masks_di = masks_di, indices = indices_red[cr], \
            other_vars = variables[cr][2:], time = False, \
            crop_name = crop_names[cr], \
            other_vars_title = ", pirrww, initr and gsprcp", \
            vars_filename = "di_all_red_k3s", subfolder = "AgMIP", \
            di_names = di_names_red, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, clusters = clusters, model = "GEPIC", \
            scatter = True, resid = True, rsq = True, fstat = True)
# pcalues are now low for most clusters for all crops. However, the residuals
# distribution is tilted (as we probably combine values from different samples)
# hence the pvalue is not meaningful. R^2 is still low in most cases.
with open("IntermediateResults/Clustering/Clusters/kMediods8" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clustersample", \
            yld = variables[cr][0], mask_crop = masks_yield[cr], \
            masks_di = masks_di, indices = indices_red[cr], \
            other_vars = variables[cr][2:], time = False, \
            crop_name = crop_names[cr], \
            other_vars_title = ", pirrww, initr and gsprcp", \
            vars_filename = "di_all_red_k8s", subfolder = "AgMIP", \
            di_names = di_names_red, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, clusters = clusters, model = "GEPIC", \
            scatter = True, resid = True, rsq = True, fstat = True)  
# similar to using 3 clusters...
    
    
# Now using cluster averages, but only for the full dataset...
with open("IntermediateResults/Clustering/Clusters/kMediods4" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clusteraverage", \
            yld = variables[cr][0], mask_crop = masks_yield[cr], \
            masks_di = masks_di, indices = indices[cr], \
            other_vars = variables[cr][2:], time = False, \
            crop_name = crop_names[cr], \
            other_vars_title = ", pirrww, initr and gsprcp", \
            vars_filename = "di_all_k4a",  subfolder = "AgMIP", \
            di_names = di_names, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, clusters = clusters, model = "GEPIC", \
            scatter = True, resid = True, rsq = True, fstat = True)
with open("IntermediateResults/Clustering/Clusters/kMediods7" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clusteraverage", \
            yld = variables[cr][0], mask_crop = masks_yield[cr], \
            masks_di = masks_di, indices = indices[cr], \
            other_vars = variables[cr][2:], time = False, \
            crop_name = crop_names[cr], \
            other_vars_title = ", pirrww, initr and gsprcp", \
            vars_filename = "di_all_k4a",  subfolder = "AgMIP", \
            di_names = di_names, figsize = figsize, lats_WA = lats_WA, \
            lons_WA = lons_WA, clusters = clusters, model = "GEPIC", \
            scatter = True, resid = True, rsq = True, fstat = True) 
# the northern clusters get good p values and quite high r^2 values in some
# cases, the southern ones less so...

# Summary: some settings give ok results for some areas, but none good enough
# to actually use... 

# 2) We then found a paper on a meta model for crop yields,
#    listing variables according to their relevance for yield modelling. We 
#    can't use their model, but hoped that this migth help to select variables
#    for linear regression    
    

# 3) Different dataset: Global dataset on historic yields (GDHY)
#    also uses models to create full dataset, but tries to recreate historic
#    data
    
# a) Visualize data availability
crops = ["wheat_spring", "wheat_winter", "soybean", "rice_major", \
             "rice_second", "maize_major", "maize_second"]
fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.01, top=0.98, left=0.1, right=0.9,
                wspace=0.3, hspace=-0.1)
for cr in range(0, len(crops)):
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                         crops[cr] + "_mask.txt", "rb") as fp:    
        yld_mask = (pickle.load(fp)).astype(float)
    yld_mask[yld_mask == 0] = np.nan
    fig.add_subplot(3, 3, cr + 1)
    OF.MapValues(yld_mask, lats_WA, lons_WA, crops[cr])
plt.suptitle("Data availability for GDHY")
# only rice_major and maize_major have a reasonable amount of data
with open("IntermediateResults/PreparedData/GDHY/" + \
                                     crops[3] + "_mask.txt", "rb") as fp:    
    rice_mask = (pickle.load(fp)).astype(float)
with open("IntermediateResults/PreparedData/GDHY/" + \
                                     crops[5] + "_mask.txt", "rb") as fp:    
    maize_mask = (pickle.load(fp)).astype(float)
    maize_mask[maize_mask == 1] = 2
joined_mask = rice_mask + maize_mask
joined_mask[joined_mask == 0] = np.nan
plt.figure()
OF.MapValues(joined_mask, lats_WA, lons_WA, \
             "Data of maize (green) and rice (red) and both (blue) in GDHY")

# b) Regression analysis per cell
crops = ["rice_major", "maize_major"]
yields = []
masks_yield = []

for cr in range(0, len(crops)):
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                         crops[cr] + "_yld.txt", "rb") as fp:    
        yields.append(pickle.load(fp))
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                         crops[cr] + "_mask.txt", "rb") as fp:    
        masks_yield.append(pickle.load(fp))
   
# variables[rice_major, maize_major]
#                        [SPEI, WaterDeficit, Precipitation, PET, DiurnalTemp]     
vars_gs, vars_detr_gs, vars_mask = OF.VariablesGDHY(crops, masks_yield, yields)
mask_vars = vars_mask[0] * vars_mask[2] * vars_mask[3] * vars_mask[4]

# - SPEI, Precipitation, PET, DiurnalTemp as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = yields[cr], \
            mask_crop = masks_yield[cr], masks_di = [mask_vars], \
            indices = [vars_gs[cr][0]], other_vars = vars_gs[cr][2:], \
            time = False, crop_name = crops[cr], \
            other_vars_title = ", Precipitation, PET and Diurnal " +
            "Temperature in growing season", vars_filename = "all", \
            subfolder = "GDHY", di_names = ["SPEI"], figsize = figsize, \
            lats_WA = lats_WA, lons_WA = lons_WA, model = "GDHY", \
            scatter = True, resid = True, rsq = True, fstat = True)
# not better than AgMIP versions
    
# - Detrended SPEI, Precipitation, PET, DiurnalTemp as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = yields[cr], \
          mask_crop = masks_yield[cr], masks_di = [mask_vars], \
          indices = [vars_detr_gs[cr][0]], other_vars = vars_detr_gs[cr][2:], \
          time = True, crop_name = crops[cr], \
          other_vars_title = ", Precipitation, PET, Diurnal Temperature" +
          " in growing season and time", vars_filename = "all_detr", \
          subfolder = "GDHY", di_names = ["Detr. SPEI"], figsize = figsize, \
          lats_WA = lats_WA, lons_WA = lons_WA, model = "GDHY", \
          scatter = True, resid = True, rsq = True, fstat = True)
# actually not that bad for the middle southern part...
    
# - Detrended SPEI as independent var
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "cellwise", yld = yields[cr], \
          mask_crop = masks_yield[cr], masks_di = [mask_vars], \
          indices = [vars_detr_gs[cr][0]], other_vars = [], \
          time = True, crop_name = crops[cr], \
          other_vars_title = "", vars_filename = "spei_detr", \
          subfolder = "GDHY", di_names = ["Detr. SPEI"], figsize = figsize, \
          lats_WA = lats_WA, lons_WA = lons_WA, model = "GDHY", \
          scatter = True, resid = True, rsq = True, fstat = True)
# still really good for around half of the cells. For that region there is a 
# significant relationship betwen SPEI and GDHY yields...
    
# - Detrended SPEI as independent var - increasing sample size by clustering
with open("IntermediateResults/Clustering/Clusters/kMediods4" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clustersample", yld = yields[cr], \
          mask_crop = masks_yield[cr], masks_di = [mask_vars], \
          indices = [vars_detr_gs[cr][0]], other_vars = [], \
          time = True, crop_name = crops[cr], \
          other_vars_title = "", vars_filename = "spei_detr_k4s", \
          subfolder = "GDHY", di_names = ["Detr. SPEI"], figsize = figsize, \
          lats_WA = lats_WA, lons_WA = lons_WA, clusters = clusters, \
          model = "GDHY", scatter = True, resid = True, rsq = True, \
          fstat = True)
with open("IntermediateResults/Clustering/Clusters/kMediods7" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clustersample", yld = yields[cr], \
          mask_crop = masks_yield[cr], masks_di = [mask_vars], \
          indices = [vars_detr_gs[cr][0]], other_vars = [], \
          time = True, crop_name = crops[cr], \
          other_vars_title = "", vars_filename = "spei_detr_k7s", \
          subfolder = "GDHY", di_names = ["Detr. SPEI"], figsize = figsize, \
          lats_WA = lats_WA, lons_WA = lons_WA, clusters = clusters, \
          model = "GDHY", scatter = True, resid = True, rsq = True, \
          fstat = True)        
# does not work well  
    
# - Detrended SPEI as independent var - cluster averages
with open("IntermediateResults/Clustering/Clusters/kMediods4" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clusteraverage", yld = yields[cr], \
          mask_crop = masks_yield[cr], \
          masks_di = [vars_mask[0], vars_mask[1]], \
          indices = vars_detr_gs[cr][0:2], other_vars = [], \
          time = True, crop_name = crops[cr], \
          other_vars_title = "", vars_filename = "di_detr_k4a", \
          subfolder = "GDHY", di_names = ["Detr. SPEI", "Detr. perc - PET"], \
          figsize = figsize, \
          lats_WA = lats_WA, lons_WA = lons_WA, clusters = clusters, \
          model = "GDHY", scatter = True, resid = True, rsq = True, \
          fstat = True)
# pvalues give significance for all clusters, good r^2 values for western 
# clusters, ok for eastern for maize, bad for eastern for rice
with open("IntermediateResults/Clustering/Clusters/kMediods7" + \
                                "_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)
for cr in range(0, len(crops)):
    OF.PlotRegressionResults(regtype = "clusteraverage", yld = yields[cr], \
          mask_crop = masks_yield[cr], \
          masks_di = [vars_mask[0], vars_mask[1]], \
          indices = vars_detr_gs[cr][0:2], other_vars = [], \
          time = True, crop_name = crops[cr], \
          other_vars_title = "", vars_filename = "di_detr_k7a", \
          subfolder = "GDHY", di_names = ["Detr. SPEI", "Detr. perc - PET"], \
          figsize = figsize, \
          lats_WA = lats_WA, lons_WA = lons_WA, clusters = clusters, \
          model = "GDHY", scatter = True, resid = True, rsq = True, \
          fstat = True)    
# This approach could be use if we use a reduced area, but as our goal is to 
# use uncorrelated areas to implement insurance schemes, we need a big area.
    

    
# 4) Bayesian and frequentist approach
    

    
    
# TODO Normality of data not necessary after all, normality of errors is 
#      needed for hypothesis test (e.g. p-values)
## - are yield model outputs from AgMIP normally distributed?
##       * using timeseries of all cells 
##       * using timeseries of cluster averages
#        
#models = ["epic-iiasa_agmerra_default", \
#         "epic-boku_grasp_default", \
#         "gepic_pgfv2_default", \
#         "papsim_agmerra_default"]
#crop_scenarios = ["_yield_whe_noirr", "_yield_mai_noirr", \
#         "_yield_soy_noirr", "_yield_ric_noirr"]
#crops = ["wheat", "maize", "soy", "rice"]
#
#yield_data = []  # yield_data[model][crop]
#for m in range(0, len(models)):
#    yield_data_tmp = []
#    for cr in range(0, len(crop_scenarios)):
#        if (m == 3) & (cr == 3):    # papsim does not have data for rice
#            continue
#        with open("IntermediateResults/PreparedData/AgMIP/" + \
#                       models[m] + crop_scenarios[cr] + ".txt", "rb") as fp: 
#            yields = pickle.load(fp)
#            yield_data_tmp.append(yields)  
#    yield_data.append(yield_data_tmp)
#        
## Histogramms
#for m in range(0, len(models)):
#    fig = plt.figure(figsize=figsize)
#    if m < 3:
#        for cr in range(0, len(crops)):
#            fig.add_subplot(2, 2, cr + 1)
#            plt.hist(yield_data[m][cr].flatten(), bins = 100, \
#                            density = True, alpha = 0.7)
#            plt.xlabel("Yield of " + crops[cr] + " in t/ha")
#            plt.ylabel("Density")
#            sk, pvals = stats.normaltest(yield_data[m][cr] \
#                                 [~np.isnan(yield_data[m][cr])].flatten())
#            plt.title(str(np.round(pvals)))
#        plt.suptitle("Model " + models[m] + " (no irrigation)")
#    else:
#        for cr in range(0, len(crops) - 1):
#            fig.add_subplot(1, 3, cr + 1)
#            plt.hist(yield_data[m][cr].flatten(), bins = 100, \
#                            density = True,  alpha = 0.7)
#            plt.xlabel("Yield of " + crops[cr] + " in t/ha")
#            plt.ylabel("Density")
#            sk, pvals = stats.normaltest(yield_data[m][cr] \
#                                  [~np.isnan(yield_data[m][cr])].flatten())
#            plt.title(str(np.round(pvals)))
#        plt.suptitle("Model " + models[m] + " (no irrigation)")
            
        
 
###############################################################################
# %% Analysis of trends in GDHY data
# We couldn't find a simple but significant regression between yields and 
# drought indicators. Therefore we decided not to use SPEI as source of 
# uncertainty in the model, but work with yield distributions directly. This
# means we can work with simple trends over time, and get the uncertainty from
# the distribution of the residuals.
crops = ["rice_major", "maize_major"]
year_start_GDHY = 1981
year_end_GDHY = 2016
len_ts = year_end_GDHY - year_start_GDHY + 1

# 1) Reclustering with area reduced to cells where at leatst one of the two 
#    crops has data
with open("IntermediateResults/Clustering/Distances/" + \
                                      "PearsonDist_spei03.txt", "rb") as fp:    
    dist = pickle.load(fp)   
with open("IntermediateResults/PreparedData/DroughtIndicators/" + \
                                          "mask_spei03_WA.txt", "rb") as fp:    
    mask_spei = pickle.load(fp)      
with open("IntermediateResults/PreparedData/GDHY/" + \
                                          crops[0] + "_mask.txt", "rb") as fp:    
    rice_mask = (pickle.load(fp)).astype(float)
with open("IntermediateResults/PreparedData/GDHY/" + \
                                          crops[1] + "_mask.txt", "rb") as fp:    
    maize_mask = (pickle.load(fp)).astype(float)
joined_mask = rice_mask + maize_mask
joined_mask[joined_mask == 2] = 1
with open("IntermediateResults/PreparedData/GDHY/" + \
                                    "RiceMaizeJoined_mask.txt", "wb") as fp:   
    pickle.dump(joined_mask, fp)

for k in range(1, 21):   
    OF.kMedoids(k, dist, mask_spei*joined_mask, \
                             "PearsonDist_spei03", version = "GDHY")    
    
for k in range(5, 10):
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                          str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    plt.figure()
    OF.VisualizeFinalCluster(clusters, medoids, lats_WA, lons_WA, "new" + str(k))
    
for k in range(5, 10):
    with open("IntermediateResults/Clustering/Clusters/old/GDHYkMediods" + \
                          str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    plt.figure()
    OF.VisualizeFinalCluster(clusters, medoids, lats_WA, lons_WA, "old" + str(k))   
    
# saving best version as optimal clustering
#for k in range(1, 21):
#    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
#                          str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
#        clusters = pickle.load(fp)
#        costs = pickle.load(fp)
#        medoids = pickle.load(fp)
#    min_costs = costs[-1]
#    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
#                      str(k) + "opt_PearsonDist_spei03.txt", "wb") as fp:  
#        pickle.dump(clusters, fp)
#        pickle.dump(costs, fp)
#        pickle.dump(medoids, fp)
#        pickle.dump(3052020, fp)
#    for s in range(0, 4):
#        with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
#            str(k) + "_PearsonDist_spei03_seed" + str(s) + ".txt", "rb") as fp:  
#            clusters = pickle.load(fp)
#            costs = pickle.load(fp)
#            medoids = pickle.load(fp)
#        if costs[-1] < min_costs:
#            min_costs = costs[-1]
#            with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" \
#                          + str(k) + "opt_PearsonDist_spei03.txt", "wb") as fp:  
#                pickle.dump(clusters, fp)
#                pickle.dump(costs, fp)
#                pickle.dump(medoids, fp)
#                pickle.dump(s, fp)
          
# optimal number of clusters

between_all = []
between_closest = []
within_cluster = []
kmax = 20

for k in range(2, kmax + 1):
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                          str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    all_dists, closest_dist = OF.MedoidMedoidDistd(medoids, dist)
    between_all.append(np.nanmean(all_dists))
    between_closest.append(np.nanmean(closest_dist))
    within_cluster.append(costs[-1]/(np.sum(~np.isnan(clusters[-1]))))
    

dists_between = [between_all, between_closest]
dists_within = [within_cluster, within_cluster]
title = ["Comparison using all clusters for SPEI", \
         "Comparison using closest clusters for SPEI"]

# plot distances
version = ["All", "Closest"]
for i in range(0,2):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,2,1) 
    ax.scatter(dists_within[i], dists_between[i],  c=range(2, kmax + 1))
    plt.title(title[i])
#    plt.xlim([0, 0.3])
#    plt.ylim([0.4, 0.7])
    plt.xlabel("Average distance within clusters")
    plt.ylabel("Average distance between clusters")
    for t, txt in enumerate(range(2, kmax + 1)):
        ax.annotate(txt, (dists_within[i][t] + 0.0013, \
                          dists_between[i][t] - 0.001)) 
    fig.add_subplot(1,2,2) 
    metric, cl_order = OF.MetricClustering(dists_within[i], dists_between[i], \
                                        refX = 0, refY = max(dists_between[i])) 
    plt.scatter(cl_order, metric)
    plt.xticks(range(2, 21))
    plt.title("Quantification of tradeoff")
    plt.xlabel("Number of clusters")
    plt.ylabel("Euclidean distance to (0, "+ \
                                  str(np.round(max(dists_between[i]), 2)) + \
                                  ") on scatter plot of distances")
    plt.suptitle("Tradeoff of distances within and between cluster")
        
    fig.savefig("Figures/Clustering/GDHYkMediods_ScatterInterVsIntraCluster" +\
                 version[i] + ".png", bbox_inches = "tight", pad_inches = 0.5)    
    
# for between_all the best k are the highest...
# for between_closest the best k are: 7, 3, 2, 6
    
# Visualize Clusters
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(bottom=0.03, top=0.97, left=0.1, right=0.9,
                wspace=0.25, hspace=-0.2)
for idx, k in enumerate([1, 2, 3, 7]):
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                      str(k) + "_PearsonDist_spei03.txt", "rb") as fp:    
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    fig.add_subplot(2, 2, idx+1)
    c = OF.VisualizeFinalCluster(clusters, medoids, \
                                       lats_WA, lons_WA, "k = " + str(k))
plt.suptitle("Clusters using area reduced to yield dataset", fontsize = 24)
fig.savefig("Figures/Clustering/GDHYkMediods.png",
                        bbox_inches = "tight", pad_inches = 0.5)    

        
# 2) Finding linear trend in GDHY data
yields = []
yields_log = []
yields_croot = []
for cr in range(0, len(crops)):
    with open("IntermediateResults/PreparedData/GDHY/" + \
                                         crops[cr] + "_yld.txt", "rb") as fp:  
        yld_tmp = pickle.load(fp)
        yields.append(yld_tmp)
        yld_tmp[yld_tmp == 0] = np.nan
        yields_log.append(np.log(yld_tmp))
        yields_croot.append(yld_tmp**(1/3))
len_ts = yields[0].shape[0]

# get spatial variance per cluster and year
for k in [1, 2, 3, 7]:
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                        str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    spatial_variance = np.empty([len_ts, len(crops), k])
    for cr in range(0, len(crops)):
        for cl in range(0, k):
            for t in range(0, len_ts):
                spatial_variance[t, cr, cl] = np.nanstd(yields[cr] \
                                                [t, clusters[-1] == (cl + 1)])
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                           "SpatialVariance_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(spatial_variance, fp)
        
# analyze spatial variance
for cr in [0, 1]:
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.3)
    for idx, k in enumerate([1, 2, 3, 7]):
        ax = fig.add_subplot(2, 2, idx+1)
        k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
        with open("IntermediateResults/LinearRegression/GDHY/" + \
                      "SpatialVariance_k" + str(k) + ".txt", "rb") as fp:    
            spatial_variance = pickle.load(fp)
        var_pred, residuals, residual_means, residual_stds, fstat, constants,\
            slopes = OF.DetrendClusterAvgGDHY(spatial_variance, k, crops)
        with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                      "SpatialVariance_k" + str(k) + ".txt", "wb") as fp:    
            pickle.dump(spatial_variance, fp); pickle.dump(var_pred, fp)
            pickle.dump(residuals, fp); pickle.dump(residual_means, fp)
            pickle.dump(residual_stds, fp); pickle.dump(fstat, fp)
            pickle.dump(constants, fp); pickle.dump(slopes, fp)
        for cl in range(0, k):
            plt.scatter(range(year_start_GDHY, year_start_GDHY + len_ts), \
                            spatial_variance[:, cr, cl], color = k_colors[cl], 
                            label = "Cluster " + str(cl+ 1) + ": " + \
                            str(np.round(fstat[cr, cl], 2)), marker = ".")
            plt.plot(range(year_start_GDHY, year_start_GDHY + len_ts), \
                     var_pred[:, cr, cl], color = k_colors[cl], alpha = 0.7)
        plt.title(str(k) + " cluster(s)")
        plt.legend(loc = "upper left")
    plt.suptitle("Spatial variance within clusters for GDHY " + \
                             crops[cr] + " yields with pvalue of trend")
    fig.savefig("Figures/LinearRegressions/GDHY/TrendSpatialVariance" + \
                "Cluster_" + crops[cr] + ".png", bbox_inches = "tight", \
                pad_inches = 0.5)   
# very low p-value of f statistic in almost all cases. Exceptions are for 
# rice_major cluster 2 and 3 (k = 7) and cluster 3 (k = 3) 

# analyse residuals of trend in spatial variance      
for cr in [0, 1]:
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.3)
    for idx, k in enumerate([1, 2, 3, 7]):
        k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
        fig.add_subplot(2, 2, idx+1)
        with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                   "SpatialVariance_k" + str(k) + ".txt", "rb") as fp:  
            spatial_var = pickle.load(fp)  
            var_pred = pickle.load(fp)  
            residuals = pickle.load(fp)  
        for cl in range(0, k):
            stat, ps = shapiro(residuals[:, cr, cl])
            plt.hist(residuals[:, cr, cl], alpha = 0.3, \
               label = "Cluster " + str(cl + 1) + ": " + \
               str(np.round(ps, 2)), color = k_colors[cl])
        plt.title(str(k) + " cluster(s)")
        plt.legend()
    plt.suptitle("Residual distribution of trends for spatial " + \
                 "variance of GDHY " + crops[cr] + \
                 " yields with p-values of Shapiro Normality Test")
    fig.savefig("Figures/LinearRegressions/GDHY/SpatialVariance" + \
                "_DistributionResidualsOfTrend_" + crops[cr] + \
                ".png", bbox_inches = "tight", pad_inches = 0.5) 
# some clusters with low Shapiro values...        

# cluster averages
with open("IntermediateResults/PreparedData/GDHY/" + \
                                    "RiceMaizeJoined_mask.txt", "rb") as fp:   
    joined_mask = pickle.load(fp)
for k in [1, 2, 3, 7]:
    yields_avg = np.empty([len_ts, len(crops), k]); yields_avg.fill(np.nan)
    yields_log_avg = np.empty([len_ts, len(crops), k])
    yields_log_avg.fill(np.nan)
    yields_croot_avg = np.empty([len_ts, len(crops), k])
    yields_croot_avg.fill(np.nan)
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                        str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    for cr in range(0, len(crops)):
        yields_avg[:, cr, :] = OF.ClusterAverage(yields[cr], clusters[-1], \
                                                            k, joined_mask)
        yields_log_avg[:, cr, :] = OF.ClusterAverage(yields_log[cr], \
                                          clusters[-1], k, joined_mask)
        yields_croot_avg[:, cr, :] = OF.ClusterAverage(yields_croot[cr], \
                                          clusters[-1], k, joined_mask)
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                             "YieldAverages_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(yields_avg, fp); pickle.dump(crops, fp)
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                           "LogYieldAverages_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(yields_log_avg, fp); pickle.dump(crops, fp)
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                        "CrootYieldAverages_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(yields_croot_avg, fp); pickle.dump(crops, fp)
        
# standardize cluster averages
for k in [1, 2, 3, 7]:
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                             "YieldAverages_k" + str(k) + ".txt", "rb") as fp:    
        yields_avg = pickle.load(fp)
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                           "SpatialVariance_k" + str(k) + ".txt", "rb") as fp:    
        spatial_variance = pickle.load(fp)
    standardized_yield_avg = yields_avg/spatial_variance
    with open("IntermediateResults/LinearRegression/GDHY/" + \
                 "StandardizedYieldAverages_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(standardized_yield_avg, fp); pickle.dump(crops, fp)
        
# detrend averages and visualize trend
cols = ["darkgreen", "darkred"]
for version in ["", "Log"]:
    for k in [1, 2, 3, 7]:
        with open("IntermediateResults/LinearRegression/GDHY/" + \
                    version + "YieldAverages_k" + str(k) + ".txt", "rb") as fp:    
            yields_avg = pickle.load(fp)
        avg_pred, residuals, residual_means, residual_stds, fstat, constants,\
            slopes = OF.DetrendClusterAvgGDHY(yields_avg, k, crops)
        with open("IntermediateResults/LinearRegression/GDHY/Detr" + \
                      version + "YieldAvg_k" + str(k) + ".txt", "wb") as fp:    
            pickle.dump(yields_avg, fp); pickle.dump(avg_pred, fp)
            pickle.dump(residuals, fp); pickle.dump(residual_means, fp)
            pickle.dump(residual_stds, fp); pickle.dump(fstat, fp)
            pickle.dump(constants, fp); pickle.dump(slopes, fp)
            pickle.dump(crops, fp)
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for cl in range(0, k):
            if k > 1:
                fig.add_subplot(2, np.ceil(k/2), cl + 1)
            for cr in [0, 1]:
                plt.scatter(range(year_start_GDHY, year_start_GDHY + len_ts), \
                       residuals[:, cr, cl], marker = ".", color = cols[cr])
                plt.fill_between(range(year_start_GDHY, year_start_GDHY + \
                                len_ts),np.repeat(residual_means[cr, cl] + \
                                residual_stds[cr, cl], len_ts), \
                                np.repeat(residual_means[cr, cl] - \
                                residual_stds[cr, cl], len_ts), \
                                color = cols[cr], alpha = 0.2)
            if (cl + 1) > (k - np.ceil(k/2)):
                plt.xlabel("Years")
            if cl%(np.ceil(k/2)) == 0:
                plt.ylabel("Yield in t/ha")
            plt.title("Cluster " + str(cl + 1))
        plt.suptitle("Residuals of linear trend (with std) for " + version + \
                     " average " + crops[0] + " (" +  cols[0] + ") and " + \
                     crops[1] + " (" + cols[1] + ")" + " yields (k = " + \
                     str(k) + ")")
        fig.savefig("Figures/LinearRegressions/GDHY/k" + str(k) + version + \
                    "Avg_Residuals.png", \
                    bbox_inches = "tight", pad_inches = 0.5)   
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for cl in range(0, k):
            if k > 1:
                ax = fig.add_subplot(2, np.ceil(k/2), cl + 1)
            else:
                ax = fig.add_subplot(1, 1, 1)
            dict_labels = {}
            for cr in [0, 1]:
                sns.regplot(x = np.array(range(year_start_GDHY, \
                      year_start_GDHY + len_ts)), y = yields_avg[:, cr, cl], \
                      color = cols[cr], ax = ax, marker = ".", truncate = True)
            if (cl + 1) > (k - np.ceil(k/2)):
                plt.xlabel("Years")
            if cl%(np.ceil(k/2)) == 0:
                plt.ylabel("Yield in t/ha")
            plt.title("Cluster " + str(cl + 1))
        plt.suptitle("Cluster average of " + version + " GDHY " + \
                 "yields (k = " + str(k) + ") and trend " + \
                 "with 95% confidence interval for " + crops[0] + " (" + \
                 cols[0] + ") and " + crops[1] + " (" + cols[1] + ")")
        fig.savefig("Figures/LinearRegressions/GDHY/k" + str(k) + version + \
                "Avg_YieldTrends.png", bbox_inches = "tight", \
                pad_inches = 0.5)   
    
# detrend standardized averages
cols = ["darkgreen", "darkred"]
for k in [1, 2, 3, 7]:
    with open("IntermediateResults/LinearRegression/GDHY/" + \
             "StandardizedYieldAverages_k" + str(k) + ".txt", "rb") as fp:    
        std_yields_avg = pickle.load(fp)
    avg_pred, residuals, residual_means, residual_stds, fstat, constants,\
        slopes = OF.DetrendClusterAvgGDHY(std_yields_avg, k, crops)
    with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                  "StandardizedYieldAverages_k" + str(k) + ".txt", "wb") as fp:    
        pickle.dump(std_yields_avg, fp); pickle.dump(avg_pred, fp)
        pickle.dump(residuals, fp); pickle.dump(residual_means, fp)
        pickle.dump(residual_stds, fp); pickle.dump(fstat, fp)
        pickle.dump(constants, fp); pickle.dump(slopes, fp)
        pickle.dump(crops, fp)
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.3)
    for cl in range(0, k):
        if k > 1:
            fig.add_subplot(2, np.ceil(k/2), cl + 1)
        for cr in [0, 1]:
            plt.scatter(range(year_start_GDHY, year_start_GDHY + len_ts), \
                   residuals[:, cr, cl], marker = ".", color = cols[cr])
            plt.fill_between(range(year_start_GDHY, year_start_GDHY + len_ts),\
                             np.repeat(residual_means[cr, cl] + \
                                           residual_stds[cr, cl], len_ts), 
                             np.repeat(residual_means[cr, cl] - \
                                           residual_stds[cr, cl], len_ts), \
                                       color = cols[cr], alpha = 0.2)
        if (cl + 1) > (k - np.ceil(k/2)):
            plt.xlabel("Years")
        if cl%(np.ceil(k/2)) == 0:
            plt.ylabel("Yield in t/ha")
        plt.title("Cluster " + str(cl + 1))
    plt.suptitle("Residuals of linear trend (with std) for standardized " + \
                 "average " + crops[0] + " (" +  cols[0] + ") and " + \
                 crops[1] + " (" + cols[1] + ")" + " yields (k = " + \
                 str(k) + ")")
    fig.savefig("Figures/LinearRegressions/GDHY/k" + str(k) + \
                "StandardizedAvg_Residuals.png", \
                bbox_inches = "tight", pad_inches = 0.5)   
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.3)
    for cl in range(0, k):
        if k > 1:
            ax = fig.add_subplot(2, np.ceil(k/2), cl + 1)
        else:
            ax = fig.add_subplot(1, 1, 1)
        dict_labels = {}
        for cr in [0, 1]:
            sns.regplot(x = np.array(range(year_start_GDHY, year_start_GDHY + \
                        len_ts)), y = std_yields_avg[:, cr, cl], \
                        color = cols[cr], ax = ax, marker = ".", \
                        truncate = True)
        if (cl + 1) > (k - np.ceil(k/2)):
            plt.xlabel("Years")
        if cl%(np.ceil(k/2)) == 0:
            plt.ylabel("Yield in t/ha")
        plt.title("Cluster " + str(cl + 1))
    plt.suptitle("Standardized cluster average of GDHY " + \
             "yields (k = " + str(k) + ") and trend " + \
             "with 95% confidence interval for " + crops[0] + " (" + \
             cols[0] + ") and " + crops[1] + " (" + cols[1] + ")")
    fig.savefig("Figures/LinearRegressions/GDHY/k" + str(k) + \
            "StandardizedAvg_YieldTrends.png", bbox_inches = "tight", \
            pad_inches = 0.5)   
        
# bootstrapping residuals         
num_bt = 100
for version in ["", "Log"]:
    for k in [1, 2, 3, 7]:
        with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                      version + "YieldAvg_k" + str(k) + ".txt", "rb") as fp:    
            yields_avg = pickle.load(fp); avg_pred = pickle.load(fp)  
            residuals = pickle.load(fp); residual_means = pickle.load(fp)  
            residual_stds = pickle.load(fp); fstat = pickle.load(fp)  
            constants = pickle.load(fp); slopes = pickle.load(fp)  
        bt_residuals, bt_slopes, bt_constants = \
                        OF.BootstrapResiduals(avg_pred, residuals, constants, \
                                           slopes, num_bt, k, crops)
        with open("IntermediateResults/LinearRegression/GDHY/Bootstrap" + \
              "Resids" + version + "YieldAvg_k" + str(k) + ".txt", "wb") as fp:    
            pickle.dump(bt_residuals, fp)
            pickle.dump(bt_slopes, fp)
            pickle.dump(bt_constants, fp)
        
# significance of trends
for version in ["", "Log"]:
    print(version)
    for k in [1, 2, 3, 7]:
        print(str(k))
        with open("IntermediateResults/LinearRegression/GDHY/" + \
                "Detr" + version + "YieldAvg_k" + str(k) + ".txt", "rb") as fp:    
            yields_avg = pickle.load(fp)  
            avg_pred = pickle.load(fp)  
            residuals = pickle.load(fp)  
            residual_means = pickle.load(fp)  
            residual_stds = pickle.load(fp)  
            fstat = pickle.load(fp)  
        print(fstat)
# very low values. But as residuals are not normally distributed these values 
# are useless...
         
# analyse residuals    
for version in ["", "Log"]:       
    for cr in [0, 1]:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for idx, k in enumerate([1, 2, 3, 7]):
            k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
            fig.add_subplot(2, 2, idx+1)
            with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                      version + "YieldAvg_k" + str(k) + ".txt", "rb") as fp:  
                yields_avg = pickle.load(fp)  
                avg_pred = pickle.load(fp)  
                residuals = pickle.load(fp)  
            for cl in range(0, k):
                stat, ps = shapiro(residuals[:, cr, cl])
                plt.hist(residuals[:, cr, cl], alpha = 0.3, \
                   label = "Cluster " + str(cl + 1) + ": " + \
                   str(np.round(ps, 2)), color = k_colors[cl])
            plt.title(str(k) + " cluster(s)")
            plt.legend()
        plt.suptitle("Distribution of residuals of linear trend in GDHY " + \
                     crops[cr] + " " + version + \
                     " yields with p-values of Shapiro Normality Test")
        fig.savefig("Figures/LinearRegressions/GDHY/"  + "ClusterAvg" + \
                    version + "_DistributionResidualsOfTrend_" + crops[cr] + \
                    ".png", bbox_inches = "tight", pad_inches = 0.5)   
        
# analyse residuals for standardized version
for cr in [0, 1]:
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.3)
    for idx, k in enumerate([1, 2, 3, 7]):
        k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
        fig.add_subplot(2, 2, idx+1)
        with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                  "StandardizedYieldAverages_k" + str(k) + ".txt", "rb") as fp:  
            yields_avg = pickle.load(fp)  
            avg_pred = pickle.load(fp)  
            residuals = pickle.load(fp)  
        for cl in range(0, k):
            stat, ps = shapiro(residuals[:, cr, cl])
            plt.hist(residuals[:, cr, cl], alpha = 0.3, \
               label = "Cluster " + str(cl + 1) + ": " + \
               str(np.round(ps, 2)), color = k_colors[cl])
        plt.title(str(k) + " cluster(s)")
        plt.legend()
    plt.suptitle("Residual Distribution of linear trend in standardized " + \
                 "GDHY " + crops[cr] + " yield averages with p-values of " + \
                 "Shapiro Normality Test")
    fig.savefig("Figures/LinearRegressions/GDHY/StandardizedClusterAvg" + \
                    "_DistributionResidualsOfTrend_" + crops[cr] + ".png", \
                    bbox_inches = "tight", pad_inches = 0.5)   
        
# compare with normally distributed residuals    
for version in ["", "Log"]:       
    for cr in [0, 1]:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for idx, k in enumerate([1, 2, 3, 7]):
            k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
            fig.add_subplot(2, 2, idx+1)
            with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                      version + "YieldAvg_k" + str(k) + ".txt", "rb") as fp:  
                yields_avg = pickle.load(fp)  
                avg_pred = pickle.load(fp)  
                residuals = pickle.load(fp)  
                residual_means = pickle.load(fp)  
                residual_stds = pickle.load(fp)  
            for cl in range(0, k):
                residuals_tmp = np.random.normal(residual_means[cr, cl], \
                                                 residual_stds[cr, cl], \
                                                 len_ts)
                stat, ps = shapiro(residuals_tmp)
                plt.hist(residuals_tmp, alpha = 0.3, \
                   label = "Cluster " + str(cl + 1) + ": " + \
                   str(np.round(ps, 2)), color = k_colors[cl])
            plt.title(str(k) + " cluster(s)")
            plt.legend()
        plt.suptitle("Distribution of residuals sampled from normal " + \
                     "distribution with mean and std of reiduals from " + \
                     "linear trend in GDHY " + crops[cr] + " " + version + \
                     " yields with p-values of Shapiro Normality Test")
        fig.savefig("Figures/LinearRegressions/GDHY/"  + "ClusterAvg" + \
                    version + "_DistributionSampledNormalResiduals_" + \
                    crops[cr] + ".png", bbox_inches = "tight", \
                    pad_inches = 0.5)   
    
# analyse residuals by Q-Q plot
for version in ["", "Log"]:       
    for cr in [0, 1]:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for idx, k in enumerate([1, 2, 3, 7]):
            ax = fig.add_subplot(2, 2, idx+1)
            k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
            with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                      version + "YieldAvg_k" + str(k) + ".txt", "rb") as fp:  
                yields_avg = pickle.load(fp)  
                avg_pred = pickle.load(fp)  
                residuals = pickle.load(fp)  
            for cl in range(0, k):
                stat, ps = shapiro(residuals[:, cr, cl])
                sm.qqplot(residuals[:, cr, cl], line ='s', ax = ax, \
                          label = "Cluster " + str(cl + 1), marker = ".", \
                          c = k_colors[cl], alpha = 0.5)  
            plt.title(str(k) + " cluster(s)")
            plt.legend()
        plt.suptitle("Q-Q-Plots for distribution of residuals of linear " + \
                     "trend in GDHY " + crops[cr] + " " + version + " yields")
        fig.savefig("Figures/LinearRegressions/GDHY/ClusterAvg" + \
                    version + "_QQPlotResidualsOfTrend_" + crops[cr] + \
                    ".png", bbox_inches = "tight", pad_inches = 0.5)           

# analyse residuals by Q-Q plot for standardized version
for cr in [0, 1]:
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.3, hspace=0.3)
    for idx, k in enumerate([1, 2, 3, 7]):
        ax = fig.add_subplot(2, 2, idx+1)
        k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
        with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                  "StandardizedYieldAverages_k" + str(k) + ".txt", "rb") as fp:  
            yields_avg = pickle.load(fp)  
            avg_pred = pickle.load(fp)  
            residuals = pickle.load(fp)  
        for cl in range(0, k):
            stat, ps = shapiro(residuals[:, cr, cl])
            sm.qqplot(residuals[:, cr, cl], line ='s', ax = ax, \
                      label = "Cluster " + str(cl + 1), marker = ".", \
                      c = k_colors[cl], alpha = 0.5)  
        plt.title(str(k) + " cluster(s)")
        plt.legend()
    plt.suptitle("Q-Q-Plots for distribution of residuals of linear " + \
                 "trend in standardized GDHY " + crops[cr] + " yield averages")
    fig.savefig("Figures/LinearRegressions/GDHY/StandardizedClusterAvg" + \
                "_QQPlotResidualsOfTrend_" + crops[cr] + \
                ".png", bbox_inches = "tight", pad_inches = 0.5)           

# compare with normally distributed residuals    
for version in ["", "Log"]:       
    for cr in [0, 1]:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for idx, k in enumerate([1, 2, 3, 7]):
            ax = fig.add_subplot(2, 2, idx+1)
            k_colors = cm.jet((np.arange(0, 255, np.ceil(255/k))).astype(int))
            with open("IntermediateResults/LinearRegression/GDHY/" + "Detr" + \
                      version + "YieldAvg_k" + str(k) + ".txt", "rb") as fp:  
                yields_avg = pickle.load(fp)  
                avg_pred = pickle.load(fp)  
                residuals = pickle.load(fp)  
                residual_means = pickle.load(fp)  
                residual_stds = pickle.load(fp)   
            for cl in range(0, k):
                residuals_tmp = np.random.normal(residual_means[cr, cl], \
                                                 residual_stds[cr, cl], \
                                                 len_ts)
                sm.qqplot(residuals_tmp, line ='s', ax = ax, \
                          label = "Cluster " + str(cl + 1), marker = ".", \
                          c = k_colors[cl], alpha = 0.5)  
            plt.title(str(k) + " cluster(s)")
            plt.legend()
        plt.suptitle("Q-Q-Plots for distribution of residuals ampled from " + \
                     " normal distribution with mean and std of residuals " + \
                     " from linear trend in GDHY " + crops[cr] + " " + \
                     version + " yields")
        fig.savefig("Figures/LinearRegressions/GDHY/ClusterAvg" + \
                    version + "_QQPlotSampledNormalResiduals_" + crops[cr] + \
                    ".png", bbox_inches = "tight", pad_inches = 0.5)  
        
# analyse residuals of bootstrapping   
for version in ["", "Log"]:       
    for cr in [0, 1]:
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                        wspace=0.3, hspace=0.3)
        for idx, k in enumerate([1, 2, 3, 7]):
            fig.add_subplot(2, 2, idx+1)
            with open("IntermediateResults/LinearRegression/GDHY/Bootstrap" + \
                              "Resids"+ version + "YieldAvg_k" + str(k) + \
                                                          ".txt", "rb") as fp:  
                bt_residuals = pickle.load(fp)  
                bt_slopes = pickle.load(fp)  
                bt_constants = pickle.load(fp)  
            for cl in range(0, k):
                stat, ps = shapiro(bt_residuals[:, :, cr, cl].flatten())
                plt.hist(bt_residuals[:, :, cr, cl].flatten(), alpha = 0.4, \
                   label = "Cluster " + str(cl + 1) + ": " + \
                   str(np.round(ps, 2)), bins = 50, density = True)
            plt.title(str(k) + " cluster(s)")
            plt.legend()
        plt.suptitle("Distribution of bootstrap residuals of linear trend" + \
                     " in GDHY " + crops[cr] + " " + version + \
                     " yields with p-values of Shapiro Normality Test")
        fig.savefig("Figures/LinearRegressions/GDHY/"  + "ClusterAvg" + \
                   version + "_DistributionBootstrapResidualsOfTrend_" + \
                   crops[cr] + ".png", bbox_inches = "tight", pad_inches = 0.5)   
        

# Summary / Interpretation
# The idea is to use a linear trend of the yield averages per cluster to 
# project into the future, and then add residuals to get a distribution of
# yields. The residual distribution is taken from the detrending over the 
# whole timeseries, which we thought reasonable as the trend was removed and
# variance in residuals should come from other things as climate variablilty.
# However, we don't get normally distributed residuals for all clusters, for
# some the Shapiro Test rejects the hyothesis of normal distribution. Q-Q Plots
# visually show a quite linear relationshipt between normal quantiles and 
# residual quantiles though.
# Transforming the original yield data before averaging does not improve the
# behaviour of the residuals.
# Looking at the plot of yield averages and the trend, it seems that the 
# vairance increases over time. Yield averages should therefore be standardized
# before detrending. We standardized by deviding by the spatial variance of
# the cluster for each year seperately. For rice we get higher p-values for
# Shapiro test in general, for maize it is just shifted which clusters get good
# values and which not... The Q-Q Plots seem smilar... 
# To use this to project yield distributions, we would have to understand the
# trend in spatial variance as well. Rsiduals of linear trend fail Shapiro 
# normality test in many cases, making p-values for significance of linear
# trend useless. There seems to be a increasing trend in spatial variance for
# most cluster, but some alsow cyclic patterns.
# As there is no simple trend in spatial variance over time (there is a linear 
# component, but also other aspects), and there is no significant improvement
# in redisual distribution, we will not use standardization.
# Comparing distribution of residuals (Histograms, Shaprio, Q-Q-Plots) with
# distribtuion of residuals drawn from a normal distribution with mean and std
# of real resiudals, suggests that partially the non-Gaussianity comes from
# having a small sample size, as it e.g. shows similar deviations from linear
# Q-Q-Plots. 
# As assuming that the linear trend can be projected for 25 years into the
# future is quite a big assumption anyway, we decided that assuming normal
# residuals is justifiable for the thesis (but want to work more on this in 
# the future).
        
        
###############################################################################
# %% ##################### PLOTS INCLUDED IN THESIS ###########################
###############################################################################    
        
# - Visualize area (and reduced area)

fig = plt.figure(figsize=figsize)
ax1 = fig.add_subplot(1 ,2 ,1)
with open("IntermediateResults/Clustering/Clusters/" + \
                      "kMediods1_PearsonDist_spei03.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp)  
OF.VisualizeFinalCluster(clusters, medoids, lats_WA, lons_WA, title = "A", \
                         fontsizet = 28, fontsizea = 22)   
ax2 = fig.add_subplot(1 ,2 ,2)
with open("IntermediateResults/Clustering/Clusters/" + \
                      "GDHYkMediods1_PearsonDist_spei03.txt", "rb") as fp:    
    clusters = pickle.load(fp)
    costs = pickle.load(fp)
    medoids = pickle.load(fp) 
OF.VisualizeFinalCluster(clusters, medoids, lats_WA, lons_WA, title = "B", \
                         fontsizet = 28, fontsizea = 22)      
plt.show()
fig.savefig("Figures/Clustering/Areas.png", bbox_inches = "tight", \
                        pad_inches = 0.5)


# - Visualizing CLuster 
DistFiles = ["PearsonDist_spei03",
             "PearsonDist_spei03_CutNeg2", 
             "PearsonDist_spei03_Boolean_CutNeg2",
             "PearsonDist_spei03_30y"]

titles = ["A", "B", "C", "D"]        

for k in [8]:
    fig = plt.figure(k, figsize=figsize)
    fig.subplots_adjust(bottom=0.03, top=0.9, left=0.1, right=0.9,
                    wspace=0.25, hspace=0.25)
    for file in range(0,len(DistFiles)):
        with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                  str(k) + "_" + DistFiles[file] + ".txt", "rb") as fp:    
            clusters = pickle.load(fp)
            costs = pickle.load(fp)
            medoids = pickle.load(fp)
        fig.add_subplot(2,2,file+1)
        c = OF.VisualizeFinalCluster(clusters, medoids,lats_WA,  \
                  lons_WA, titles[file], fontsizet = 22, fontsizea = 18)
fig.savefig("Figures/Clustering/thesis_kMediods" + str(k) + ".png",
                        bbox_inches = "tight", pad_inches = 0.5)    

# Optimal number of clusters

with open("IntermediateResults/Clustering/Distances/" + \
                                "PearsonDist_spei03.txt", "rb") as fp:    
    dist = pickle.load(fp) 
    
between_closest_fullarea = []
between_closest_gdhyarea = []
within_cluster_fullarea = []
within_cluster_gdhyarea = []
kmax = 20

for k in range(2, kmax + 1):
    with open("IntermediateResults/Clustering/Clusters/kMediods" + \
                      str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    all_dists, closest_dist = OF.MedoidMedoidDistd(medoids, dist)
    between_closest_fullarea.append(np.nanmean(closest_dist))
    within_cluster_fullarea.append(costs[-1]/(np.sum(~np.isnan(clusters[-1]))))
    
    with open("IntermediateResults/Clustering/Clusters/GDHYkMediods" + \
                          str(k) + "_PearsonDist_spei03.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    all_dists, closest_dist = OF.MedoidMedoidDistd(medoids, dist)
    between_closest_gdhyarea.append(np.nanmean(closest_dist))
    within_cluster_gdhyarea.append(costs[-1]/(np.sum(~np.isnan(clusters[-1]))))
    

dists_between = [between_closest_fullarea, between_closest_gdhyarea]
dists_within = [within_cluster_fullarea, within_cluster_gdhyarea]
title = ["A", "B"]

# plot distances
fig1 = plt.figure(figsize = figsize)
fig2 = plt.figure(figsize = figsize)
fig1.subplots_adjust(bottom=0.2, top=0.6, left=0.2, right=0.8,
                wspace=0.3, hspace=0.3)
fig2.subplots_adjust(bottom=0.2, top=0.6, left=0.2, right=0.8,
                wspace=0.3, hspace=0.3)
for i in range(0,2):
    ax = fig1.add_subplot(1, 2, i+1) 
    ax.scatter(dists_within[i], dists_between[i],  c=range(2, kmax + 1), s =40)
    ax.set_title(title[i], fontsize = 22)
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0.42, 0.68])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel("Avg. distance within clusters", fontsize = 20)
    ax.set_ylabel("Avg. distance between clusters", fontsize = 20)
    for t, txt in enumerate(range(2, kmax + 1)):
        if (txt == 2) or (t%5) == 3:
            if txt < 10:
                ax.annotate(txt, (dists_within[i][t] - 0.012, \
                                  dists_between[i][t] + 0.0015), fontsize = 15) 
            else:
                ax.annotate(txt, (dists_within[i][t] - 0.018, \
                                  dists_between[i][t] + 0.0015), fontsize = 15)                        
    ax = fig2.add_subplot(1, 2, i+1) 
    metric, cl_order = OF.MetricClustering(dists_within[i], dists_between[i], \
                                        refX = 0, refY = max(dists_between[i])) 
    ax.scatter(cl_order, metric, s =40)
    ax.set_xticks(range(2, 21))
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_title(title[i], fontsize = 22)
    ax.set_xlabel("Number of clusters", fontsize = 20)
    ax.set_ylabel("Euclidean distance to (0, "+ \
                                  str(np.round(max(dists_between[i]), 2)) + \
                                  ")",  fontsize = 20)
    
fig1.savefig("Figures/Clustering/thesis_ScatterInterVsIntraClustes.png", \
            bbox_inches = "tight", pad_inches = 0.5)       
fig2.savefig("Figures/Clustering/thesis_RankingOfK.png", \
            bbox_inches = "tight", pad_inches = 0.5)      

  
# GDHY detrend averages and visualize trend
cols = ["royalblue", "darkred"]
k = 2
version = ""
title = ["A", "B"]
year_start_GDHY = 1981
year_end_GDHY = 2016
len_ts = year_end_GDHY - year_start_GDHY + 1

with open("IntermediateResults/LinearRegression/GDHY/" + \
            version + "YieldAverages_k" + str(k) + ".txt", "rb") as fp:    
    yields_avg = pickle.load(fp)
    
fig = plt.figure(figsize = figsize)
fig.subplots_adjust(bottom=0.2, top=0.6, left=0.2, right=0.8,
                wspace=0.3, hspace=0.3)
for cl in range(0, k):
    ax = fig.add_subplot(1, 2, cl + 1)
    dict_labels = {}
    for cr in [0, 1]:
        sns.regplot(x = np.array(range(year_start_GDHY, \
              year_start_GDHY + len_ts)), y = yields_avg[:, cr, cl], \
              color = cols[cr], ax = ax, marker = ".", truncate = True, scatter_kws={'s':42})
    plt.xlabel("Years", fontsize = 20)
    plt.ylabel("Yield in t/ha", fontsize = 20)
    plt.title(title[cl], fontsize = 22)
    plt.legend(["Rice", "Maize"], fontsize = 18, fancybox = True)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
fig.savefig("Figures/LinearRegressions/GDHY/thesis_k" + str(k) + version + \
        "Avg_YieldTrends.png", bbox_inches = "tight", \
        pad_inches = 0.5) 