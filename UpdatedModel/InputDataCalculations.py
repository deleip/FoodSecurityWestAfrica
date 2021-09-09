# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:07:00 2021

@author: leip
"""


# %% IMPORTING NECESSARY PACKAGES AND SETTING WORKING DIRECTORY

# set the right directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# import other modules
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import ModelCode.DataPreparation as DP
import ModelCode.GroupingClusters as GC

if not os.path.isdir("InputData/Visualization"):
    os.mkdir("InputData/Visualization")

# %% 1. Profitable Areas

# Only including cells, where either maize or rice has profitable average 
# yields (based on linear regression evaluated for baseyear 2016)

DP.ProfitableAreas()
# creates ProcessedData/MaskProfitableArea.txt
# TODO visualizations
    
# combine with SPEI mask
with open("ProcessedData/MaskProfitableArea.txt", "rb") as fp:    
    mask_profitable = pickle.load(fp)
with open("ProcessedData/mask_SPEI03_WA.txt", "rb") as fp:    
    mask_SPEI = pickle.load(fp)
    
maskAreaUsed = mask_profitable * mask_SPEI

with open("InputData/Other/MaskAreaUsed.txt", "wb") as fp:    
    pickle.dump(maskAreaUsed, fp)
# creates InputData/Other/MaskAreaUsed.txt

# %% 2. Average farm gate prices

# based on prepared price dataset and weighted with areas

with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
    MaskAreaUsed = pickle.load(fp)
    
prices = DP.CalcAvgProducerPrices(rice_mask = MaskAreaUsed, maize_mask = MaskAreaUsed)

with open("InputData/Prices/RegionFarmGatePrices.txt", "wb") as fp:    
    pickle.dump(prices, fp)
# creates InputData/Prices/RegionFarmGatePrices.txt


# %% 3. Average calorie demand per person and day

# based on some country values (ProcessedData/CountryCaloricDemand.csv)
# we calculate the average caloric demand per person and day, using area as weight
                
DP.AvgCaloricDemand()      
# creates InputData/Other/AvgCaloricDemand.txt

# %% 4. calculate pearson distance between all cells

with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
    MaskAreaUsed = pickle.load(fp)
    
DP.CalcPearsonDist(MaskAreaUsed)
# creates InputData/Other/PearsonCorrSPEI03.txt
# creates InputData/Other/PearsonDistSPEI03.txt

# %% 5. run clustering algorithm 

with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
    MaskAreaUsed = pickle.load(fp)
    
with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
    pearsonDist = pickle.load(fp)
    
for k in range(1, 20):
    print(k, flush = True)
    DP.kMedoids(k, pearsonDist, MaskAreaUsed, "PearsonDistSPEI")
# creates Inputdata/Clusters/Clustering/kMedoidsX_PearsonDistSPEI.txt    

# %% 6. best number of cluster

with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
    pearsonDist = pickle.load(fp)  

between_all = []
between_closest = []
within_cluster = []
kmax = 19

for k in range(2, kmax + 1):
    with open("InputData/Clusters/Clustering/kMediods" + \
                          str(k) + "_PearsonDistSPEI.txt", "rb") as fp:  
        clusters = pickle.load(fp)
        costs = pickle.load(fp)
        medoids = pickle.load(fp)
    all_dists, closest_dist = DP.MedoidMedoidDistd(medoids, pearsonDist)
    between_all.append(np.nanmean(all_dists))
    between_closest.append(np.nanmean(closest_dist))
    within_cluster.append(costs/(np.sum(~np.isnan(clusters))))
    
dists_between = [between_all, between_closest]
dists_within = [within_cluster, within_cluster]
title = ["Comparison using all clusters for SPEI", \
         "Comparison using closest clusters for SPEI"]

# plot distances
version = ["All", "Closest"]
for i in range(0,2):
    fig = plt.figure(figsize = (24, 13.5))
    ax = fig.add_subplot(1,2,1) 
    ax.scatter(dists_within[i], dists_between[i],  c=range(2, kmax + 1))
    plt.title(title[i])
    plt.xlabel("Average distance within clusters")
    plt.ylabel("Average distance between clusters")
    for t, txt in enumerate(range(2, kmax + 1)):
        ax.annotate(txt, (dists_within[i][t] + 0.0013, \
                          dists_between[i][t] - 0.001)) 
    fig.add_subplot(1,2,2) 
    metric, cl_order = DP.MetricClustering(dists_within[i], dists_between[i], \
                                        refX = 0, refY = max(dists_between[i])) 
    plt.scatter(cl_order, metric)
    plt.xticks(range(2, 21))
    plt.title("Quantification of tradeoff")
    plt.xlabel("Number of clusters")
    plt.ylabel("Euclidean distance to (0, "+ \
                                  str(np.round(max(dists_between[i]), 2)) + \
                                  ") on scatter plot of distances")
    plt.suptitle("Tradeoff of distances within and between cluster")
        
    fig.savefig("InputData/Visualization/kMediods_ScatterInterVsIntraCluster" +\
                 version[i] + ".png", bbox_inches = "tight", pad_inches = 0.5)      
        
# creates InputData/Visualization/kMediods_ScatterInterVsIntraClusterAll.png
#         InputData/Visualization/kMediods_ScatterInterVsIntraClusterClosest.png
# Using the average over all clusters, the more clusters the better the result
# of our metric
# Using only the closest cluster, the optimum lies at 9 cluster.   
       
# %% 7. Adjacency matrix

with open("InputData/Clusters/Clustering/kMediods9_PearsonDistSPEI.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    
fig = plt.figure()
cmap = mpl.cm.Paired
bounds = np.arange(0.5, 10, 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(np.flip(clusters, axis = 0), cmap = cmap)
plt.title("Division in 9 cluster")
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='horizontal',
             ticks = range(1, 10))
plt.show()
fig.savefig("InputData/Visualization/kMediods9.png", bbox_inches = "tight", pad_inches = 0.5)     
plt.close()

AdjacencyMatrix = np.array([[1, 0, 1, 0, 1, 1, 1, 0, 1],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 1, 0, 1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0, 1],
                            [1, 1, 1, 0, 1, 0, 1, 1, 0],
                            [1, 0, 1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 1, 1, 0, 1, 0, 1],
                            [0, 1, 1, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 1, 0, 1, 1, 0, 1]])

with open("InputData/Clusters/AdjacencyMatrices/k9AdjacencyMatrix.txt", "wb") as fp:
     pickle.dump(AdjacencyMatrix, fp)
# creates InputData/Clusters/AdjacencyMatrices/k9AdjacencyMatrix.txt
        
# %% 8. Cluster groupings

for aim in ["Similar", "Dissimilar"]:
    for adj in [True, False]:
        for s in [1, 2, 3, 5, 9]:
            ShiftedGrouping, BestCosts, valid = GC.GroupingClusters(k = 9, size = s, aim = aim, adjacent = adj, 
                                 title = None, figsize = None)
# creates GroupingSizeX(Dis)Similar(Adj).txt  
 
# %% 9. Yield trends

with open("InputData/Other/CultivationCosts.txt", "rb") as fp:
    costs = pickle.load(fp)
with open("InputData//Prices/RegionFarmGatePrices.txt", "rb") as fp:    
    prices = pickle.load(fp)

# yield thresholds for making profits
threshold = costs / prices
# 1.83497285, 1.03376958

DP.YldTrendsCluster(k = 9)  
# creates: ProcessedData/YieldAverages_k9.txt
#          InputData/YieldTrends/DetrYieldAvg_k9.txt
     

# plot yield trends
k = 9
with open("ProcessedData/YieldAverages_k" + str(k) + ".txt", "rb") as fp:    
    yields_avg = pickle.load(fp)
    crops = pickle.load(fp)        
    
cols = ["darkgreen", "darkred"]
start_year = 1981
len_ts = yields_avg.shape[0]

fig = plt.figure(figsize = (24, 13.5))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                wspace=0.3, hspace=0.3)
for cl in range(0, k):
    if k > 6:
        ax = fig.add_subplot(3, int(np.ceil(k/3)), cl + 1)
    elif k > 2:
        ax = fig.add_subplot(2, int(np.ceil(k/2)), cl + 1)
    else:
        ax = fig.add_subplot(1, k, cl + 1)
    dict_labels = {}
    for cr in [0, 1]:
        sns.regplot(x = np.array(range(start_year, \
              start_year + len_ts)), y = yields_avg[:, cr, cl], \
              color = cols[cr], ax = ax, marker = ".", truncate = True)
        plt.plot(range(start_year, start_year + len_ts), \
                 np.repeat(threshold[0], len_ts), ls = "--", alpha = 0.7)
        plt.plot(range(start_year, start_year + len_ts), \
                 np.repeat(threshold[1], len_ts), ls = "--", alpha = 0.7)
    if (cl + 1) > (k - np.ceil(k/2)):
        plt.xlabel("Years")
    if cl%(np.ceil(k/2)) == 0:
        plt.ylabel("Yield in t/ha")
    plt.title("Cluster " + str(cl + 1))
    plt.ylim([0, 5.5])
plt.suptitle("Cluster average of GDHY " + \
         "yields (k = " + str(k) + ") and trend " + \
         "with 95% confidence interval for " + crops[0] + " (" + \
         cols[0] + ") and " + crops[1] + " (" + cols[1] + ")")
fig.savefig("InputData/Visualization/k" + str(k) + \
        "AvgYieldTrends.png", bbox_inches = "tight", \
        pad_inches = 0.5)   
plt.close()
