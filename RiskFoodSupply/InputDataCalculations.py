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
import seaborn as sns
from termcolor import colored
from scipy.stats import shapiro

from matplotlib.patches import Patch
from string import ascii_uppercase as letter
from matplotlib.lines import Line2D

import ModelCode.DataPreparation as DP
import ModelCode.GroupingClusters as GC
from ModelCode.PlotMaps import MapValues
from ModelCode.PlotMaps import PlotClusterGroups

if not os.path.isdir("InputData/Visualization"):
    os.mkdir("InputData/Visualization")

from PlottingScripts.PlottingSettings import publication_colors
from PlottingScripts.PlottingSettings import cluster_letters

print("Preparing input data ...", flush = True)

# %% 1. Profitable Areas

print("... profitable area", flush = True)

# Only including cells, where either maize or rice has profitable average 
# yields (based on linear regression evaluated for baseyear 2016)

DP.ProfitableAreas()
# creates ProcessedData/MaskProfitableArea.txt
    
# combine with SPEI mask
with open("ProcessedData/MaskProfitableArea.txt", "rb") as fp:    
    mask_profitable = pickle.load(fp)
with open("ProcessedData/mask_SPEI03_WA.txt", "rb") as fp:    
    mask_SPEI = pickle.load(fp)
 
maskAreaUsed = mask_profitable * mask_SPEI

with open("InputData/Other/MaskAreaUsed.txt", "wb") as fp:    
    pickle.dump(maskAreaUsed, fp)
# creates InputData/Other/MaskAreaUsed.txt

# plot map of area used 
MapValues(maskAreaUsed, title = "Area used", 
          file = "InputData/Visualization/AreaUsed", plot_cmap = False)
    
# %% 2. Average farm gate prices

print("... average farm gate prices", flush = True)

# based on prepared price dataset and weighted with areas

with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
    MaskAreaUsed = pickle.load(fp)
    
prices = DP.CalcAvgProducerPrices(rice_mask = MaskAreaUsed, maize_mask = MaskAreaUsed)

with open("InputData/Prices/RegionFarmGatePrices.txt", "wb") as fp:    
    pickle.dump(prices, fp)
# creates InputData/Prices/RegionFarmGatePrices.txt


# %% 3. Average calorie demand per person and day

print("... average calorie demand per person and day", flush = True)

# based on some country values (ProcessedData/CountryCaloricDemand.csv)
# we calculate the average caloric demand per person and day, using area as weight
                
DP.AvgCaloricDemand()      
# creates InputData/Other/AvgCaloricDemand.txt

# %% 4. calculate pearson distance between all cells

print("... Pearson Distance between cells", flush = True)

with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
    MaskAreaUsed = pickle.load(fp)
    
DP.CalcPearsonDist(MaskAreaUsed)
# creates InputData/Other/PearsonCorrSPEI03.txt
# creates InputData/Other/PearsonDistSPEI03.txt

# %% 5. run clustering algorithm 

print("... clustering algorithm", flush = True)

# k-medoids algrithm based on gridded SPEI values

with open("InputData/Other/MaskAreaUsed.txt", "rb") as fp:    
    MaskAreaUsed = pickle.load(fp)
    
with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
    pearsonDist = pickle.load(fp)
    
for k in range(1, 20):
    print(k, flush = True)
    DP.kMedoids(k, pearsonDist, MaskAreaUsed, "PearsonDistSPEI")
    PlotClusterGroups(k = k, title = "Division in " + str(k) + " cluster", 
              file = "InputData/Visualization/MapkMediods" + str(k))
# creates Inputdata/Clusters/Clustering/kMedoidsX_PearsonDistSPEI.txt    

# %% 6. best number of cluster

print("... best number of clusters", flush = True)

with open("InputData/Other/PearsonDistSPEI03.txt", "rb") as fp:    
    pearsonDist = pickle.load(fp)  

between_all = []
between_closest = []
within_cluster = []
kmax = 19

# calculate the distances within clusters and between clusters
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
title = ["a. Comparison using all clusters for SPEI", \
         "Comparison using closest clusters for SPEI"]

# plot distances for different numbers of clusters
version = ["All", "Closest"]
for i in range(0,2):
    fig = plt.figure(figsize = (18, 9))
    ax = fig.add_subplot(1,2,1) 
    ax.scatter(dists_within[i], dists_between[i],  c=range(2, kmax + 1))
    plt.title("a. Inter- and intra-cluster similarity", fontsize = 22)
    plt.xlabel("Similarity within clusters", fontsize = 16)
    plt.ylabel("Similarity between clusters", fontsize = 16)
    plt.xticks(fontsize =14)
    plt.yticks(fontsize =14)
    for t, txt in enumerate(np.arange(2, kmax + 1, 4)):
        ax.annotate(txt, (dists_within[i][3*t] - 0.0024, \
                          dists_between[i][3*t] + 0.0029)) 
    fig.add_subplot(1,2,2) 
    metric, cl_order = DP.MetricClustering(dists_within[i], dists_between[i], \
                                        refX = 0, refY = max(dists_between[i])) 
    plt.scatter(cl_order, metric)
    plt.xticks(range(2, kmax + 1), fontsize =14)
    plt.yticks(fontsize =14)
    plt.title("b. Quantification of tradeoff", fontsize = 22)
    plt.xlabel("Number of clusters", fontsize = 16)
    plt.ylabel("Euclidean distance to (0, "+ \
                                  str(np.round(max(dists_between[i]), 2)) + \
                                  ") in figure a.", fontsize = 16)
    # plt.suptitle("Tradeoff of distances within and between cluster")
        
    fig.savefig("InputData/Visualization/kMediods_ScatterInterVsIntraCluster" +\
                 version[i] + ".png", bbox_inches = "tight", pad_inches = 0.5)      
        
# creates InputData/Visualization/kMediods_ScatterInterVsIntraClusterAll.png
#         InputData/Visualization/kMediods_ScatterInterVsIntraClusterClosest.png
# Using the average over all clusters, the more clusters the better the result
# of our metric
# Using only the closest cluster, the optimum lies at 9 cluster.   
       
# %% 7. Adjacency matrix

print("... adjacency matrix \n        " + \
      colored("!!This was included manually, needs to be change if different" + \
              "input data or number of clusters is used!!", "red"), flush = True)

# manually setting up an adjacency matrix (to be used when making cluster groups)

with open("InputData/Clusters/Clustering/kMediods9_PearsonDistSPEI.txt", "rb") as fp:  
    clusters = pickle.load(fp)
    
fig = plt.figure()

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

print("... Cluster groupinds", flush = True)

for aim in ["Similar", "Dissimilar"]:
    for adj in [True, False]:
        for s in [1, 2, 3, 5, 9]:
            ShiftedGrouping, BestCosts, valid = GC.GroupingClusters(k = 9, size = s, aim = aim, adjacent = adj, 
                                 title = None, figsize = None)
# creates GroupingSizeX(Dis)Similar(Adj).txt  
 
# %% 9. Yield trends

print("... yield trends", flush = True)

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
    
cols = [publication_colors["green"], publication_colors["yellow"]]
start_year = 1981
len_ts = yields_avg.shape[0]

fig = plt.figure(figsize = (16, 11))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                wspace=0.15, hspace=0.35)
years = range(start_year, start_year + len_ts)
ticks = np.arange(start_year + 2, start_year + len_ts + 0.1, 8)
for cl in range(0, k):
    pos = letter.index(cluster_letters[cl-1]) + 1
    if k > 6:
        ax = fig.add_subplot(3, int(np.ceil(k/3)), pos)
    elif k > 2:
        ax = fig.add_subplot(2, int(np.ceil(k/2)), pos)
    else:
        ax = fig.add_subplot(1, k, pos)
    dict_labels = {}
    for cr in [0, 1]:
        sns.regplot(x = np.array(range(start_year, \
              start_year + len_ts)), y = yields_avg[:, cr, cl], \
              color = cols[cr], ax = ax, marker = ".", truncate = True)
    plt.plot(range(start_year, start_year + len_ts), \
             np.repeat(threshold[0], len_ts), ls = "--", 
             color = cols[0], alpha = 0.85)
    plt.plot(range(start_year, start_year + len_ts), \
             np.repeat(threshold[1], len_ts), ls = "--", 
             color = cols[1], alpha = 0.85)
    val_max = np.max(yields_avg[:,0,cl])
    ax.set_ylim(bottom = -0.05 * val_max)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(ticks)   
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_fontsize(14)
    # if cl > 5:
    #     plt.xlabel("Years")
    # if cl%3 == 0:
    #     plt.ylabel("Yield in t/ha")
    plt.title("Region "  + cluster_letters[cl-1], fontsize = 18)
    # plt.ylim([0, 5.5])
# plt.suptitle("Cluster average of GDHY " + \
         # "yields (k = " + str(k) + ") and trend " + \
         # "with 95% confidence interval for " + crops[0] + " (" + \
         # cols[0] + ") and " + crops[1] + " (" + cols[1] + ")")
         
  
# add a big axis, hide frame, ticks and tick labels of overall axis
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Year", fontsize = 22, labelpad = 18)
plt.ylabel("Crop yield, t/ha", fontsize = 22, labelpad = 18)


legend_elements = [Line2D([0], [0], lw = 2, ls = "dashed", color= "black",
                          label="Threshold for profitability"),
                    Patch(color = publication_colors["green"], label='Rice', alpha = 0.6),
                    Patch(color = publication_colors["yellow"], label='Maize', alpha = 0.6)]
ax.legend(handles = legend_elements, fontsize = 18, bbox_to_anchor = (0.5, -0.1),
          loc = "upper center")


fig.savefig("InputData/Visualization/k" + str(k) + \
        "AvgYieldTrends.png", bbox_inches = "tight")   
plt.close()



# shapiro normality test for residuals
with open("InputData/YieldTrends/DetrYieldAvg_k" + str(k) + ".txt", "rb") as fp:  
    yields_avg = pickle.load(fp)
    avg_pred = pickle.load(fp)
    residuals = pickle.load(fp)
    residual_means = pickle.load(fp)
    residual_stds = pickle.load(fp)
    fstat = pickle.load(fp)
    constants = pickle.load(fp)
    slopes = pickle.load(fp)
    crops = pickle.load(fp)
    years = pickle.load(fp)

shapiro_statistics = np.empty([2, 9])
shapiro_pvalues = np.empty([2, 9])
for cl in range(0, k):
    for cr in [0, 1]:
        shapiro_statistics[cr, cl] = shapiro(residuals[:, cr, cl])[0]
        shapiro_pvalues[cr, cl] = shapiro(residuals[:, cr, cl])[1]


cols = [publication_colors["green"], publication_colors["yellow"]]
fig = plt.figure(figsize = (16, 11))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                wspace=0.15, hspace=0.35)
for cl in range(0, k):
    pos = letter.index(cluster_letters[cl-1]) + 1
    if k > 6:
        ax = fig.add_subplot(3, int(np.ceil(k/3)), pos)
    elif k > 2:
        ax = fig.add_subplot(2, int(np.ceil(k/2)), pos)
    else:
        ax = fig.add_subplot(1, k, pos)
    for cr in [0, 1]:
        plt.hist(residuals[:, cr, cl], alpha = 0.7, color = cols[cr])
    plt.title("Region "  + cluster_letters[cl-1] + ", p_rice: " + 
              str(round(shapiro_pvalues[0, cl], 3)) + ", p_maize: " + str(round(shapiro_pvalues[1, cl], 3)), fontsize = 14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_fontsize(14)

# add a big axis, hide frame, ticks and tick labels of overall axis
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Yield residuals", fontsize = 22, labelpad = 18)
plt.ylabel("Density (sample size: 36)", fontsize = 22, labelpad = 18)


fig.savefig("InputData/Visualization/k" + str(k) + \
        "DistributionResiduals.png", bbox_inches = "tight")   
plt.close()

# legend = plt.figure(figsize  = (5, 3))
# legend_elements = [Line2D([0], [0], lw = 2, ls = "dashed", color= "black",
#                           label="Threshold for profitability"),
#                     Patch(color = "darkgreen", label='Rice', alpha = 0.6),
#                     Patch(color = "darkred", label='Maize', alpha = 0.6)
#                     ]

# ax = legend.add_subplot(1, 1, 1)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.legend(handles = legend_elements1, fontsize = 14, loc = 6)

# legend.savefig("InputData/Visualization/YieldTrendsLegend.jpg", 
#                 bbox_inches = "tight", pad_inches = 1, format = "jpg")
# plt.close(legend)