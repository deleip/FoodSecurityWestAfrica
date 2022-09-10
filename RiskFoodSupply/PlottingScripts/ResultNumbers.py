# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:37:35 2022

@author: leip
"""


# set the right directory
import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(dir_path)

# import all project related functions
import FoodSecurityModule as FS  

# import other modules
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_uppercase as letter
from datetime import datetime
from scipy.stats import norm

from PlottingScripts.PlottingSettings import cluster_letters


if not os.path.isdir("Figures/PublicationPlots/ResultNumbers"):
    os.mkdir("Figures/PublicationPlots/ResultNumbers")


# %% ##########################################################################

#### set up file

if os.path.exists("Figures/ResultNumbers.txt"):
    os.remove("Figures/ResultNumbers.txt")
    
res = open("Figures/ResultNumbers.txt", "a")
res.write("Numbers for results part of paper; created on " + str(datetime.now().strftime("%B %d, %Y, at %H:%M")) + "\n\n")
res.close()


#### 1. max. percentage of food shortage in risk neutral scenario

print("1. max. percentage of food shortage in risk neutral scenario \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("-"*65 + "\n\n")
res.write("1. max. percentage of food shortage in risk neutral scenario" + "\n\n")
res.close()

p = "fixed"
y = "fixed"
alpha = 0.5

pl1 = plt.figure(figsize = (14, 8))
    
pl1.subplots_adjust(hspace = 0.5, wspace = 0.1)

for cl in range(1, 10):
    pos = letter.index(cluster_letters[cl-1]) + 1
    ax = pl1.add_subplot(3, 3, pos)
    
    # get results
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = y,
                                   pop_scenario = p,
                                   probF = alpha)

    data = meta_sol["food_supply"].flatten()/args["demand"][0] * 100
    
    mu, std = norm.fit(data[~np.isnan(data)])
    minFood = np.min(data[~np.isnan(data)])
    
    plt.hist(meta_sol["food_supply"].flatten()/args["demand"][0] * 100, bins = 200, alpha = 0.6,
              density = True)
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf, "k", linewidth = 2)
    title = r"Region " + cluster_letters[cl-1] + ": mu = %.2f, std = %.2f" % (mu, std)
    plt.title(title)
    
    res = open("Figures/ResultNumbers.txt", "a")
    res.write(" Region " +  cluster_letters[cl-1]  + ": mu = %.2f, std = %.2f, min = %.2f" % (mu, std, minFood) + "\n")
    res.close()
    
    
    
pl1.savefig("Figures/PublicationPlots/ResultNumbers/ShortageRiskNeutral.jpg", 
            bbox_inches = "tight", pad_inches = 0.2, format = "jpg")


#### 2. max. percentage of food shortage in risk neutral scenario

print("2. max. percentage of food shortage in risk neutral scenario \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("2. number clusters that can reach reliability level" + "\n\n")
res.close()

probs90 = []
probs99 = []

for cl in range(1, 10):
    
    # get results
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = y,
                                   pop_scenario = p,
                                   probF = 0.9)

    prob90 = meta_sol["probF"]
    probs90.append(prob90)
    
    settings, args, yield_information, population_information, penalty_methods,  \
    status, all_durations, exp_incomes, crop_alloc, meta_sol, \
    crop_allocF, meta_solF, crop_allocS, meta_solS, \
    crop_alloc_vss, meta_sol_vss, VSS_value, validation_values, fn = \
                FS.LoadFullResults(k_using = cl,
                                   yield_projection = y,
                                   pop_scenario = p,
                                   probF = 0.99)

    prob99 = meta_sol["probF"]
    probs99.append(prob99)
    
    res = open("Figures/ResultNumbers.txt", "a")
    res.write(" Region " +  cluster_letters[cl-1]  + ": resulting prob. for 90/99%% target reliability = %.1f/%.1f%%" % (prob90 * 100, prob99 * 100) + "\n")
    res.close()
    
res = open("Figures/ResultNumbers.txt", "a")
res.write("\n Number of clusters reaching reliability of 90%% = %d" % (sum(np.round(np.array(probs90),2) >= 0.9)))
res.write("\n Number of clusters reaching reliability of 99%% = %d" % (sum(np.round(np.array(probs99),2) >= 0.99)) + "\n")
res.close()

#### 3. Additional costs for reliability

print("3. Additional costs for reliability \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("3. Additional costs for reliability" + "\n\n")
res.close()


tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                out_type = "agg_sum",
                                sizes = 1,
                                yield_projection = "trend",
                                pop_scenario = "fixed",
                                probF = 0.5)
        
costs50_best = tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0]

tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                out_type = "agg_sum",
                                sizes = 1,
                                yield_projection = "trend",
                                pop_scenario = "fixed",
                                probF = 0.9)
        
costs90_best = tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0]
 
tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                out_type = "agg_sum",
                                sizes = 1,
                                yield_projection = "trend",
                                pop_scenario = "fixed",
                                probF = 0.99)
        
costs99_best = tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0]
       
tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                out_type = "agg_sum",
                                sizes = 1,
                                yield_projection = "fixed",
                                pop_scenario = "High",
                                probF = 0.5)
        
costs50_worst = tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0]

tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                out_type = "agg_sum",
                                sizes = 1,
                                yield_projection = "fixed",
                                pop_scenario = "High",
                                probF = 0.9)
        
costs90_worst = tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0]
 
tmp = FS.Panda_GetResultsSingScen(output_var = 'Total cultivation costs (sto. solution)',
                                out_type = "agg_sum",
                                sizes = 1,
                                yield_projection = "fixed",
                                pop_scenario = "High",
                                probF = 0.99)
        
costs99_worst = tmp.loc[:,"Total cultivation costs (sto. solution) - Aggregated over all groups"].values[0]
    

inc_best_90 = ((costs90_best/costs50_best) - 1) * 100
inc_best_99 = ((costs99_best/costs50_best) - 1) * 100
inc_worst_90 = ((costs90_worst/costs50_worst) - 1) * 100
inc_worst_99 = ((costs99_worst/costs50_worst) - 1) * 100

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Best-case scenario: \n")
res.write(" Cost increase from rliability level 50%% to 90/99%%= %.2f/%.2f%%" % (inc_best_90, inc_best_99) + "\n\n")
res.write(" Worst-case scenario: \n")
res.write(" Cost increase from rliability level 50%% to 90/99%% = %.2f/%.2f%%" % (inc_worst_90, inc_worst_99) + "\n")
res.close()   

#### 4. Reduction of exp. food shortage with reliability

print("4. Reduction of exp. food shortage with reliability \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("4. Reduction of exp. food shortage with reliability" + "\n\n")
res.close()


avgShortage_best50 = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                                 output_var = "Average aggregate food shortage",
                                                 out_type = "agg_sum", 
                                                 sizes = 1,
                                                 probF = 0.5,
                                                 yield_projection = "trend",
                                                 pop_scenario = "fixed").iloc[0,1]

avgShortage_best90 = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                                 output_var = "Average aggregate food shortage",
                                                 out_type = "agg_sum", 
                                                 sizes = 1,
                                                 probF = 0.9,
                                                 yield_projection = "trend",
                                                 pop_scenario = "fixed").iloc[0,1]

avgShortage_best99 = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                                 output_var = "Average aggregate food shortage",
                                                 out_type = "agg_sum", 
                                                 sizes = 1,
                                                 probF = 0.99,
                                                 yield_projection = "trend",
                                                 pop_scenario = "fixed").iloc[0,1]


avgShortage_worst50 = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                                 output_var = "Average aggregate food shortage",
                                                 out_type = "agg_sum", 
                                                 sizes = 1,
                                                 probF = 0.5,
                                                 yield_projection = "fixed",
                                                 pop_scenario = "High").iloc[0,1]

avgShortage_worst90 = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                                 output_var = "Average aggregate food shortage",
                                                 out_type = "agg_sum", 
                                                 sizes = 1,
                                                 probF = 0.9,
                                                 yield_projection = "fixed",
                                                 pop_scenario = "High").iloc[0,1]

avgShortage_worst99 = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                                 output_var = "Average aggregate food shortage",
                                                 out_type = "agg_sum", 
                                                 sizes = 1,
                                                 probF = 0.99,
                                                 yield_projection = "fixed",
                                                 pop_scenario = "High").iloc[0,1]


dec_best_90 = (1- (avgShortage_best90/avgShortage_best50)) * 100
dec_best_99 = (1- (avgShortage_best99/avgShortage_best50)) * 100
dec_worst_90 = (1- (avgShortage_worst90/avgShortage_worst50)) * 100
dec_worst_99 = (1- (avgShortage_worst99/avgShortage_worst50)) * 100

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Best-case scenario: \n")
res.write(" Decrease of expected food shortage from reliability level 50%% to 90/99%% = %.2f/%.2f%%" % (dec_best_90, dec_best_99) + "\n\n")
res.write(" Worst-case scenario: \n")
res.write(" Decrease of expected food shortage from reliability level 50%% to 90/99%% = %.2f/%.2f%%" % (dec_worst_90, dec_worst_99) + "\n")
res.close()   


#### 5. Number of clusters reaching solvency target

print("5. Number of clusters reaching solvency target \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("5. Number of clusters reaching solvency target \n")
res.close()

for risk in [0.01, 0.05]:
    for tax in [0.01, 0.05, 0.1]:
        res = open("Figures/ResultNumbers.txt", "a")
        res.write("\n For risk = %.d%%, tax = %d%%: " % (risk * 100, tax * 100))
        res.close()

        tmp = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                        output_var = "Resulting probability for solvency",
                                        out_type = "all", 
                                        sizes = 1,
                                        yield_projection = "fixed",
                                        pop_scenario = "fixed",
                                        tax = tax,
                                        risk = risk).iloc[0,1]

        res = open("Figures/ResultNumbers.txt", "a")
        res.write("%d" % (sum(np.round(np.array(tmp),2) >= 0.9)) + " cluster \n")
        res.write("       " + str(np.round(np.array(tmp),2)) + "\n")        
        res.close()
        

#### 6. Variation of solvency probability for low tax and high risk

print("6. Variation of solvency probability for low tax and high risk \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("6. Variation of solvency probability for low tax (1%%) and high risk (5%%) \n\n")
res.close()

tmp = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                output_var = "Resulting probability for solvency",
                                out_type = "all", 
                                sizes = 1,
                                yield_projection = "fixed",
                                pop_scenario = "fixed",
                                tax = 0.01,
                                risk = 0.05).iloc[0,1]

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Min. solvency probability = %.2f%%, max. solvency probability = %.2f%%" % (min(tmp) * 100, max(tmp) * 100))
res.close()

#### 7. Overall reliability without cooperation

print("7. Overall reliability without cooperation \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("7. Overall reliability without cooperation \n\n")
res.close()

prob_best = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                output_var = "Resulting probability for food security",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                sizes = 1,
                                yield_projection = "trend",
                                pop_scenario = "fixed").iloc[0,1]

prob_worst = FS.Panda_GetResultsSingScen(file = "current_panda", 
                                output_var = "Resulting probability for food security",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                sizes = 1,
                                yield_projection = "fixed",
                                pop_scenario = "High").iloc[0,1]

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Overall reliability for best-case = %.2f%%, for worst-case = %.2f%%" % (prob_best * 100, prob_worst * 100))
res.close()

#### 8. Overall reliability for different cooperation levels

print("8. Overall reliability for different cooperation levels \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("8. Overall reliability for different cooperation levels \n")
res.close()

for metric in ["equality", "medoids"]:
    res = open("Figures/ResultNumbers.txt", "a")
    res.write("\n Grouping according to " + metric + "\n")
    res.close()    
    
    for (y, p, scen) in [("fixed", "High", "worst-case"), 
                ("fixed", "fixed", "stationary"),
                ("trend", "fixed", "best-case")]:

        tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                        output_var = "Resulting probability for food security",
                                        out_type = "agg_avgweight", 
                                        var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                        yield_projection = y,
                                        pop_scenario = p,
                                        grouping_aim = "Similar",
                                        grouping_metric = metric)[0].iloc[:,1])
        
        res = open("Figures/ResultNumbers.txt", "a")
        res.write("      " + scen + ": " + str(np.round(np.array(tmp), 2)) + "\n")
        res.close()    
     
#### 9. Solvency probabilities with cooperation

print("9. Solvency probabilities with cooperation \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("9. Solvency probabilities with cooperation \n")
res.close()

for metric in ["equality", "medoids"]:
    res = open("Figures/ResultNumbers.txt", "a")
    res.write("\n Grouping according to " + metric + "\n")
    res.close()    
    
    for (y, p, scen) in [("fixed", "High", "worst-case"), 
                ("fixed", "fixed", "stationary"),
                ("trend", "fixed", "best-case")]:

        tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                        output_var = "Resulting probability for solvency",
                                        out_type = "agg_avgweight", 
                                        var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                        yield_projection = y,
                                        pop_scenario = p,
                                        grouping_aim = "Similar",
                                        grouping_metric = metric)[0].iloc[:,1])
        
        res = open("Figures/ResultNumbers.txt", "a")
        res.write("      " + scen + ": " + str(np.round(np.array(tmp), 2)) + "\n")
        res.close()    
 
    
tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                output_var = "Resulting probability for solvency",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                yield_projection = "trend",
                                pop_scenario = "fixed",
                                grouping_aim = "Similar",
                                grouping_metric = "medoids")[0].iloc[:,1])

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n Increase of solvency probability in best-case: %.2f" % ((tmp[-1] - tmp[0]) * 100) + " percentage points \n")
res.close()  

tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                output_var = "Resulting probability for solvency",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                yield_projection = "fixed",
                                pop_scenario = "fixed",
                                grouping_aim = "Similar",
                                grouping_metric = "medoids")[0].iloc[:,1])

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Increase of solvency probability in  stationary case: %.2f" % ((tmp[-1] - tmp[0]) * 100) + " percentage points \n")
res.close()  

tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                output_var = "Resulting probability for solvency",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                yield_projection = "fixed",
                                pop_scenario = "High",
                                grouping_aim = "Similar",
                                grouping_metric = "medoids")[0].iloc[:,1])

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Increase of solvency probability in worst-case: %.2f" % ((tmp[-1] - tmp[0]) * 100) + " percentage points \n")
res.close()  

#### 10. Reduction of debt by full cooperation

print("10. Reduction of debt by full cooperation \n", flush = True)

res = open("Figures/ResultNumbers.txt", "a")
res.write("\n" + "-"*65 + "\n\n")
res.write("10. Reduction of debt by full cooperation \n\n")
res.close()

tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                output_var = "Average aggregate debt after payout per capita (including only samples with negative final fund)",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                yield_projection = "trend",
                                pop_scenario = "fixed",
                                grouping_aim = "Similar",
                                grouping_metric = "medoids")[0].iloc[:,1])

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Reduction by factor %.2f in best-case" % (tmp[0]/tmp[-1]) + "\n")
res.close()  

tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                output_var = "Average aggregate debt after payout per capita (including only samples with negative final fund)",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                yield_projection = "fixed",
                                pop_scenario = "fixed",
                                grouping_aim = "Similar",
                                grouping_metric = "medoids")[0].iloc[:,1])

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Reduction by factor %.2f in stationary case" % (tmp[0]/tmp[-1]) + "\n")
res.close()  

tmp = list(FS.Panda_GetResults(file = "current_panda", 
                                output_var = "Average aggregate debt after payout per capita (including only samples with negative final fund)",
                                out_type = "agg_avgweight", 
                                var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                                yield_projection = "fixed",
                                pop_scenario = "High",
                                grouping_aim = "Similar",
                                grouping_metric = "medoids")[0].iloc[:,1])

res = open("Figures/ResultNumbers.txt", "a")
res.write(" Reduction by factor %.2f in worst-case" % (tmp[0]/tmp[-1]) + "\n")
res.close()  
