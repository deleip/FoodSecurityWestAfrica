# Default Model Settings 
# Last modified May 24, 2021, at 18:59
# (reset to original values)

# "prob" if desired probabilities are given and penalties are to be 
# calculated accordingly. "penalties" if input penalties are to be used 
# directly 
PenMet = "prob" 

# demanded probabilities for food security and solvency 
probF = 0.99 
probS = 0.95 

# penalties (if PenMet == "prob" used as initial guesses to calculate the 
# correct penalties) 
rhoF = None 
rhoS = None 

# specification whether solvency constraint should be included in model
solv_const = "off"

# number of clusters in which the area is devided 
k = 9 

# clusters considered in the model run 
k_using = [3] 

# number of crops considered 
num_crops = 2 

# yield projections to use ("fixed" or "trend") 
yield_projection = "fixed" 

# first year of simulation 
sim_start = 2017 

# population scenario to use ('fixed', 'Medium', 'High', 'Low', 
# 'ConstantFertility', 'InstantReplacement', 'ZeroMigration' 
# 'ConstantMortality', 'NoChange' and 'Momentum') 
pop_scenario = "fixed" 

# risk level that is covered by the government 
risk = 0.05 

# sample size 
N = 10000 

# sample siize for validation (if None, no validation is done) 
validation_size = None 

# number of years to simulate 
T = 20 

# seed for generation of yield samples 
seed = 201120 

# tax rate to be paied on farmers profit 
tax = 0.01 

# the percentage that determines how high the guaranteed income will be 
# depending on the expected income 
perc_guaranteed = 0.9 

# tax rate to be paied on farmers profit 
ini_fund = 0 

# food import that will be subtracted from demand in each year 
food_import = 0 

# accuracy demanded from the target probabilities (given as share of
# target probability)
accuracyF_demandedProb = 0.001
accuracyS_demandedProb = 0.001

# accuracy demanded from the maximum probabilities (given as share of
# maximum probability))
accuracyF_maxProb = 0.001
accuracyS_maxProb = 0.001

# accuracy of the penalties given thorugh size of the accuracy interval:
# the size needs to be smaller than final rho * shareDiff
accuracyF_rho = 0.01
accuracyS_rho = 0.01

# if penalty is found according to import/debt, what accuracy should be used 
# (share of diff between max and min import/debt)
accuracy_help = 0.002

