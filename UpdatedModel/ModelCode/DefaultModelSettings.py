# Default Model Settings 
# Last modified May 02, 2021, at 11:44
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

