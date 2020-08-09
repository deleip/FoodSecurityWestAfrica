# FoodSecurityWestAfrica

## Aim

The aim of this project is a stylized two-stage stochastic optimization model for food security in West Africa. West 
Africa is spatially subdivided in a modifiable number of clusters and uncertainty is included by yield distributions 
per cluster, year and crop. The model covers a given time period (default is 2017-2041) and includes a government fund 
which is built up by farmers paying taxes on their profits which is used for payouts to guarantee a certain income for 
farmers in catastrophic years. The model output is an allocation of arable land in each year and cluster to the 
different crops. The objective is a to minimize costs while assuring government solvency (i.e. the final government 
fund should be positive) and food security (i.e. producing a certain amount of calories every year) with given 
probabilities each. 

## Content

- **Data.py**
- **Functions_Data.py**
- **Analysis.py**
- **Functions_Analysis.py**
- **StochasticOptimization.py**
- **Functions_StochasticOptimization.py**
- **IntermediateResults**

## Functionality

The first step is the preparation of different data sets (drought indices, AgMIP crop yield data, GDHY
crop yield data, crop calendar of 2000, CRU data on precipitation and PET, UN world population 
scenarios, SEDAC gridded world population in 2015, farm-gate prices) for further usage. This is implemented in 
**Data.py** and depends on routines defined in **Functions_Data.py**. Next, drought indices are used to 
subdivide West Africa into clusters and the optimal cluster number is determined; then a yield model is set up through 
regression analysis to estimate yield distributions for the model input. Both steps are implemented in 
**Analysis.py**, depending on routines defined in **Functions_Analysis.py**. As the underlying datasets have 
an aggregated size of over 16GB, the files **Data.py** and **Analysis.py** cannot be run based on the content 
provided on github. However, urls to the respective datasets are included in **Data.py**.

The core of the project is the stochastic optimization model. This is implemented in 
**Functions_StochasticOptimization.py**, and is run by **StochasticOptimization.py** to analyze the behavior 
of the model and the influence of the parameters that can be varied. **StochasticOptimization.py** also 
includes visualization of this analysis. The github repository provides all data necessary to run the model for 
_K = 1, 2, 3_ or _7_ clusters in the folder **IntermediateResults**.

Modifiable model settings with their default values are:

| parameter description | variable name | default value |
| --- | --- | --- |
| number of clusters | *k* | 1 |
| number of clusters that have to be catastrophic for a catastrophic year | *num_cl_cat* | 1 |
| yield scenario (“fixed” or “trend”) | *yield_projection* | "fixed" |
| first year of the simulation | *yield_year* | 2017 |
| should stylized values be used (True or False, leftover from test phase) | *stilised* | False |
| UN population scenario (“Medium”, “High”, “Low”, “ConstantFertility”, “InstantReplacement”, “ZeroMigration”, “ConstantMortality”, “NoChange”, “Momentum”) or fixed yield distributions over time (“fixed”) | *pop_scenario* | "fixed" |
| risk level (given as frequency) | *risk* | 20 |
| sample size | *N_c* | 3500 |
| number of covered years | *T_max* | 25 |
| seed for reproducible yield realizations | *seed* | 150620 |
| tax rate | *tax* | 0.03 |
| percentage of expected income that will be covered by government in case of catastrophes | *perc_guaranteed* | 0.75 |

The model can be called by the function

    crop_alloc, meta_sol, rhoF, rhoS, settings, args
        = OptimizeFoodSecurityProblem(probF, probS, rhoFini, rhoSini, **kwargs)
        
Here, `**kwargs` is a placeholder for the above listed settings. Settings that are kept at their default do not need to be included. The parameter *probF* is the probability
with which the food constraint needs to be met, *probS* is the probability with which the government needs to stay solvent. Note, that for some settings very high probabilities might be infeasible. The function will first call

    settings = DefaultSettingsExcept(**kwargs)
    
to get a dictionary of all settings, including the expected income which is calculated depending on the other settings. Then

    rhoF, rhoS = GetPenalties(settings, probF, probS, rhoFini, rhoSini)
    
will determine the correct penalties *rhoF* and *rhoS* for the given probabilities, starting with *rhoFini* and *rhoSini* as first guesses and using an algorithm based on the bisection method. Next

    x_ini, const, args, meta_cobyla, other = SetParameters(settings)
    
will prepare all model inputs depending on the settings. The array *x_ini* is an initial guess for the crop allocation, *const* defines the model constraints (i.e. crop areas need to be positive and respect the available arable area), all arguments that need to be passed to the objective function by the optimizer except the crop areas as a dictionary *args* (i.e. yield realizations, food demand, terminal years, cultivation costs etc.), technical information for the solver as a dictionary *meta_cobyla* and some additional information on the parameters for potential analysis in the dictionary *other*. Finally, the solver **scipy.optimize.fmin_cobyla** is called within the function

    crop_alloc, meta_sol, duration = OptimizeMultipleYears(x_ini, const, args, meta_cobyla, rhoF, rhoS)

This returns the optimal crop allocation *crop_alloc*, meta information *meta_sol* about the solution (e.g. the 
minimized value of the objective function, the final fund for all realizations, or the yearly shortcomings from the food 
demand for each realization), and the time the solver took to find the solution as *duration*.

The main function **OptimizeFoodSecurityProblem** finally returns the optimal crop allocation *crop_alloc*, the meta information *meta_sol*, the penalties *rhoF* and *rhoS* corresponding to the input probabilities *probF* and *probS*, the dictionary *settings* of all settings, and the dictionary *args* of all additional arguments that are passed to the objective function.

The model can be called for specific penalties *rhoF* and *rhoS* instead of given probabilities by the following combination:

    settings = DefaultSettingsExcept(**kwargs)
    x_ini, const, args, meta_cobyla, other = SetParameters(settings)
    crop_alloc, meta_sol, duration = OptimizeMultipleYears(x_ini, const, args, meta_cobyla, rhoF, rhoS)
