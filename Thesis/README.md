# FoodSecurityWestAfrica

## Aim

The aim of this project is a simplified two-stage stochastic optimization model for food security in West Africa. West 
Africa is spatially subdivided in a modifiable number of clusters and uncertainty is included by yield distributions 
per cluster, year and crop. The model covers a given time period (default is 2017-2041) and includes a government fund 
which is built up by farmers paying taxes on their profits which is used for payouts to guarantee a certain income for 
farmers in catastrophic years. The model output is an allocation of arable land in each year and cluster to the 
different crops. The objective is a to minimize costs while assuring government solvency (i.e. the final government 
fund should be positive) and food security (i.e. producing a certain amount of kcal every year) with given 
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
_K = 1,\,2,\,3_ or _7_ clusters in the folder **IntermediateResults**.
