# General Settings 
# Last modified March 27, 2022, at 10:24

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
accuracyF_rho = 0.001
accuracyS_rho = 0.001

# if penalty is found according to import/debt, what accuracy should be used 
# (share of diff between max and min import/debt)
accuracy_help = 0.001

# should model progress be logged?
logs_on = True
# should model progress be reported in console?
console_output = True

# figsize used for all figures
figsize = (24, 13.5)

# close figures after plotting
close_plots = True