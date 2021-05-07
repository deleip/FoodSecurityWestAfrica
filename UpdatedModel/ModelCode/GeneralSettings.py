# General Settings 
# Last modified May 07, 2021, at 23:42
# (reset to original values)

# accuracy demanded from the target probabilities (given as share of
# target probability)
accuracyF_targetProb = 0.005
accuracyS_targetProb = 0.005

# accuracy demanded from the maximum probabilities (given as share of
# maximum probability)
accuracyF_maxProb = 0.01
accuracyS_maxProb = 0.01

# accuracy of the penalties given thorugh size of the accuracy interval:
# the size needs to be smaller than final rho / shareDiff
shareDiffF = 10
shareDiffS = 10

# if penalty is found according to import/debt, what accuracy should be used 
# (share of diff between max and min import/debt)
accuracy_help = 0.01

# should model progress be logged?
logs_on = True
# should model progress be reported in console?
console_output = True

# figsize used for all figures
figsize = (24, 13.5)

# close figures after plotting
close_plots = True