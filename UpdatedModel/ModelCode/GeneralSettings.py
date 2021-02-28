# Last modified February 15, 2021, at 10:20
# (reset to original values)

# accuracy demanded from the probabilities as decimal places (given as float,
# not as percentage)
accuracyF = 3
accuracyS = 3

# accuracy of the penalties given thorugh size of the accuracy interval:
# the size needs to be smaller than final rho / shareDiff
shareDiffF = 10
shareDiffS = 10

# accuracy of debts used in the algorithm to find the right rhoS in cases where
# probS cannot be reached (given as the share of the difference between
# debt_bottom and debt_top)
accuracy_debt = 0.005

# accuracy of imports used in the algorithm to find the right rhoF in cases
# where probF cannot be reached (given as number of decimal places)
accuracy_import = 3

# should model progress be logged?
logs_on = True
# should model progress be reported in console?
console_output = True

# figsize used for all figures
figsize = (24, 13.5)

# close figures after plotting
close_plots = True