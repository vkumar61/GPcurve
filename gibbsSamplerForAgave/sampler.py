import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
import syntheticData
import functions
import objects
import os
import time

startTime = time.time()
# Initialize variables
generationParam, data = syntheticData.dataGenerator(syntheticData.SYNTHETICPARAMETERS, objects.DATA)

variables = functions.initialization(objects.PARAMETERS, data)

endTime = time.time()
print(endTime-startTime)

#vectors to store diffusion samples and their probabilities
dVect = []
dVect.append(variables.dIndu)
pVect = []

startTime = time.time()

for i in range(60):
    variables = functions.diffusionSampler(variables, data)
    dVect.append(variables.dIndu)
    pVect.append(variables.P)

endTime = time.time()
print(endTime-startTime)

plot1 = syntheticData.plots(variables, generationParam, dVect, pVect)
plot2 = functions.probPlot(pVect)


plot1.savefig('C:/Users/User/Documents/GitHub/GPcurve/gibbsSamplerForAgave/syntheticFigures/plot.png')
plot2.savefig('C:/Users/User/Documents/GitHub/GPcurve/gibbsSamplerForAgave/syntheticFigures/prob.png')