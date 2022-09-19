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

for i in range(1000):
    variables = functions.diffusionSampler(variables, data)
    dVect.append(variables.dIndu)
    pVect.append(variables.P)

endTime = time.time()
print(endTime-startTime)

plot1 = syntheticData.plots(variables, generationParam, dVect, pVect)
plot2 = functions.probPlot(pVect)

plt.show()

np.savetxt("samples.csv", dVect, delimiter=", ", fmt="% s")
np.savetxt("probability.csv", pVect, delimiter=", ", fmt="% s")