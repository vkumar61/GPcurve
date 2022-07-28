import numpy as np
from types import SimpleNamespace
from scipy import stats
import functions
import objects
import matplotlib.pyplot as plt



# Initialize variables
generationParam, data = functions.dataGenerator(objects.SYNTHETICPARAMETERS, objects.DATA)

variables = functions.initialization(objects.PARAMETERS, data)

#vectors to store diffusion samples and their probabilities
dVect = []
dVect.append(variables.dIndu)
pVect = []

for i in range(10000):
    variables = functions.diffusionSampler(variables, data)
    dVect.append(variables.dIndu)
    pVect.append(variables.P)


plt.plot(pVect)
plt.show()
