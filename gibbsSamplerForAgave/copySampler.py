import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
import syntheticData
import functions
import objects
import os
import time


#read datafile
with open("C:/Users/User/Downloads/data_to_share.txt") as inp:
    tempData = [i.strip().split('\t') for i in inp]

cleanData = []
for i in tempData:
    if i != ['']:
        cleanData.append(i)


x = np.array([float(i[2]) for i in cleanData])
y = np.array([float(i[3]) for i in cleanData])

# Initialize variables
data = SimpleNamespace(**objects.DATA)
data.trajectoriesIndex = np.array([int(i[0]) for i in cleanData])
data.trajectories = np.vstack((x,y)).T
data.deltaT = 0.0001
data.nData = len(data.trajectoriesIndex)
data.nTrajectories = np.unique(data.trajectoriesIndex)

variables = functions.initialization(objects.PARAMETERS, data)

#vectors to store diffusion samples and their probabilities
dVect = []
dVect.append(variables.dIndu)
pVect = []

startTime = time.time()
for i in range(10000):
    variables = functions.diffusionSampler(variables, data)
    dVect.append(variables.dIndu)
    pVect.append(variables.P)
endTime = time.time()

plot1 = functions.plots(variables, dVect, pVect, data)
plot2 = functions.probPlot(pVect)

plot1.savefig('C:/Users/User/Documents/GitHub/GPcurve/gibbsSamplerForAgave/figures/MAP.png')
plot2.savefig('C:/Users/User/Documents/GitHub/GPcurve/gibbsSamplerForAgave/figures/prob.png')

print(endTime-startTime)