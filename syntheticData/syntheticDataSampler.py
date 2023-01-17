import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
import functions
import objects
import os
import time
import csv


#read datafile
with open("C:/Users/vkuma/Downloads/data_to_share.txt") as inp:
    tempData = [i.strip().split('\t') for i in inp]

#clean the data
cleanData = []
for i in tempData:
    if i != ['']:
        cleanData.append(i)

#coordinates for trajectories
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
pVect.append(variables.P)

startTime = time.time()
for i in range(1):
    variables = functions.diffusionSampler(variables, data)
    dVect.append(variables.dIndu)
    pVect.append(variables.P)
endTime = time.time()

plot1 = functions.plots(variables, dVect, pVect, data)
plot2 = functions.probPlot(pVect)
plot3 = functions.probPlot(pVect[100:])
plot4 = functions.meanPlot(variables, dVect, data)

plot1.savefig('map450.png')
plot2.savefig('prob450.png')
plot3.savefig('probTrunc450.png')
plot4.savefig('mean450.png')

print(variables.dInduPrior)
print(endTime-startTime)
print(pVect.index(max(pVect)))

startTime = time.time()
np.savetxt("samples" + str(variables.covL) + " " + str(variables.covLambda) + ".csv", dVect, delimiter=", ", fmt="% f")
np.savetxt("probability" + str(variables.covL) + " " + str(variables.covLambda) + ".csv", pVect, delimiter=", ", fmt="% f")
endTime = time.time()
print(endTime-startTime)