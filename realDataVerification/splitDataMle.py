print('main file working')
#imports
import dataSplitFunctions as func
import numpy as np
import os

#hyper parameters
covL = 40
covLambda = 1

#number of samples to generate
nIter = 100
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))

#load vectors from read csv File
splitData = func.dataReader('C:/Users/vkuma/Research/LearningDiffusionMaps/LargeDatasets/dataset1/movie001.txt')

#Points of trajectory where learning is possible
func.mle()
'''
    dataCoordinates = np.empty((0,2))
    for i in range((nData-1)):
        if (trajectoriesIndex[i] == trajectoriesIndex[i+1]):
            dataCoordinates = np.vstack((dataCoordinates, trajectories[i]))

    #Points of trajectory that are "sampled"
    sampleCoordinates = np.empty((0,2))
    for i in range(1,nData):
        if (trajectoriesIndex[i] == trajectoriesIndex[i-1]):
            sampleCoordinates = np.vstack((sampleCoordinates, trajectories[i]))
            
diff = sampleCoordinates - dataCoordinates
    num = np.sum(diff * diff)
    den = 4*deltaT*len(diff)
    mle = num/den
'''