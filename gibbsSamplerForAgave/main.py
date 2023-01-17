print('main file working')
#imports
import sampler
import numpy as np
import readData
import os

#hyper parameters
covL = 40
covLambda = 1

#number of samples to generate
nIter = 100
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))

#load vectors from read csv File
dataVect, dataVectIndex, deltaT = readData.dataReader('C:/Users/vkuma/Research/LearningDiffusionMaps/LargeDatasets/dataset1/movie001.txt', 150000)
print('data was read safely and there are ' + str(max(set(dataVectIndex))) + ' trajectories and ' + str(len(dataVectIndex)) + " data points")

#transform data to micrometers adjusting for pixel size
dataVect = dataVect*97/100

#generate samples
sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)
print(deltaT)