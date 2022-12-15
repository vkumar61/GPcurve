print('main file working')
#imports
import sampler
import numpy as np
import readData
import os

#hyper parameters
covL = 50
covLambda = 1

#number of samples to generate
nIter = 100
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))

#load vectors from read csv File
dataVect, dataVectIndex, deltaT = readData.dataReader('C:/Users/vkuma/Research/LearningDiffusionMaps/LargeDatasets/dataset1/movie010.txt')
print('data was read safely and there are ' + str(max(set(dataVectIndex))) + ' trajectories')

#dataVect = dataVect*97
#generate samples
sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)
print(deltaT)