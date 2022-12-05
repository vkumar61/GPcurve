print('main file working')
#imports
import sampler
import numpy as np
import readData
import os

#hyper parameters
covL = 20
covLambda = 1

#number of samples to generate
nIter = 100000
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))

#load vectors from read csv File
dataVect, dataVectIndex, deltaT = readData.dataReader(os.getcwd() + '/dataset1/movie010.txt', 4)
print('data was read safely and there are ' + str(max(set(dataVectIndex))) + ' trajectories')

#generate samples
#sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)
print(np.shape(dataVect))
print(np.shape(dataVectIndex))
print(deltaT)