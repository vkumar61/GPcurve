print('main file working')
#imports
import sampler
import numpy as np
import readData
import os
cwd = os.getcwd()
print(cwd)

#hyper parameters
covL = 40
covLambda = 1

#number of samples to generate
nIter = 1000
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))

#load vectors from read csv File
datapath = 'C:/Users/vkuma/Research/LearningDiffusionMaps/LargeRealDatasets/dataset1/movie001.txt'
# datapath = os.environ["DATAPATH"] + "Diffusion/movie001.txt"
dataVect, dataVectIndex, deltaT = readData.dataReader(datapath, 4)
print(
    'Data was read safely and there are '
    + str(max(set(dataVectIndex)))
    + ' trajectories and '
    + str(len(dataVectIndex))
    + " data points"
)

#transform data to nanometers adjusting for pixel size
dataVect = dataVect*97

print(dataVect)
#generate samples
sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)
print(deltaT)