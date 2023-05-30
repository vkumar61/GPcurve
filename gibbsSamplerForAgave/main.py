print('main file working')
#imports
import sampler
import numpy as np
import readData
import os
cwd = os.getcwd()
print(cwd)

#Choose your hyperparameters here, if None, their value will be set later
covL = None
covLambda = None

#number of samples to generate
nIter = 1000
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))

#load vectors from read csv File
datapath = 'C:/Users/vkuma/Research/LearningDiffusionMaps/SyntheticData/syntheticData_20230524_181523/data.csv'
datapath = os.environ["DATAPATH"] + "Diffusion/data.csv"
dataVect, dataVectIndex, deltaT = readData.dataReader(datapath, 1)

print(
    'Data was read safely and there are '
    + str(max(set(dataVectIndex)))
    + ' trajectories and '
    + str(len(dataVectIndex))
    + " data points"
)

print(dataVect)
#generate samples
sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)
print(deltaT)