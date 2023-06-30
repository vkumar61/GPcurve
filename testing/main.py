#imports
import sampler
import readData
import numpy as np
import os

cwd = os.getcwd()

#hyper parameters
covL = None
covLambda = None

#number of samples to generate
nIter = 100

# load real data from csv File
dataPath = os.path.join(cwd, 'Data', 'SyntheticData', 'syntheticData_20230630_135158', 'data.csv')
dataVect, dataVectIndex, deltaT = readData.dataReader(dataPath, 1)

#transform data to nanometers adjusting for pixel size
dataVect = dataVect*97

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