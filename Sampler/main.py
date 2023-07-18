#imports
import sampler
import readData
import numpy as np
import os

#set seed
np.random.seed(42)

#get working directory
cwd = os.getcwd()

#hyper parameters
covL = None
covLambda = None

#number of samples to generate
nIter = 500

# load real data from csv File
dataPath = os.path.join(cwd, 'Data' , 'CleanData', 'movie009.txt')
dataVect, dataVectIndex, deltaT = readData.dataReader(dataPath, 97)

print(
    'Data was read safely and there are '
    + str(len(np.unique(dataVectIndex)))
    + ' trajectories and '
    + str(len(dataVectIndex))
    + " data points"
)

#generate samples
sampler.analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL)
print(deltaT)