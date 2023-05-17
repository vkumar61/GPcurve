print('main file working')
#imports
import syntheticDataSampler
import numpy as np
import os

#hyper parameters
covL = 45
covLambda = 1
nIter = 10000
print('Will attempt to run ' + str(nIter) + ' iterations with length parameter ' + str(covL))


syntheticDataSampler.analyze(nIter, covLambda, covL)