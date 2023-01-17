print('main file working')
#imports
import syntheticDataSampler
import numpy as np
import os

#hyper parameters
covL = 40
covLambda = 1

syntheticDataSampler.analyze(100, covLambda, covL)