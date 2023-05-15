import numpy as np
from types import SimpleNamespace
#from scipy import stats
#import matplotlib.pyplot as plt
import functions
import objects
import time
import pickle

def analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL):
    print('Inititalization Started')
    startTime = time.time()
    #initialize data and variables
    data = SimpleNamespace(**objects.DATA)
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = np.unique(data.trajectoriesIndex)
    variables = functions.initialization(objects.PARAMETERS, data, covLambda, covL)
    endTime = time.time()
    file = open(str(nIter) + " " + str(variables.covLambda) + " " + str(variables.covL) + "variables.pkl","wb")
    pickle.dump(variables, file) 
    file.close()
    file = open(str(nIter) + " " + str(variables.covLambda) + " " + str(variables.covL) + "data.pkl","wb")
    pickle.dump(data, file) 
    file.close()
    print("Initialization Done: " + str(endTime - startTime))
    print("Mle: " + str(variables.dInduPrior))

    #vectors to store diffusion samples and their probabilities
    dVect = []
    dVect.append(variables.dIndu)
    pVect = []
    pVect.append(variables.P)
    
    startTime = time.time()
    for i in range(nIter):
        decider = np.random.uniform()
        if (i % 10000) == 0:
            print(str(i) + ' samples taken')
        if decider < 0.9:
            variables = functions.diffusionMapSampler(variables, data)
            dVect.append(variables.dIndu)
            pVect.append(variables.P)
        else:
            variables = functions.diffusionPointSampler(variables, data)
            dVect.append(variables.dIndu)
            pVect.append(variables.P)
    endTime = time.time()
        
    print(str(nIter) + " samples in " + str(endTime-startTime) + " seconds." )
    print("Index # of max probability: " + str(pVect.index(max(pVect))))

    #Save Samples as CSV files
    startTime = time.time()
    np.savetxt(str(nIter) + "samples(" + str(variables.covLambda) + " " + str(variables.covL) + ").csv", dVect, delimiter=", ", fmt="% f")
    np.savetxt(str(nIter) + "probability(" + str(variables.covLambda) + " " + str(variables.covL) + ").csv", pVect, delimiter=", ", fmt="% f")
    endTime = time.time()
    print("Time to save files: " + str(endTime-startTime))
    
    return()