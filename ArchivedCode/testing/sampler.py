import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
import functions
import objects
import time
import pickle
import h5py

#set seed
np.random.seed(42)

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
    print("Mle: " + str(variables.mle))

    #vectors to store diffusion samples and their probabilities
    dVect = []
    dVect.append(variables.dIndu)
    pVect = []
    pVect.append(variables.P)
    
    variables.epsilon = 0.5
    const = 0
    startTime = time.time()
    for i in range(nIter):
        print(f"Iteration {i+1}/{nIter} ", end="")
        t = time.time()

        decider = np.random.uniform()
        if (i % 10000) == 0:
            print(f"({time.time()-startTime:3f} s)")
            print(f"Iteration {i+1}/{nIter} ", end="")
        if decider < const:
            variables = functions.diffusionMapSampler(variables, data)
            dVect.append(variables.dIndu)
            pVect.append(variables.P)
        else:
            variables, dVectTemp, pVectTemp = functions.diffusionPointSampler(variables, data)
            dVect += list(dVectTemp)
            pVect += list(pVectTemp)
        print(f"({time.time()-t:2f} s)")
    endTime = time.time()
        
    print(str(nIter) + " samples in " + str(endTime-startTime) + " seconds." )
    print("Index # of max probability: " + str(pVect.index(max(pVect))))

    #Save Samples as h5 files and time
    startTime = time.time()
    h5f = h5py.File(str(nIter) + '(' + str(variables.covLambda) + " " + str(variables.covL) + ').h5', 'w')
    h5f.create_dataset('samples', data = dVect)
    h5f.create_dataset('prob', data = pVect)
    h5f.close()
    endTime = time.time()
    print("Time to save files: " + str(endTime-startTime))

    return()