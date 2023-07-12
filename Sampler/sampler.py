import numpy as np
from types import SimpleNamespace
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
    print("Initialization Sucessful: " + str(endTime - startTime))
    print("The flat MLE is: " + str(variables.mle))

    #vectors to store diffusion samples and their initial probabilities
    dVect = []
    dVect.append(variables.dIndu)
    pVect = []
    pVect.append(variables.P)
    
    #redefine perturbation magnitude for samples
    variables.epsilon = 0.2
    startTime = time.time()

    #iterate over the number of loops sampling from the point sampler
    for i in range(nIter):
        print(f"Iteration {i+1}/{nIter}", end=" ")
        t = time.time()
        variables, dVectTemp, pVectTemp = functions.diffusionPointSampler(variables, data)
        dVect += list(dVectTemp)
        pVect += list(pVectTemp)
        print(f"({time.time()-t:.3f}s)")
    endTime = time.time()
        
    print(str(nIter) + " iterations in " + str(endTime-startTime) + " seconds." )
    print("Total # of samples: " + str(len(pVect)))
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