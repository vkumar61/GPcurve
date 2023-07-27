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
    #Progress Marker
    print('Inititalization Started')
    
    #initialize data and variables and time them    
    startTime = time.time()
    data = SimpleNamespace(**objects.DATA)
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = len(np.unique(data.trajectoriesIndex))
    variables = functions.initialization(objects.PARAMETERS, data, covLambda, covL)
    endTime = time.time()

    #save variables and data dictionaries to pickle files for easy access when plotting
    file = open(str(nIter) + " " + str(variables.covLambda) + " " + str(variables.covL) + "variables.pkl","wb")
    pickle.dump(variables, file) 
    file.close()
    file = open(str(nIter) + " " + str(variables.covLambda) + " " + str(variables.covL) + "data.pkl","wb")
    pickle.dump(data, file) 
    file.close()

    #Progress Marker
    print("Initialization Sucessful: " + str(endTime - startTime))
    print("The flat MLE is: " + str(variables.mle))

    #vectors to store diffusion samples and their initial probabilities
    h5 = h5py.File('test.h5', 'w')
    h5.create_dataset(name='P', shape=(nIter,1), chunks=(1,1), dtype='f')
    h5.create_dataset(name='d', shape=(nIter,variables.nIndu), chunks=(1,variables.nIndu), dtype='f')
    
    #redefine perturbation magnitude for samples
    variables.epsilon = 0.1
    startTime = time.time()

    #initial temp and cooling rate for targeted annealing
    initTemp = 10
    coolRate = np.log(initTemp)/nIter

    #Burn in super aggressively but do not save the samples, updating proposal size to maintain healthy acceptance rate
    burnIter = 2*nIter
    for i in range(burnIter):
        variables.temperature = functions.expCooling(i, initTemp, coolRate)
        print(f"Iteration {i+1}/{burnIter}", end=" ")
        t = time.time()
        variables, dVectTemp, pVectTemp, accRate = functions.diffusionPointSampler(variables, data)
        dVect.append(variables.dIndu)
        pVect.append(variables.P)
        if accRate > 40:
            variables.epsilon *= 1.1
        elif accRate < 20:
            variables.epsilon *= 0.9
        print(f"({time.time()-t:.3f}s)")

    #set temperature back to 1, and iterate a few times to find ideal magnitude of proposal to maintain healthy acceptance
    variables.temperature = 1
    variables.epsilon = 0.1
    
    for i in range(50):
        print(f"Iteration {i+1}/{50}", end=" ")
        t = time.time()
        variables, dVectTemp, pVectTemp, accRate = functions.diffusionPointSampler(variables, data)
        dVect.append(variables.dIndu)
        pVect.append(variables.P)
        print(f"({time.time()-t:.3f}s)")
        if accRate > 22.5:
            variables.epsilon *= 1.1
        elif accRate < 27.5:
            variables.epsilon *= 0.9

    #generate MCMC samples
    for i in range(nIter):
        print(f"Iteration {i+1}/{nIter}", end=" ")
        t = time.time()
        variables, dVectTemp, pVectTemp, accRate = functions.diffusionPointSampler(variables, data)
        dVect.append(variables.dIndu)
        pVect.append(variables.P)
        print(f"({time.time()-t:.3f}s)")
        if accRate > 40:
            variables.epsilon *= 2
        elif accRate < 10:
            variables.epsilon /= 2

        h5['P'][i] = variables.P
        h5['']
        
    endTime = time.time()
        
    print(str(nIter) + " iterations in " + str(endTime-startTime) + " seconds." )

    #Save Samples as h5 files and time
    startTime = time.time()
    h5f = h5py.File(str(nIter) + '(' + str(variables.covLambda) + " " + str(variables.covL) + ').h5', 'w')
    h5f.create_dataset('samples', data = dVect)
    h5f.create_dataset('prob', data = pVect)
    h5f.close()
    endTime = time.time()
    print("Time to save files: " + str(endTime-startTime))

    return()