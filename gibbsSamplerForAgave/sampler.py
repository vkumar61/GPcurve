
import h5py
import numpy as np
from types import SimpleNamespace
#from scipy import stats
#import matplotlib.pyplot as plt
import functions
import objects

def analyze(nIter, dataVect, dataVectIndex, deltaT, covLambda, covL):

    #initialize data and variables
    data = SimpleNamespace(**objects.DATA)
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = np.unique(data.trajectoriesIndex)
    variables = functions.initialization(objects.PARAMETERS, data, covLambda, covL)

    #vectors to store diffusion samples and their probabilities
    dVect = []
    dVect.append(variables.dIndu)
    pVect = []
    pVect.append(variables.P)

    h5 = h5py.File("test", 'w')
    h5.create_dataset(name="dIndu", shape=(nIter, 2500), chunks=(1,2500), dtype=float)
    h5.close()

    for i in range(nIter):
        variables = functions.diffusionSampler(variables, data)
        # dVect.append(variables.dIndu)
        # pVect.append(variables.P)

        h5 = h5py.File("test", 'r+')
        h5["dIndu"][i, :] = variables.dIndu
        h5.close()


    #Save Samples as CSV files
    np.savetxt(str(nIter) + "samples(" + str(variables.covLambda) + " " + str(variables.covL) + ").csv", dVect, delimiter=", ", fmt="% f")
    np.savetxt(str(nIter) + "probability(" + str(variables.covLambda) + " " + str(variables.covL) + ").csv", pVect, delimiter=", ", fmt="% f")
    
    return()