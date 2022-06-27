import numpy as np
import math

def dataGenerator(generationParam, data):
    
    #Extract necassary variables
    xInitial = generationParam.xInitial
    yInitial = generationParam.yInitial
    timeVect = data.timeVect
    d0 = generationParam.d0
    dVariance = generationParam.dVariance
    nData = len(timeVect)
    
    #Sample observed diffusion coefficient across trajectory
    varMatrix = dVariance*np.eye(nData)
    meanVect = d0*np.ones(nData)
    dObserved = np.random.multivariate_normal(meanVect, varMatrix)

    #Initialize Trajectory
    trajectory = np.zeros((nData,2))
    trajectory[0] = [xInitial, yInitial]

    #Sample Trajectory
    for i in range(len(timeVect[1:])):
        mean = trajectory[i-1]
        sd = math.sqrt(2*dObserved[i-1]*(timeVect[i]-timeVect[i-1]))
        trajectory[i] = np.random.normal(mean, sd)

    #seperate x and y from trajectory
    xData = trajectory[:,0]
    yData = trajectory[:,1]

    #save all variables created
    generationParam.dObserved = dObserved
    data.dataX = xData
    data.dataY = yData
    data.nData = nData
    
    
    return generationParam, data