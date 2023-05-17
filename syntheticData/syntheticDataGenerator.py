import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt


#These are the neccasary variables for synthetic data generation
SYNTHETICPARAMETERS = {

    #Knowns to generate Ground Truth
    'xInitial': 0,
    'yInitial': 0,
    'd0': 20,
    'dVariance': 0,
    'nTrajectories': 15000,
    'lengthTrajectories': 10,
    'deltaT': 1/30,

    #Ground Truth
    'dObserved': None,

}

#Define function that establishes form of diffusion coefficient through space
def diffusion(point):

    x = point[0]
    y = point[1]
    xterm = np.sin(x/30)
    yterm = np.sin(y/30)
    value = 2*(2+xterm+yterm)
    
    return value

# This function generates synthetic data
def dataGenerator(data, generationParam):

    #initialize
    data = SimpleNamespace(**data)
    generationParam = SimpleNamespace(**generationParam)

    #Extract necassary variables
    xInitial = generationParam.xInitial
    yInitial = generationParam.yInitial
    deltaT = generationParam.deltaT
    d0 = generationParam.d0
    dVariance = generationParam.dVariance
    nTraj = generationParam.nTrajectories
    lengthTraj = generationParam.lengthTrajectories
    nData = nTraj*lengthTraj

    #Initialize Trajectory
    trajectories = np.empty((0,2))
    tempTraj = np.zeros((lengthTraj,2))
    trajIndex = np.zeros(nData)
    dObserved = np.zeros(nData)
    dObserved[0] = d0

    #Sample Trajectory
    for h in range(nTraj):

        #initial position
        tempTraj[0] = [np.random.randint(1,250), np.random.randint(1,250)]
        trajIndex[h*lengthTraj] = h+1


        #loop through full length of each trajectory
        for i in range(1,lengthTraj):

            #Sample diffusion
            mean = tempTraj[i-1]
            dPoint = diffusion(mean)
            sd = np.sqrt(2*dPoint*(deltaT))

            tempTraj[i] = np.random.normal(mean, sd)

            #save index of trajectory and the observed diffusion at that point
            trajIndex[h*lengthTraj+i] = h+1
            dObserved[h*lengthTraj+i] = dPoint

        trajectories = np.concatenate((trajectories,tempTraj))


    #save all variables created
    generationParam.dObserved = dObserved
    data.trajectories = trajectories
    data.nData = nData
    data.deltaT = deltaT
    data.trajectoriesIndex = trajIndex
    data.nTrajectories = generationParam.nTrajectories

    
    return generationParam, data

#This function generates a plot of the MAP as a contour plot
def plots(variables, generationParam, dVect, pVect):

    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates

    groundTruth = diffusion(fineCoordinates.T)
    shape = (nFineX, nFineY)

    unshapedMap = cInduFine.T @ cInduInduInv @ dVect[pVect.index(max(pVect))]

    shapedGroundTruth = np.reshape(groundTruth, shape)
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    fig = plt.figure()

    mapPlot = plt.contour(shapedX, shapedY, shapedMap, colors = 'r', levels = [10, 20, 30, 40, 50])
    truthPlot = plt.contour(shapedX, shapedY, shapedGroundTruth, colors = 'c', levels = [10, 20, 30, 40, 50])
    plt.clabel(mapPlot, inline=1, fontsize=10)
    plt.clabel(truthPlot, inline=1, fontsize=10)

    return fig