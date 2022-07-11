import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt

# This function generates synthetic data
def dataGenerator(generationParam, data):
    
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
    minimum = generationParam.min
    maximum = generationParam.max
    
    #Define function that establishes form of diffusion coefficient through space
    def diffusion(mean):
        
        xterm = np.exp(-((mean[0] - 0)**2)/(2*20**2))
        yterm = np.exp(-((mean[1] - 0)**2)/(2*20**2))
        value = 10*xterm*yterm
        #value = (np.linalg.norm(mean)/10)+1
        return value

    #Initialize Trajectory
    trajectories = np.empty((0,2))
    tempTraj = np.zeros((lengthTraj,2))
    trajIndex = np.zeros(nData)
    dObserved = np.zeros(nData)
    dObserved[0] = d0

    

    #Sample Trajectory
    for h in range(nTraj):
        
        #initial position
        tempTraj[0] = [xInitial[h],yInitial[h]]
        trajIndex[h*lengthTraj] = h+1
        

        #loop through full length of each trajectory
        for i in range(1,lengthTraj):

            #Sample diffusion
            mean = tempTraj[i-1]
            dPoint = diffusion(mean)
            sd = np.sqrt(2*dPoint*(deltaT))
            
            tempTraj[i] = np.random.normal(mean, sd)

            while (np.any(tempTraj[i] < minimum) or np.any(tempTraj[i] > maximum)):
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

'''This is a function that returns a covariance matrix between two position vectors
    based on square exponential kernal with parameters covLambda and covL'''    
def covMat(coordinates1, coordinates2, covLambda, covL):
    #Create empty matrix for covariance
    C = np.zeros((len(coordinates1), len(coordinates2)))
    
    #loop over all indecies in covariance matrix
    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            #Calculate distance between points
            dist = np.sqrt((coordinates1[i,0] - coordinates2[j,0])**2 + (coordinates1[i,1] - coordinates2[j,1])**2)
            #Determine each element of covariance matrix
            C[i][j] = (covLambda**2)*(np.exp(((-1)*((dist)**2))/(covL**2)))

    #Return Covariance Matrix
    return C

#initialize sampler parameters
def initialization(variables, data):

    #declare variables as object of 
    variables = SimpleNamespace(**variables)

    trajectories = data.trajectories
    nData = data.nData
    nTraj = data.nTrajectories
    trajectoriesIndex = data.trajectoriesIndex
    nInduX = variables.nInduX
    nInduY = variables.nInduY
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    covLambda = variables.covLambda
    covL = variables.covL
    epsilon = variables.epsilon
    
    dataX = trajectories[:,0]
    dataY = trajectories[:,1]
    minX = min(dataX)
    #minX = -20
    minY = min(dataY)
    #minY = -20
    maxX = max(dataX)
    #maxX = 20
    maxY = max(dataY)
    #maxY = 20

   #define coordinates for Inducing points
    x = np.linspace(minX, maxX, nInduX)
    y = np.linspace(minY, maxY, nInduY)
    xTemp, yTemp = np.meshgrid(x, y)
    X = np.reshape(xTemp, -1)
    Y = np.reshape(yTemp, -1)
    induCoordinates = np.vstack((X, Y)).T
    
    #define coordinates for Fine points
    x = np.linspace(minX, maxX, nFineX)
    y = np.linspace(minY, maxY, nFineY)
    xTemp, yTemp = np.meshgrid(x, y)
    X = np.reshape(xTemp, -1)
    Y = np.reshape(yTemp, -1)
    fineCoordinates = np.vstack((X, Y)).T

    #Points of trajectory where learning is possible
    dataCoordinates = np.empty((0,2))
    for i in range((nData-1)):
        if (trajectoriesIndex[i] == trajectoriesIndex[i+1]):
            dataCoordinates = np.vstack((dataCoordinates,trajectories[i]))

    #Points of trajectory that are "sampled"
    sampleCoordinates = np.empty((0,2))
    for i in range(1,nData):
        if (trajectoriesIndex[i] == trajectoriesIndex[i-1]):
            sampleCoordinates = np.vstack((sampleCoordinates,trajectories[i]))

    #detrmine Covarince matrices
    cInduIndu = covMat(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = covMat(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = covMat(induCoordinates, fineCoordinates, covLambda, covL)
    cInduInduInv = np.linalg.inv(cInduIndu + epsilon*np.eye(nInduX*nInduY))
    cInduInduChol = np.linalg.cholesky(cInduIndu + np.eye(nInduX*nInduY)*epsilon)
    
    #Initial Guess
    dIndu = 20 * np.ones(nInduX * nInduY)


    #Initial Probability
    dData = cInduData.T @ cInduInduInv @ dIndu
    sd = np.vstack((dData, dData)).T
    prob = np.sum(stats.norm.logpdf(sampleCoordinates, loc = dataCoordinates, scale = np.sqrt(2*sd*1)))

    #save all variable parameters
    variables.sampleCoordinates = sampleCoordinates
    variables.dataCoordinates = dataCoordinates
    variables.induCoordinates = induCoordinates
    variables.fineCoordinates = fineCoordinates
    variables.cInduIndu = cInduIndu
    variables.cInduData = cInduData
    variables.cInduFine = cInduFine
    variables.cInduInduChol = cInduInduChol
    variables.cInduInduInv = cInduInduInv
    variables.dIndu = dIndu
    variables.P = prob

    return variables

def diffusionSampler(variables, data):

    #necassary variables
    nIndu = variables.nInduX*variables.nInduY
    cInduIndu = variables.cInduIndu
    cInduData = variables.cInduData
    cInduInduInv = variables.cInduInduInv
    means = variables.dataCoordinates
    samples = variables.sampleCoordinates
    data = data.trajectories
    chol = variables.cInduInduChol

    # Propose new dIndu
    dInduOld = variables.dIndu
    dInduNew = dInduOld + np.random.randn(nIndu) @ chol * 0.0001

    #Make sure sampled diffusion vallues are all positive
    #while np.any(dInduNew<0):
    #    dInduNew = dInduOld + np.random.randn(nIndu) @ chol * 0.01

    # Calculate probabilities of induced samples
    def probability(dIndu):

        # Prior
        prior = stats.multivariate_normal.logpdf(dIndu.T, mean = np.zeros(nIndu), cov=cInduIndu)
        
        #grnd of data associated with fIndu
        dData = cInduData.T @ cInduInduInv @ dIndu
        sd = np.vstack((dData, dData)).T
        
        #Likelihood of that data
        lhood = np.sum(stats.norm.logpdf(samples, loc = means, scale = np.sqrt(2*sd*1)))
        prob = lhood + prior

        return lhood, prob

    #Probability of old and new function
    lhoodOld, pOld = probability(dInduOld)
    lhoodNew, pNew = probability(dInduNew)
    
    #Acceptance value
    acc_prob = pNew - pOld

    #Accept or Reject
    if np.log(np.random.rand()) < acc_prob:
        variables.dIndu = dInduNew
        variables.P = lhoodNew

    return variables


def plots (dVect, pVect, variables):


    mapCoordinates = variables.dataCoordinates
    cInduInduInv = variables.cInduInduInv
    cInduData = variables.cInduData

    mapIndex = pVect.index(max(pVect))
    
    #Inducing Points
    initialIndu = dVect[0]
    mapIndu = dVect[mapIndex]
    
    #Transformed to data Points
    learnedMap = cInduData.T @ cInduInduInv @ mapIndu
    initialMap = cInduData.T @ cInduInduInv @ initialIndu

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    initialPlot = ax.scatter(mapCoordinates[:,1], mapCoordinates[:,1], initialMap)
    mapPlot = ax.scatter(mapCoordinates[:,1], mapCoordinates[:,1], learnedMap)

    return fig