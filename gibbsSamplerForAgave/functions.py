import numpy as np
from types import SimpleNamespace
from scipy import stats

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
    
    #Sample observed diffusion coefficient across trajectory
    varMatrix = dVariance*np.eye(nData)
    meanVect = d0*np.ones(nData)
    dObserved = abs(np.random.multivariate_normal(meanVect, varMatrix))

    #Initialize Trajectory
    trajectories = np.empty((0,2))
    tempTraj = np.zeros((lengthTraj,2))
    trajIndex = np.zeros(nData)

    #Sample Trajectory
    for h in range(nTraj):
        
        #initial position
        tempTraj[0] = [xInitial[h],yInitial[h]]
        trajIndex[h*lengthTraj] = h+1

        #diffusion
        for i in range(1,lengthTraj):
            mean = tempTraj[i-1]
            sd = np.sqrt(2*dObserved[i-1]*(deltaT))
            tempTraj[i] = np.random.normal(mean, sd)
            trajIndex[h*lengthTraj+i] = h+1

        trajectories = np.concatenate((trajectories,tempTraj))


    #save all variables created
    generationParam.dObserved = dObserved
    data.trajectories = trajectories
    data.nData = nData
    data.deltaT = deltaT
    data.trajectoriesIndex = trajIndex
    
    
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
    minY = min(dataY)
    maxX = max(dataX)
    maxY = max(dataY)

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
    cInduInduChol = np.linalg.cholesky(cInduIndu+np.eye(nInduX*nInduY)*epsilon)
    
    #Initial Guess
    dIndu = 10 * np.ones(nInduX * nInduY)


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
    dInduNew = dInduOld + np.random.randn(nIndu) @ chol * 0.1

    #Make sure sampled diffusion vallues are all positive
    while np.any(dInduNew<0):
        dInduNew = dInduOld + np.random.randn(nIndu) @ chol * 0.1

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

        return prob

    #Probability of old and new function
    pOld = probability(dInduOld)
    pNew = probability(dInduNew)
    
    #Acceptance value
    acc_prob = pNew - pOld

    if np.log(np.random.rand()) < acc_prob:
        variables.dIndu = dInduNew

    return variables