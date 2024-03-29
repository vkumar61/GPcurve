#from cProfile import label
#from statistics import mode
from ctypes import pointer
import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import numba as nb
from math import lgamma
from scipy.spatial import KDTree

def find_closest_point_indices(target_point, points, k=20):
    kdtree = KDTree(points)
    distances, indices = kdtree.query(target_point, k=k)
    return indices

@nb.njit(cache=True)
def loggammapdf(x, shape, scale):
    return - shape*np.log(scale) - lgamma(shape) + (shape-1)*np.log(x) - x/scale

@nb.njit(cache=True)
def logNormpdf(diff, sigma):
    return -np.log(np.abs(sigma))-0.5*(diff/sigma)**2

#create a covariance matrix based on data at hand
@nb.njit(cache=True)
def covMat(coordinates1, coordinates2, covLambda, covL):

    #Create empty matrix for covariance
    C = np.zeros((len(coordinates1), len(coordinates2)))
    
    #loop over all indices in covariance matrix
    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            #Calculate distance between points
            dist = np.sqrt((coordinates1[i,0] - coordinates2[j,0])**2 + (coordinates1[i,1] - coordinates2[j,1])**2)
            #Determine each element of covariance matrix
            C[i, j] = (covLambda**2)*(np.exp(((-1)*((dist)**2))/(covL**2)))

    #Return Covariance Matrix
    return C

#initialize sampler parameters
def initialization(variables, data, covLambda, covL):

    #declare variables as instance of SimpleNamespace
    variables = SimpleNamespace(**variables)

    #pull necassary variables
    trajectories = data.trajectories
    nData = data.nData
    nTraj = data.nTrajectories
    trajectoriesIndex = data.trajectoriesIndex
    nInduX = variables.nInduX
    nInduY = variables.nInduY
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    epsilon = variables.epsilon
    deltaT = data.deltaT
    dataX = trajectories[:,0]
    dataY = trajectories[:,1]
    minX = min(dataX)
    minY = min(dataY)
    maxX = max(dataX)
    maxY = max(dataY)
    nIndu = nInduX*nInduY
    covL = variables.covL

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
    for i in range(nData-1):
        if (trajectoriesIndex[i] == trajectoriesIndex[i+1]):
            dataCoordinates = np.vstack((dataCoordinates, trajectories[i]))

    #Points of trajectory that are "sampled"
    sampleCoordinates = np.empty((0,2))
    for i in range(1,nData):
        if (trajectoriesIndex[i] == trajectoriesIndex[i-1]):
            sampleCoordinates = np.vstack((sampleCoordinates, trajectories[i]))

    #Initial Guess with MLE
    diff = sampleCoordinates - dataCoordinates
    num = np.sum(diff * diff)
    den = 4*deltaT*len(diff)
    mle = num/den
    dIndu = mle * np.ones(nInduX * nInduY)
    
    #Potential MLE(for each inducing point based on the covariance kernal) to make initial guess significantly more accurate:
    diff = sampleCoordinates - dataCoordinates
    dMleData = np.sum(diff*diff, axis = 1)/(4*deltaT)

    #Establish Hyperparameters more accurately if not chosen by user
    if covL == None:
        covL = np.max([maxX-minX, maxY-minY]) * 0.1
    if covLambda == None:
        covLambda = .5 * mle

    #detrmine Covarince matrices
    cInduIndu = covMat(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = covMat(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = covMat(induCoordinates, fineCoordinates, covLambda, covL)
    cInduInduInv = np.linalg.pinv(cInduIndu)
    cDataIndu = cInduData.T @ cInduInduInv
    cInduInduChol = np.linalg.cholesky(cInduIndu + epsilon*np.eye(nInduX*nInduY))
    print('shape of cov matrix:' + str(np.shape(cInduData.T)))
    print('shape of mleData matrix:' + str(np.shape(dMleData)))

    #loop through induncing points and make them the value of the MLE of the k nearest datapoints
    # k = np.floor(len(dMleData)/nIndu)
    # for i in range(nIndu):
    #     closest = find_closest_point_indices(induCoordinates[i], dataCoordinates, k=k)
    #     dIndu[i] = np.mean(dMleData[closest])

    # Set up dData
    dData = cDataIndu @ dIndu

    #Initial Probability
    P = -np.inf

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
    variables.P = P
    variables.mle = mle
    variables.priorMean = dIndu
    variables.cDataIndu = cDataIndu
    variables.dData = dData
    variables.covLambda = covLambda
    variables.covL = covL

    return variables

#This function is a Metropolis sampler that samples the whole map from the posterior
def diffusionMapSampler(variables, data):

    # Extract variables
    nIndu = variables.nInduX*variables.nInduY
    cInduIndu = variables.cInduIndu
    cInduData = variables.cInduData
    cInduInduInv = variables.cInduInduInv
    deltaT = data.deltaT
    means = variables.dataCoordinates
    samples = variables.sampleCoordinates
    data = data.trajectories
    chol = variables.cInduInduChol
    dInduOld = variables.dIndu
    cDataIndu = variables.cDataIndu
    P = variables.P
    mle = variables.mle
    priorMean = variables.priorMean
    epsilon = variables.epsilon

    # Set constants
    

    # Define probability of inducing points
    def probability(dIndu_, dData_):

        # Prior
        diff = dIndu_ - priorMean
        prior =  (
            -0.5*(diff.T @ (cInduInduInv @ diff))
            # -0.5*np.log(np.linalg.slogdet(cInduIndu+np.eye(nIndu)*epsilon)[1])
        )
        
        #Likelihood of that data
        lhood = np.sum(
            stats.norm.logpdf(
                samples,
                loc=means,
                scale=np.sqrt(2*np.vstack((dData_, dData_)).T*deltaT)
            )
        )
        prob = lhood + prior

        return prob

    # Propose new dIndu
    dInduNew = dInduOld + chol @ np.random.randn(nIndu) * 0.001
    dDataNew = cDataIndu @ dInduNew

    # Make sure sampled diffusion values are all positivelhood
    if (np.any(dDataNew < 0) or np.any(dInduNew < 0)):
        return variables

    # Probability of old and new function
    pOld = P
    pNew = probability(dInduNew, dDataNew)
    #print(pNew)
    
    # Acceptance value
    acc_prob = pNew - pOld
    if acc_prob > np.log(np.random.rand()):
        variables.dIndu = dInduNew
        variables.dData = dDataNew
        variables.P = pNew

    return variables

#This function is a Metropolis sampler that samples individual points from the posterior 
def diffusionPointSampler(variables, data):
    
    # Define numba version
    @nb.jit(nopython=True, cache = True)
    def diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, pOld, dDataOld, priorMean):

        # Calculate probabilities of induced samples
        def probability(dIndu_, dData_):
            
            # Prior
            diff = dIndu_ - priorMean
            prior = -0.5*(diff.T @ (cInduInduInv @ diff))
            
            #Likelihood of that data
            lhood = 0
            for i in range(samples.shape[0]):
                for j in range(samples.shape[1]):
                    lhood += (
                        -.5 * (samples[i, j] - means[i, j])**2 / (2*dData_[i]*deltaT)
                        -.5 * np.log(2*np.pi*2*dData_[i]*deltaT)
                    )
            prob = lhood + prior

            return prob
        
        #initialize inverse space
        alphaVect = cInduInduInv @ dInduOld

        #Counter for acceptances
        accCounter = 0

        #Propose new dIndu by sampling random points in inverse space
        for pointIndex in range(len(alphaVect)):

            #Propose new alpha point
            oldAlphaPoint = alphaVect[pointIndex]
            alphaDiff = 0.01 * oldAlphaPoint * np.random.randn()
            newAlphaPoint = oldAlphaPoint + alphaDiff
            
            #Incorporate new point into dInduNew and dDataNew
            dInduNew = dInduOld + cInduIndu[:, pointIndex] * alphaDiff
            dDataNew = dDataOld + cInduData[pointIndex, :] * alphaDiff

            # Make sure sampled diffusion vallues are all positive
            if (np.all(dDataNew > 0) and np.all(dInduNew > 0)):
                    
                #Probability of old and new function
                pOld = pOld
                pNew = probability(dInduNew, dDataNew)

                #Compute acceptance ratio
                acc_prob = (
                    pNew - pOld
                    +logNormpdf(diff = alphaDiff, sigma = 0.01 * newAlphaPoint)
                    -logNormpdf(diff = alphaDiff, sigma = 0.01 * oldAlphaPoint)
                    )
                
                #Accept or Reject
                if acc_prob > np.log(np.random.rand()):
                    accCounter += 1
                    dInduOld = dInduNew
                    dDataOld = dDataNew
                    pOld = pNew

        # #Propose new dIndu by sampling random point
        # for pointIndex in range(len(dInduOld)):

        #     #Propose new dIndu point
        #     oldPoint = dInduOld[pointIndex]
        #     newPoint = np.random.gamma(propshape, scale=oldPoint/propshape)
            
        #     #Incorporate new point into dIndu
        #     dInduNew = np.copy(dInduOld)
        #     dInduNew[pointIndex] = newPoint
        #     dDataNew = dData + cDataIndu[:,pointIndex] * (newPoint - oldPoint)
            
        #     #Probability of old and new function
        #     pOld = P
        #     pNew = probability(dInduNew, dDataNew)

        #     #Accept or reject
        #     acc_prob = (
        #         pNew - pOld
        #         + loggammapdf(oldPoint, propshape, scale=newPoint/propshape)
        #         - loggammapdf(newPoint, propshape, scale=oldPoint/propshape)
        #     )
        #     if acc_prob > np.log(np.random.rand()):
        #         dInduOld = dInduNew
        #         dData = dDataNew
        #         P = pNew      

        print(f'{accCounter} is the # of accepted samples') 
        return dInduOld, pOld, dDataOld
    
    #necassary variables
    nIndu = variables.nInduX*variables.nInduY
    cInduIndu = variables.cInduIndu
    cInduData = variables.cInduData
    cInduInduInv = variables.cInduInduInv
    deltaT = data.deltaT
    means = variables.dataCoordinates
    samples = variables.sampleCoordinates
    data = data.trajectories
    chol = variables.cInduInduChol
    dInduOld = variables.dIndu
    cDataIndu = variables.cDataIndu
    P = variables.P
    dData = variables.dData
    priorMean = variables.priorMean

    # Run numba version
    dIndu, P , dData = diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, P, dData, priorMean)
    variables.dIndu = dIndu
    variables.P = P
    variables.dData = dData

    return variables

#This function generates a plot of the MAP as a contour plot
def plots(variables, dVect, pVect, data):

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #sample with maximum probability
    unshapedMap = cInduFine.T @ (cInduInduInv @ dVect[pVect.index(max(pVect))])
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.figure()
    mapPlot = plt.contour(shapedX, shapedY, shapedMap, levels = 25)
    plt.clabel(mapPlot, inline=1, fontsize=10)
    plt.scatter(trajectories[:,0], trajectories[:,1], alpha = 0.01)

    return fig

#This function plots the probability
def probPlot(pVect):

    #generate plot
    fig = plt.figure()
    plt.plot(pVect)
    plt.title("Probability per Sample")
    plt.xlabel("Iteration/Sample")
    plt.ylabel("Log Probability")

    return fig

#This function plots the mean of all dVect samples
def meanPlot(variables, dVect, data):

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #take mean of all samples
    unshapedMap = cInduFine.T @ (cInduInduInv @ np.mean(dVect, 0))
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.figure()
    mapPlot = plt.contour(shapedX, shapedY, shapedMap, levels = 25, cmap = cm.autumn)
    plt.clabel(mapPlot, inline=1, fontsize=10)
    plt.scatter(trajectories[:,0], trajectories[:,1], alpha = 0.01, c = "black")

    return fig

def plotThreeD(variables, dVect, data):

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #take mean of all samples
    unshapedMap = cInduFine.T @ (cInduInduInv @ np.mean(dVect, 0))
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.axes(projection='3d')
    fig.plot_surface(shapedX, shapedY, shapedMap, cmap=cm.coolwarm)
    fig.scatter3D(trajectories[:,0], trajectories[:,1], 0, color = "green", alpha = 0.1, label = "Particle Data")
    #plt.clabel(mapPlot, inline=1, fontsize=10)
    #plt.scatter(trajectories[:,0], trajectories[:,1], alpha = 0.01)
    
    #black_proxy = plt.Circle((0, 0), 1, fc="k", alpha = 0.5)
    fig.set_xlabel(r"X ($\mu m$)")
    fig.set_ylabel(r"Y ($\mu m$)")
    fig.set_zlabel(r"Diff. Coefficient ($\mu m$)")
    fig.set_title('Learned Diffusion Map')

    #fig.legend([black_proxy], ["Particle Data"])
    fig.legend()
    return fig