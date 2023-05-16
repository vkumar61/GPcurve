#from cProfile import label
#from statistics import mode
from ctypes import pointer
import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import numba as nb

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
            C[i][j] = (covLambda**2)*(np.exp(((-1)*((dist)**2))/(covL**2)))

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
    variables.covLambda = covLambda
    variables.covL = covL
    epsilon = variables.epsilon
    deltaT = data.deltaT
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
            dataCoordinates = np.vstack((dataCoordinates, trajectories[i]))

    #Points of trajectory that are "sampled"
    sampleCoordinates = np.empty((0,2))
    for i in range(1,nData):
        if (trajectoriesIndex[i] == trajectoriesIndex[i-1]):
            sampleCoordinates = np.vstack((sampleCoordinates, trajectories[i]))

    #detrmine Covarince matrices
    cInduIndu = covMat(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = covMat(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = covMat(induCoordinates, fineCoordinates, covLambda, covL)
    cInduInduInv = np.linalg.inv(cInduIndu + epsilon*np.eye(nInduX*nInduY))
    cInduInduChol = np.linalg.cholesky(cInduIndu + np.eye(nInduX*nInduY)*epsilon)

    #Initial Guess with MLE
    diff = sampleCoordinates - dataCoordinates
    num = np.sum(diff * diff)
    den = 4*deltaT*len(diff)
    mle = num/den
    dIndu = mle * np.ones(nInduX * nInduY)
    
    #Potential MLE(for each inducing point based on the covariance kernal) to make initial guess significantly more accurate:
    diff = sampleCoordinates - dataCoordinates
    dMleData = np.sqrt(np.sum(diff*diff, axis = 1))*(1/(2*deltaT))
    print('shape of cov matrix:' + str(np.shape(cInduData.T)))
    print('shape of mleData matrix:' + str(np.shape(dMleData)))
    prediction = (dMleData @ cInduData.T)/((np.ones(np.shape(dMleData)) @ cInduData.T))
    print('shape of inducingpoints' + str(np.shape(prediction)))
    print(prediction)

    #Initial Probability
    dData = cInduData.T @ (cInduInduInv @ dIndu)
    sd = np.vstack((dData, dData)).T
    prob = np.sum(stats.norm.logpdf(sampleCoordinates, loc = dataCoordinates, scale = np.sqrt(2*sd*1)))

    dData = cInduData.T @ (cInduInduInv @ prediction)
    sd = np.vstack((dData, dData)).T
    prob1 = np.sum(stats.norm.logpdf(sampleCoordinates, loc = dataCoordinates, scale = np.sqrt(2*sd*1)))

    print("Probability of flat surface is: " + str(prob))
    print("Probability of varying surface is: " + str(prob1))

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
    variables.dInduPrior = mle

    return variables

#This function is a Metropolis sampler that samples the whole map from the posterior
def diffusionMapSampler(variables, data):

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
    dInduPrior = variables.dInduPrior
    dInduOld = variables.dIndu

    # Propose new dIndu
    dInduNew = dInduOld + np.random.randn(nIndu) @ chol * 0.1

    #Make sure sampled diffusion vallues are all positive
    if np.any(dInduNew < 0):
        return variables

    priorMean = dInduPrior*np.ones(nIndu)
    # Calculate probabilities of induced samples
    def probability(dIndu):

        # Prior
        diff = dIndu - priorMean
        prior =  (-1/2)*(diff.T @ (cInduInduInv @ diff))
        
        #grnd of data associated with fIndu
        dData = cInduData.T @ (cInduInduInv @ dIndu)
        sd = np.vstack((dData, dData)).T
        
        #Likelihood of that data
        lhood = np.sum(stats.norm.logpdf(samples, loc = means, scale = np.sqrt(2*sd*deltaT)))
        prob = lhood + prior

        return prob

    #Probability of old and new function
    pOld = probability(dInduOld)
    pNew = probability(dInduNew)
    
    #Acceptance value
    acc_prob = pNew - pOld

    if np.log(np.random.rand()) < acc_prob:
        variables.dIndu = dInduNew
        variables.P = pNew

    return variables

#This function is a Metropolis sampler that samples individual points from the posterior 
def diffusionPointSampler(variables, data):

    # Define numba version
    @nb.jit(nopython=True)
    def diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, deltaT, means, samples, data, chol, dInduPrior, dInduOld):

        # Propose new dIndu by sampling random point
        for pointIndex in range(len(dInduOld)):
            oldPoint = dInduOld[pointIndex]
            dInduNew = np.copy(dInduOld)
            newPoint = oldPoint + np.random.randn() * 0.25    

            #Make sure sampled diffusion vallues are all positive
            while (newPoint < 0):
                newPoint = oldPoint + np.random.randn() * 0.25
            
            #Set new map and prior
            dInduNew[pointIndex] = newPoint
            priorMean = dInduPrior*np.ones(nIndu)
            
            # Calculate probabilities of induced samples
            def probability(dIndu):

                # Prior
                diff = dIndu - priorMean
                prior = (-1/2)*(diff.T @ (cInduInduInv @ diff))
                
                #grnd of data associated with fIndu
                dData = cInduData.T @ (cInduInduInv @ dIndu)
                sd = np.vstack((dData, dData)).T
                
                #Likelihood of that data
                # lhood = np.sum(stats.norm.logpdf(samples, loc = means, scale = np.sqrt(2*sd*deltaT)))
                lhood = 0
                for i in range(samples.shape[0]):
                    for j in range(samples.shape[1]):
                        lhood += (
                            -.5 * (samples[i, j] - means[i, j])**2 / (2*sd[i, j]*deltaT)
                            - .5 * np.log(2*np.pi*sd[i, j]*deltaT)
                        )
                prob = lhood + prior

                return prob

            #Probability of old and new function
            pOld = probability(dInduOld)
            pNew = probability(dInduNew)
            
            #Acceptance value
            acc_prob = pNew - pOld

            if np.log(np.random.rand()) < acc_prob:
                dIndu = dInduNew
                P = pNew
        
        return dIndu, P
    
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
    dInduPrior = variables.dInduPrior
    dInduOld = variables.dIndu

    # Run numba version
    dIndu, P = diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, deltaT, means, samples, data, chol, dInduPrior, dInduOld)
    variables.dIndu = dIndu
    variables.P = P

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