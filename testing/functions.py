import numpy as np
from types import SimpleNamespace
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import numba as nb

#setseed
np.random.seed(42)

#function that calculates logpdf of Normal distribution asumming n=1 and assuming normalization gets subtrcted
@nb.njit(cache=True)
def logNormpdf(diff, sigma):
    return -np.log(np.abs(sigma))-0.5*(diff/sigma)**2

#function that randomizes index sequence for point sampler
@nb.njit(cache=True)
def indexShuffler(length):
    indices = np.arange(length)
    for i in range(length):
        j = int(np.random.random() * (length - i)) + i
        indices[i], indices[j] = indices[j], indices[i]
    return indices

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
            C[i, j] = (covLambda**2)*(np.exp(((-0.5)*((dist)**2))/(covL**2)))

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
    covLambda = variables.covLambda

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

    #Estimate Hyperparameters if not chosen by user
    if covL == None:
        covL = np.max([maxX-minX, maxY-minY]) * 0.1
    if covLambda == None:
        covLambda = mle/10

    #define coordinates for Inducing points
    xIndu = np.linspace(minX-covL, maxX+covL, nInduX)
    yIndu = np.linspace(minY-covL, maxY+covL, nInduY)
    xTemp, yTemp = np.meshgrid(xIndu, yIndu)
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
    
    #find the inducing points that are on the outside of grid far from data
    remove1 = []
    for i in xIndu:
        for j in induCoordinates[np.where(induCoordinates[:,0] == i)]:
            if(np.min(np.linalg.norm(dataCoordinates-j, axis=1)) >= 0.67*covL):
                remove1.append(j)
            else:
                break
    for i in yIndu:
        for j in induCoordinates[np.where(induCoordinates[:,1] == i)]:
            if(np.min(np.linalg.norm(dataCoordinates-j, axis=1)) >= 0.67*covL):
                remove1.append(j)
            else:
                break
    for i in xIndu:
        for j in reversed(induCoordinates[np.where(induCoordinates[:,0] == i)]):
            if(np.min(np.linalg.norm(dataCoordinates-j, axis=1)) >= 0.67*covL):
                remove1.append(j)
            else:
                break
    for i in yIndu:
        for j in reversed(induCoordinates[np.where(induCoordinates[:,1] == i)]):
            if(np.min(np.linalg.norm(dataCoordinates-j, axis=1)) >= 0.67*covL):
                remove1.append(j)
            else:
                break

    #remove the inducing points not near data
    indicies = []
    for i in remove1:    
        indicies.append(np.where(np.all(induCoordinates==i,axis=1))[0][0])
    induCoordinates = np.delete(induCoordinates, indicies, axis = 0)

    #set up initial sample and mean of prior
    dIndu = mle * np.ones(len(induCoordinates))
    priorMean = dIndu.copy()

    #determine Covarince matrices
    cInduIndu = covMat(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = covMat(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = covMat(induCoordinates, fineCoordinates, covLambda, covL)
    cInduInduInv = np.linalg.inv(cInduIndu + epsilon*np.mean(cInduIndu)*np.eye(len(dIndu)))
    cDataIndu = cInduData.T @ cInduInduInv
    cInduInduChol = np.linalg.cholesky(cInduIndu + epsilon*np.mean(cInduIndu)*np.eye(len(dIndu)))

    dData = cDataIndu @ dIndu
    
    if np.any(dData < 0):
        print("Increase the length scale, the # of inducing points, "+
              "set the initial Sample to a flat plane")
        exit()
        
    #Likelihood of that data
    lhood = np.sum(
        stats.norm.logpdf(
            sampleCoordinates,
            loc=dataCoordinates,
            scale=np.sqrt(2*np.vstack((dData, dData)).T*deltaT)
        )
    )

    #Prior of the Data
    diff = dIndu - priorMean
    prior =  (
            -0.5*(diff.T @ (cInduInduInv @ diff))
        )
    P = lhood + prior
    print(f"The initial probability is {P}")

    #Initial Probability
    #P = -np.inf

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
    variables.priorMean = priorMean
    variables.cDataIndu = cDataIndu
    variables.dData = dData
    variables.covLambda = covLambda
    variables.covL = covL

    return variables

#This function is a Metropolis sampler that samples the whole map from the posterior
def diffusionMapSampler(variables, data):

    # Extract variables
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
    nIndu = len(dInduOld) 

    # Define probability of inducing points
    def probability(dIndu_, dData_):

        # Prior ignoring normalization
        diff = dIndu_ - priorMean
        prior =  (
            -0.5*(diff.T @ (cInduInduInv @ diff))
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
    a = np.random.exponential(epsilon/10)
    dInduNew = dInduOld + chol @ np.random.randn(nIndu) * a
    dDataNew = cDataIndu @ dInduNew

    # Make sure sampled diffusion values are all positive
    if (np.any(dDataNew < 0) or np.any(dInduNew < 0)):
        return variables

    # Probability of old and new function
    pOld = P
    pNew = probability(dInduNew, dDataNew)
    
    # Acceptance value
    acc_prob = pNew - pOld
    if acc_prob > np.log(np.random.rand()):
        variables.dIndu = dInduNew
        variables.dData = dDataNew
        variables.P = pNew

    return variables

@nb.jit(nopython=True, cache = True)
def diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, pOld, dDataOld, priorMean, covLambda, epsilon):

    pVectTemp = np.zeros((nIndu))
    dVectTemp = np.zeros((nIndu, nIndu))

    # Calculate probabilities of induced samples
    def probability(dIndu_, dData_):
        
        # Prior ignoring normalization
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

    #Counter for acceptances and iterations
    accCounter = 0
    iter = 0
    #shuffle the index to avoid local minima from order of iteration
    shuffledIndex = indexShuffler(len(alphaVect))

    #Propose new dIndu by sampling random points in inverse space
    for pointIndex in shuffledIndex:

        #Propose new alpha point
        oldAlphaPoint = alphaVect[pointIndex]
        a = np.random.exponential(epsilon)
        alphaDiff = a * oldAlphaPoint * np.random.randn()
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
                +logNormpdf(diff = alphaDiff, sigma = a * newAlphaPoint)
                -logNormpdf(diff = alphaDiff, sigma = a * oldAlphaPoint)
                )
            
            #Accept or Reject
            if acc_prob > np.log(np.random.rand()):
                accCounter += 1
                dInduOld = dInduNew
                dVectTemp[iter] = dInduNew
                pVectTemp[iter] = pNew
                dDataOld = dDataNew
                pOld = pNew
            else:
                dVectTemp[iter] = dInduOld
                pVectTemp[iter] = pOld
        else:
            dVectTemp[iter] = dInduOld
            pVectTemp[iter] = pOld
        iter += 1    
    print(f'{accCounter} is the # of accepted samples') 

    return dInduOld, pOld, dDataOld, dVectTemp, pVectTemp

#This function is a Metropolis sampler that samples individual points from the posterior 
def diffusionPointSampler(variables, data):
    
    #necassary variables
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
    covLambda = variables.covLambda
    epsilon = variables.epsilon
    nIndu = len(dInduOld)
    
    # Run numba version
    dIndu, P , dData, dVect, pVect = diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, P, dData, priorMean, covLambda, epsilon)
    variables.dIndu = dIndu
    variables.P = P
    variables.dData = dData

    return variables, dVect, pVect

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