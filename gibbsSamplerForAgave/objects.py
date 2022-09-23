#These are the necassary variables to run the Gibbs Sampler
PARAMETERS = {

    #General Knowns
    'nInduX': 11,   #number of x inducing points
    'nInduY': 11,   #number of y grid points (total inducing points is nInduY*nInduX)
    'nFineX': 50,   #fine grid points in x
    'nFineY': 50,   #fine grid points in y (total fine points is nFineY*nFineX)

    #Knowns to be evaluated
    'dataCoordinates': None,    #all points of each trajectory exepts final location
    'sampleCoordinates': None,  #all points of each trajectory exepts initial location
    'induCoordinates': None,    #coordinates of the inducing points
    'fineCoordinates': None,    #coordinates of fine grid points
    'cInduIndu': None,          #covariance matrix between inducing points
    'cInduData': None,          #covariance matrix between inducing points and data points
    'cInduFine': None,          #covariance matrix between inducing points and fine grid points
    'cInduInduChol': None,      #cholesky decomposition of cInduIndu
    'cInduInduInv': None,       #inverse of cInduIndu

    # Variables
    'P': float,     #probability of each sample
    'dIndu': None,  #diffusion coefficient sample map at inducing points

    # Priors
    'covLambda': 1,     #coefficient of covariance square exponential kernal (1 only used if hyper parameters on specified)
    'covL': 20,         #lenghts parameter of covariance square exponential kernal (20 only used if hyper parameters on specified)
    'dInduPrior': None, #Prior on Inducing point MAP (set to MLE in init)


    # Sampler parameters
    'epsilon': 1e-3,    #perturbation parameter to keep matrix decomp numerically stable
}

#This is the object of data
DATA = {
    #NEED TO BE LOADED IN
    'trajectories': None,       #coordinates of the data
    'trajectoriesIndex': None,  #index of trajectory number
    'deltaT': None,             #time passed in between each frame of data

    #INITIALIZED INDEPENDENTLY BASED ON PARAMATERS ABOVE
    'nData': None,              #number of datapoints (set to len(nData))
    'nTrajectories': None,      #number of trajectories (set to np.unique(trajectoriesIndex))
}