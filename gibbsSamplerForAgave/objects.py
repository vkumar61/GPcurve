#These are the necassary variables to run the gibbs Sampler
PARAMETERS = {

    #General Knowns
    'nInduX': 11,
    'nInduY': 11,
    'nFineX': 50,
    'nFineY': 50,

    #Knowns to be evaluated
    'dataCoordinates': None,
    'sampleCoordinates': None,
    'induCoordinates': None,
    'fineCoordinates': None,
    'cInduIndu': None,
    'cInduData': None,
    'cInduFine': None,
    'cInduInduChol': None,
    'cInduInduInv': None,

    # Variables
    'P': float,
    'dIndu': None,

    # Priors
    'covLambda': 1,
    'covL': 20,
    'dInduPrior': None,


    # Sampler parameters
    'epsilon': 1e-3,
}

#This is the object of data
DATA = {
    'trajectories': None,
    'trajectoriesIndex': None,
    'deltaT': None,
    'nData': None,
    'nTrajectories': None,
}