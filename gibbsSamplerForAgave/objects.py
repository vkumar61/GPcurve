#These are the necassary variables to run the gibbs Sampler
PARAMETERS = {

    #General Knowns
    'nInduX': 10,
    'nInduY': 10,
    'nFineX': 50,
    'nFineY': 50,
    'covLambda': 5000,
    'covL': 2,

    #Knowns to be evaluated
    'induCoordinates': None,
    'fineCoordinates': None,
    'cInduIndu': None,
    'cInduData': None,
    'cInduFine': None,
    'cInduInduChol': None,
    'cInduInduInv': None,

    # Variables
    'P': float,
    'alpha': float,
    'fIndu': None,

    # Priors
    'ell': 1,
    'sig': 10,
    'fInduMean': 0,
    'alphaShape': 1,
    'alphaScale': 1000,


    # Sampler parameters
    'epsilon': 1e-3,
}

#These are the neccasary variables for synthetic data generation
SYNTHETICPARAMETERS = {

    #Knowns to generate Ground Truth
    'xInitial': 5,
    'yInitial': 6,
    'd0': 1,
    'dVariance': 2,

    #Ground Truth
    'dObserved': None,
}



#This is the object of data
DATA = {
    'dataX': None,
    'dataY': None,
    'timeVect': None,
    'nData': None,
}