PARAMETERS = {

    #Knowns for Ground Truth
    'grndAlpha': 10,
    'grndMeanX': 0,
    'grndMeanY': 0,
    'grndSigmaX': 5,
    'grndSigmaY': 5,
    'grndMag': 1000,
    'grndZ': None,

    #General Knowns
    'nDataX': 20,
    'nDataY': 20,
    'nInduX': 10,
    'nInduY': 10,
    'nFineX': 50,
    'nFineY': 50,
    'covLambda': 5000,
    'covL': 2,
    'minX': -5,
    'maxX': 5,
    'minY': -5,
    'maxY': 5,

    #Knowns to be evaluated
    'data': None,
    'grndCoordinates': None,
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