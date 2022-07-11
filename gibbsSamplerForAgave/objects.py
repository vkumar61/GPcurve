import numpy as np


#These are the necassary variables to run the gibbs Sampler
PARAMETERS = {

    #General Knowns
    'nInduX': 10,
    'nInduY': 10,
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
    'covLambda': 5000,
    'covL': 2,
    'fInduMean': 0,


    # Sampler parameters
    'epsilon': 1e-3,
}

#These are the neccasary variables for synthetic data generation
SYNTHETICPARAMETERS = {

    #Knowns to generate Ground Truth
    'xInitial': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'yInitial': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'min':-30,
    'max':30,
    'd0': 20,
    'dVariance': 0,
    'nTrajectories': 10,
    'lengthTrajectories': 1000,
    'deltaT': 1,

    #Ground Truth
    'dObserved': None,
    
}

#This is the object of data
DATA = {
    'trajectories': None,
    'trajectoriesIndex': None,
    'deltaT': None,
    'nData': None,
    'nTrajectories': None,
}