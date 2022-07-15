import numpy as np


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
    'covL': 10,
    'fInduMean': 0,


    # Sampler parameters
    'epsilon': 1e-3,
}

#These are the neccasary variables for synthetic data generation
SYNTHETICPARAMETERS = {

    #Knowns to generate Ground Truth
    'xInitial': 0,
    'yInitial': 0,
    #'min':-30,
    #'max':30,
    'd0': 20,
    'dVariance': 0,
    'nTrajectories': 100,
    'lengthTrajectories': 10,
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