import numpy as np

def dataReader(path, scale=1, deltaT = 1/30):
    # Read the CSV file, considering the header
    data = np.genfromtxt(path, delimiter=', ', skip_header=1)

    # Separate columns into individual arrays
    dataVectIndex = data[:, 0]
    dataVect = data[:, 1:]
    
    #make pixel adjustment to nanometers
    dataVect = dataVect[::]*scale
    dataVectIndex = dataVectIndex[::]
    
    #put time step manually as unavailable from data file
    deltaT = deltaT

    return dataVect, dataVectIndex, deltaT