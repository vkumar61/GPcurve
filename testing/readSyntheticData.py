import numpy as np

def dataReader(path, num):
    
    #to reduce runtime and memeory needs we only take nth element from the data
    n = num
    
    #read datafile

    # Read the CSV file, considering the mentioned points
    data = np.genfromtxt(path, delimiter=', ', skip_header=1)

    # Separate columns into individual arrays
    dataVectIndex = data[:, 0]
    dataVect = data[:, 1:]
    
    #sub sample along with pixel size adjustment
    dataVect = dataVect[::n]*1
    dataVectIndex = dataVectIndex[::n]
    
    #put time step manually as unavailable from data file
    deltaT = (1/30)*n

    #print('We only used every ' + str(n) + 'th datapoint')
    return dataVect, dataVectIndex, deltaT