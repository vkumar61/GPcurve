import numpy as np

def dataReader(path):

    #read datafile
    with open(path) as inp:
        tempData = [i.strip().split('\t') for i in inp]

    #clean the data
    cleanData = []
    for i in tempData:
        if i != ['']:
            cleanData.append(i)
    
    #coordinates for trajectories
    x = np.array([float(i[3]) for i in cleanData])
    y = np.array([float(i[4]) for i in cleanData])

    #save organized dat in respective vectors
    dataVect = np.vstack((x,y)).T
    dataVectIndex = np.array([int(i[1]) for i in cleanData])
    deltaT = 1/30                                         #put manually as unavailable from data file
    return dataVect, dataVectIndex, deltaT


