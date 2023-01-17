import numpy as np

def dataReader(path, num):
    
    #to reduce runtime and memeory needs we only take nth element from the data
    n = num
    
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
    
    #sub sample
    dataVect = dataVect[1:n]
    dataVectIndex = dataVectIndex[1:n]
    
    #put time step manually as unavailable from data file
    deltaT = (1/30)

    #print('We only used every ' + str(n) + 'th datapoint')
    return dataVect, dataVectIndex, deltaT