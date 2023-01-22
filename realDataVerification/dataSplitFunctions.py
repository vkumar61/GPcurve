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


    #put time step manually as unavailable from data file
    deltaT = (1/30)


    #Adjust units of data to micrometers
    
    #create grid of unit size 10x10 based on where the data exists
    splitData = {}
    dataRounded = np.floor(dataVect/10)*10
    for i in np.unique(dataRounded[:,0]):
        splitData[i] = {}
        for j in np.unique(dataRounded[:,1]):
            splitData[i][j] = np.empty((0,2))
    
    #split up data based on grid location
    for i in dataVect:
        x = np.floor(i[0]/10)*10
        y = np.floor(i[1]/10)*10

        splitData[x][y] = np.vstack((splitData[x][y], i))
    print(splitData[50][160])
    
    #print('We only used every ' + str(n) + 'th datapoint')
    return splitData, deltaT

def mle(sampleCoordinates, dataCoordinates, deltaT):

    #Initial Guess with MLE
    diff = sampleCoordinates - dataCoordinates
    num = np.sum(diff * diff)
    den = 4*deltaT*len(diff)
    mle = num/den

    return mle