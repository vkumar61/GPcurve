import numpy as np

file = 'dataset1/movie011.txt'
#read datafile
with open(file) as inp:
    tempData = [i.strip().split('\t') for i in inp]

#clean the data
cleanData = []
for i in tempData:
    if i != ['']:
        cleanData.append(i)

#coordinates and indicies for trajectories
x = np.array([float(i[3]) for i in cleanData])
y = np.array([float(i[4]) for i in cleanData])
dataVectIndex = np.array([int(i[1]) for i in cleanData])

#organize arrays
arr1 = np.array(dataVectIndex)
arr2 = np.array(x)
arr3 = np.array(y)

# Combine the arrays into a single 2D array
data = np.vstack((arr1, arr2, arr3)).T

# Save the data to the CSV file
np.savetxt('movie011.txt', data, delimiter=', ', fmt='%s', header='particle#, xPos, yPos', comments='')
