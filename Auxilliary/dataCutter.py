import os
import numpy as np

# Define necassary variables
dataPath = 'C:/Users/vkuma/PresseLab/Research/Data/CleanData/movie003.txt'

# Read the CSV file, considering the header
data = np.genfromtxt(dataPath, delimiter=', ', skip_header=1)

# Split the data in half using slicing
data1 = data[::2]
data2 = data[1::2]

# Save the new arrays to separate files in the respective directory
os.makedirs('movie003', exist_ok=True)
np.savetxt('movie003/data1.csv', data1, delimiter=', ', header='Trajectory Index, X Position, Y Position', comments='')
np.savetxt('movie003/data2.csv', data2, delimiter=', ', header='Trajectory Index, X Position, Y Position', comments='')