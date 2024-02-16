#Necessary imports
import numpy as np
import inspect
import os
from datetime import datetime
import matplotlib.pyplot as plt

#initial constants
fieldOfView = [0, 25000, 0, 25000] #[Xmin, Xmax, Ymin, Ymax] in nm for field of view
averageTrajLength = 20     #mean length of each trajectory
averageNumOfTraj = 10000   #mean for the number of trajectories (must be a multiple of 10)
timestep = 1/30            #temporal resolution (microscope frequency) in Hz

#function that saves diffusion function along with syntheticData
def saveFunction(function, file_path):
    """Save the source code of a function to a text file."""
    source_code = inspect.getsource(function)
    with open(file_path, 'w') as file:
        file.write(source_code)

#Define function that establishes form of diffusion coefficient through space use (nm^2)/s as units
def diffusion(x, y, step_number):
    value = (1e5*np.exp(-step_number/averageTrajLength)*np.ones_like(x))
    return np.abs(value)

#The exact number of trajectories to be generated
#Note: this might not match the final trajectory count due to particles that leave field of view and return
nTraj = int(averageNumOfTraj + (np.random.binomial(2*averageNumOfTraj/10, 0.5) - averageNumOfTraj/10))

#empty lists that will save data
xVect = []
yVect = []
particleIndex = []
tracker = 0
flag = False

for i in range(1, nTraj + 1):
    step_number = 0
    xPrev = np.random.uniform(fieldOfView[0], fieldOfView[1])
    yPrev = np.random.uniform(fieldOfView[2], fieldOfView[3])

    xVect.append(xPrev)
    yVect.append(yPrev)
    particleIndex.append(i + tracker)

    trajLength = 4 + np.random.geometric(p=1/averageTrajLength)

    for j in range(1, trajLength + 1):
        dPoint = diffusion(xPrev, yPrev, step_number)
        sd = np.sqrt(2 * dPoint * (timestep))
        xNew = np.random.normal(xPrev, sd)
        yNew = np.random.normal(yPrev, sd)

        if (fieldOfView[0] <= xNew <= fieldOfView[1]) and (fieldOfView[2] <= yNew <= fieldOfView[3]):
            xVect.append(xNew)
            yVect.append(yNew)
            if flag:
                tracker += 1
                particleIndex.append(i + tracker)
                flag = False
            else:
                particleIndex.append(i + tracker)
        else:
            flag = True

        xPrev = xNew
        yPrev = yNew
        step_number += 1

# Convert lists to NumPy arrays
arr1 = np.array(particleIndex)
arr2 = np.array(xVect)
arr3 = np.array(yVect)
plt.scatter(arr1, arr2)
plt.show()

# Combine the arrays into a single 2D array
data = np.vstack((arr1, arr2, arr3)).T

# Generate a unique directory name using timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
directory = f"syntheticData_{timestamp}"

# Create the parent directory
parent_directory = "SyntheticData"
os.makedirs(parent_directory, exist_ok=True)

# Create the unique directory
unique_directory = os.path.join(parent_directory, directory)

# Check if the directory already exists(should not happen due to unique time stamp)
if os.path.exists(unique_directory):
    overwrite = input(f"The directory '{unique_directory}' already exists. Existing data may be overwritten. Do you want to continue? (y/n): ")
    if overwrite.lower() != 'y':
        print("Operation canceled. Please run the code again with a different directory name.")
        exit()

# Create the unique directory if it doesn't exist
os.makedirs(unique_directory, exist_ok=True)

# Specify the CSV file path
csv_file = os.path.join(unique_directory, 'data.csv')

# Save the data to the CSV file
np.savetxt(csv_file, data, delimiter=', ', fmt='%s', header='particle#, xPos, yPos', comments='')

# Specify the file path to save the diffusion source code
output_file = os.path.join(unique_directory, 'groundTruth.txt')

# Save the source code to the text file
saveFunction(diffusion, output_file)

print(f"The generated files are saved at: '{unique_directory}'.")