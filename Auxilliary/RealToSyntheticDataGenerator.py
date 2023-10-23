#necassary imports
import numpy as np
import numba as nb
import pickle
import h5py
import inspect
import os
from datetime import datetime
import matplotlib.pyplot as plt

#create a covariance matrix based on data at hand
@nb.njit(cache=True)
def covMat(coordinates1, coordinates2, covLambda, covL):

    #Create empty matrix for covariance
    C = np.zeros((len(coordinates1), len(coordinates2)))
    
    #loop over all indices in covariance matrix
    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            #Calculate distance between points
            dist = np.sqrt((coordinates1[i,0] - coordinates2[j,0])**2 + (coordinates1[i,1] - coordinates2[j,1])**2)
            #Determine each element of covariance matrix
            C[i, j] = (covLambda**2)*(np.exp(((-0.5)*((dist)**2))/(covL**2)))

    #Return Covariance Matrix
    return C

#function that saves diffusion function along with syntheticData
def saveFunction(function, file_path):
    """Save the source code of a function to a text file."""
    source_code = inspect.getsource(function)
    with open(file_path, 'w') as file:
        file.write(source_code)

#Define function that establishes form of diffusion coefficient through space use (nm^2)/s as units
def diffusion(x, y, matrix):
    value = (covMat(np.vstack((x ,y)).T, variables.induCoordinates, variables.covLambda, variables.covL) @ matrix)
    return value


#load in the three files associated with the synthetic Data
file = open("C:/Users/vkuma/PresseLab/Research/10000 3608.9641175884217 4750.10358variables.pkl", "rb")
variables = pickle.load(file)
file = open("/Users/vkuma/PresseLab/Research/10000 3608.9641175884217 4750.10358data.pkl", "rb")
data = pickle.load(file)
file = "/Users/vkuma/PresseLab/Research/Results (1).h5"
f = h5py.File(file, 'r')
dVect= f['d'][()]
pVect = f['P'][()]

#Extract specifics from data namespace
xData = data.trajectories[:, 0]
yData = data.trajectories[:, 1]
nTraj = len(np.unique(data.trajectoriesIndex))
nData = len(xData)

#initial constants
fieldOfView = [np.min(xData), np.max(xData), np.min(yData), np.max(yData)]  #[Xmin, Xmax, Ymin, Ymax] in nm for field of view
averageTrajLength = np.ceil(nData/nTraj)                                    #mean length of each trajectory
averageNumOfTraj = 175000/averageTrajLength                                 #mean for the number of trajectories as a multiple of 10
timestep = data.deltaT                                                      #temporal resolution (microscope frequency) in Hz

#The exact number of trajectories to be generated
#Note: this might not match the final trajectory count due to particles that leave field of view and return
nTraj = int(averageNumOfTraj + (np.random.binomial(2*averageNumOfTraj/10, 0.5) - averageNumOfTraj/10))

#empty lists that will save data
xVect = []
yVect = []
particleIndex = []

# Generate x and y coordinates for plotting
x = np.linspace(fieldOfView[0], fieldOfView[1], 100)
y = np.linspace(fieldOfView[2], fieldOfView[3], 100)
X, Y = np.meshgrid(x, y)

# matrix that represents learned Map in inverse space for computational efficiency
matrix = (variables.cInduInduInv @ np.mean(dVect[-100000:], 0))

# Calculate the diffusion values for each (x, y) coordinate
Z = diffusion(X.reshape(-1), Y.reshape(-1), matrix)
Z = Z.reshape(np.shape(X))

# Plot the diffusion function in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Diffusion Coefficient')
ax.set_title('Diffusion Function')

# Show the plot
plt.show()

#particle return tracker, this makes sure to count a particle leaving and 
#then returning in the field of view as two separate particles
tracker = 0
flag = False

# Learn the biased grid based on the dat
# Define the grid parameters
nGrid = 500
grid_size = (500, 500)  # Number of grid cells in each dimension

# Calculate the width and height of each grid cell
gridWidth = (fieldOfView[1] - fieldOfView[0]) / nGrid
gridHeight = (fieldOfView[3] - fieldOfView[2]) / nGrid

# Initialize an empty grid for density values
density_grid = np.zeros(grid_size)

# Iterate through the x and y coordinates
for x, y in zip(xData, yData):  # Replace x_coordinates and y_coordinates with your actual data
    # Map the (x, y) coordinate to the corresponding grid cell indices
    grid_x = int((x - fieldOfView[0]) / (fieldOfView[1] - fieldOfView[0]) * (grid_size[0] - 1))
    grid_y = int((y - fieldOfView[2]) / (fieldOfView[3] - fieldOfView[2]) * (grid_size[1] - 1))

    # Increment the density value for the corresponding grid cell
    density_grid[grid_y, grid_x] += 1

# Normalize the density values
biasValues = np.ones(np.shape(density_grid))/np.sum(np.ones(np.shape(density_grid)))  # Normalize by dividing by the total number of data points

# Generate data with biased initialization
for i in range(1, nTraj + 1):
    # Sample a grid cell based on the bias values
    gridProbs = biasValues.flatten() / np.sum(biasValues)  # Normalize the bias values to probabilities
    selectedGrid = np.random.choice(np.arange(nGrid**2), p=gridProbs)
    gridX = selectedGrid % nGrid
    gridY = selectedGrid // nGrid

    # Calculate the boundaries of the selected grid cell
    gridMinX = fieldOfView[0] + gridX * gridWidth
    gridMaxX = gridMinX + gridWidth
    gridMinY = fieldOfView[2] + gridY * gridHeight
    gridMaxY = gridMinY + gridHeight

    # Initialize positions within the selected grid cell
    xPrev = np.random.uniform(fieldOfView[0], fieldOfView[1])
    yPrev = np.random.uniform(fieldOfView[2], fieldOfView[3])

    xVect.append(xPrev)
    yVect.append(yPrev)
    particleIndex.append(i + tracker)

    # Sample trajectory length from a geometric distribution with mean 20 and minimum length of 5
    trajLength = 4 + np.random.geometric(p=(1/(averageTrajLength-4)))

    # Loop through the full length of each trajectory
    for j in range(1, trajLength + 1):
        # Sample diffusion
        dPoint = diffusion(xPrev, yPrev, matrix)[0]
        if dPoint < 0:
            dPoint = 0.1
        sd = np.sqrt(2 * dPoint * (timestep))
        xNew = np.random.normal(xPrev, sd)
        yNew = np.random.normal(yPrev, sd)

        # Save new positions if the particle is in the field of view
        if (fieldOfView[0] <= xNew <= fieldOfView[1]) and (fieldOfView[2] <= yNew <= fieldOfView[3]):
            # This part should be considered a new trajectory
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

        # Update positions
        xPrev = xNew
        yPrev = yNew

# Convert lists to NumPy arrays
arr1 = np.array(particleIndex)
arr2 = np.array(xVect)
arr3 = np.array(yVect)

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