#Necessary imports
import numpy as np
import inspect
import os
from datetime import datetime
import matplotlib.pyplot as plt

#function that saves diffusion function along with syntheticData
def saveFunction(function, file_path):
    """Save the source code of a function to a text file."""
    source_code = inspect.getsource(function)
    with open(file_path, 'w') as file:
        file.write(source_code)

#Define function that establishes form of diffusion coefficient through space use (nm^2)/s as units
def diffusion(x, y):
    value = (1e5 + 50000*(np.sin(x/10000)+np.sin(y/10000)) + 
            20000*np.exp(-((x-5000)**2+(y-5000)**2)/1e7) + 
            20000*np.exp(-((x-10000)**2+(y-7500)**2)/1e7) + 
            20000*np.exp(-((x-15000)**2+(y-17500)**2)/1e7) + 
            20000*np.exp(-((x-5000)**2+(y-17500)**2)/1e7) + 
            20000*np.exp(-((x-17500)**2+(y-17500)**2)/1e7) + 
            20000*np.exp(-((x-2500)**2+(y-2500)**2)/1e7))
    return np.abs(value/4)


#initial constants
fieldOfView = [0, 20000, 0, 20000] #[Xmin, Xmax, Ymin, Ymax] in nm for field of view
averageTrajLength = 20     #mean length of each trajectory
averageNumOfTraj = 8000   #mean for the number of trajectories as a multiple of 10
timestep = 1/30            #temporal resolution (microscope frequency) in Hz

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

# Calculate the diffusion values for each (x, y) coordinate
Z = diffusion(X, Y)

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

# Set this flag to True for unbiased initialization
stochastic_init = False

# Define the number of grid for field of view and subgrid that dominates
nGrid = 500
centerGrid = 400

# Initialize the bias values for each grid
biasValues = np.zeros((nGrid, nGrid))

# Calculate the width and height of each grid cell
gridWidth = (fieldOfView[1] - fieldOfView[0]) / nGrid
gridHeight = (fieldOfView[3] - fieldOfView[2]) / nGrid

# Assign a total probability of 99.9% to the center 400x400 grids
centerGridStart = (nGrid - centerGrid) // 2
centerGridEnd = centerGridStart + centerGrid
centerGridTotalProb = 0.99

# Randomly assign probabilities to each cell in the center 400x400 grids
centerGridProbs = np.random.dirichlet(np.ones(centerGrid * centerGrid), size=1) * centerGridTotalProb
biasValues[centerGridStart:centerGridEnd, centerGridStart:centerGridEnd] = centerGridProbs.reshape((centerGrid, centerGrid))

for i in range(200):
    randomSpotX = np.random.randint(0,nGrid-100)
    sizeX = np.random.randint(2, 50)
    randomSpotY = np.random.randint(0,nGrid-100)
    sizeY = np.random.randint(2, 50)
    biasValues[randomSpotX:randomSpotX+sizeX, randomSpotY:randomSpotY+sizeY] = np.zeros((sizeX, sizeY))

# Distribute the remaining probability randomly to the rest of the grids
remainingGrids = np.where(biasValues == 0)
remainingGridsCount = np.sum(remainingGrids)
remainingGridProbEach = (1 - centerGridTotalProb) / remainingGridsCount
biasValues[remainingGrids] = remainingGridProbEach

# Generate data with initialization based on the flag value
if stochastic_init:
    # Generate data with purely stochastic initialization
    for i in range(1, nTraj + 1):
        xPrev = np.random.uniform(fieldOfView[0], fieldOfView[1])
        yPrev = np.random.uniform(fieldOfView[2], fieldOfView[3])

        xVect.append(xPrev)
        yVect.append(yPrev)
        particleIndex.append(i + tracker)

        trajLength = 4 + np.random.geometric(p=1/20)

        for j in range(1, trajLength + 1):
            dPoint = diffusion(xPrev, yPrev)
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
else:
    # Generate data with biased initialization
    for i in range(1, nTraj + 1):
        gridProbs = biasValues.flatten() / np.sum(biasValues)  # Normalize the bias values to probabilities
        selectedGrid = np.random.choice(np.arange(nGrid**2), p=gridProbs)
        gridX = selectedGrid % nGrid
        gridY = selectedGrid // nGrid

        gridMinX = fieldOfView[0] + gridX * gridWidth
        gridMaxX = gridMinX + gridWidth
        gridMinY = fieldOfView[2] + gridY * gridHeight
        gridMaxY = gridMinY + gridHeight

        xPrev = np.random.uniform(gridMinX, gridMaxX)
        yPrev = np.random.uniform(gridMinY, gridMaxY)

        xVect.append(xPrev)
        yVect.append(yPrev)
        particleIndex.append(i + tracker)

        trajLength = 4 + np.random.geometric(p=1/20)

        for j in range(1, trajLength + 1):
            dPoint = diffusion(xPrev, yPrev)
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