'''
When this file is run, it takes a given diffusion coefficient map, defined in the def diffusion(x, y),
and outputs randomized protien trajectories. The ground truth diffusion map is changed by changing the
function inside of diffusion(point), where point = (x,y) is a set of coordinates. There are some other
variables that the user may choose to tune, such as average trajectory length, field of view, etc. all 
of those variables are at the begining of the code and commented conveniantly so the user can changed 
conveniantly at initialization. Once all the variables are set the code simulates randomized trajectories
using a random walk.
'''

#Necessary imports
import numpy as np
import inspect
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#function that saves diffusion function along with syntheticData
def saveFunction(function, file_path):
    """Save the source code of a function to a text file."""
    source_code = inspect.getsource(function)
    with open(file_path, 'w') as file:
        file.write(source_code)

#Define function that establishes form of diffusion coefficient through space use (nm^2)/s as units
def diffusion(x, y):
    value = 50000 + 35000*(np.sin((x/10000)) + np.sin(y/2500) + np.sin((x+y)/5000) + np.sin(x*y/50000000)) + 75000*np.exp(-((x-10000)**2+(y-7500)**2)/10000000)
    return np.abs(value/2)

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
stochastic_init = True

# Define the biased regions as four rectangles and a square
biasWidth = (fieldOfView[1] - fieldOfView[0]) / 9  # Adjust the width of the rectangles as desired
biasRegions = [
    [fieldOfView[0] + biasWidth, fieldOfView[0] + 4 * biasWidth, fieldOfView[2], fieldOfView[3]],  # Left rectangle
    [fieldOfView[0] + 2 * biasWidth, fieldOfView[0] + 6 * biasWidth, fieldOfView[2], fieldOfView[3]],  # Middle rectangle
    [fieldOfView[0] + 5 * biasWidth, fieldOfView[0] + 7 * biasWidth, fieldOfView[2], fieldOfView[3]],  # Right rectangle
    [fieldOfView[0], fieldOfView[0] + 3 * biasWidth, fieldOfView[2], fieldOfView[2] + 3 * biasWidth]  # Bottom-left square
]

# Generate data with biased initialization
for i in range(1, nTraj + 1):
    # Sample a region based on the bias
    if not stochastic_init and np.random.rand() <= 0.995:  # 90% chance of sampling the biased region
        regionIndex = np.random.choice(len(biasRegions))
        region = biasRegions[regionIndex]

        # Adjust the height of the region
        if regionIndex == 0:  # Left rectangle
            regionHeight = np.random.uniform(0.1, 0.5)  # Adjust the range of height as desired
            regionWidth = np.random.uniform(0.5, 1)  # Adjust the range of height as desired
            # Initialize positions within the selected region
            xMin = region[0] + (region[1] - region[0]) * (1 - regionWidth) / 2
            xMax = region[1] - (region[1] - region[0]) * (1 - regionWidth) / 2
            yMin = region[2] + (region[3] - region[2]) * (1 - regionHeight) / 2
            yMax = region[3] - (region[3] - region[2]) * (1 - regionHeight) / 2

        elif regionIndex == 1:
            regionHeight = np.random.uniform(0.3, 1)  # Adjust the range of height as desired
            regionWidth = np.random.uniform(0.5, 1)  # Adjust the range of height as desired
            # Initialize positions within the selected region
            xMin = region[0] + (region[1] - region[0]) * (1 - regionWidth) / 2
            xMax = region[1] - (region[1] - region[0]) * (1 - regionWidth) / 2
            yMin = region[2] + (region[3] - region[2]) * (1 - regionHeight) / 2
            yMax = region[3] - (region[3] - region[2]) * (1 - regionHeight) / 2
        elif regionIndex == 2:
            regionHeight = np.random.uniform(0.1, 0.7)  # Adjust the range of height as desired
            regionWidth = np.random.uniform(0.5, 1)  # Adjust the range of height as desired
            # Initialize positions within the selected region
            xMin = region[0] + (region[1] - region[0]) * (1 - regionWidth) / 2
            xMax = region[1] - (region[1] - region[0]) * (1 - regionWidth) / 2
            yMin = region[2] + (region[3] - region[2]) * (1 - regionHeight) / 2
            yMax = region[3] - (region[3] - region[2]) * (1 - regionHeight) / 2
        else:  # Bottom-left square
            regionSize = np.random.uniform(0, 0.1)  # Adjust the size of the square as desired
            xMin = region[0]
            xMax = region[1] - regionSize * (region[1] - region[0])
            yMin = region[2]
            yMax = region[2] - regionSize * (region[3] - region[2])

        xPrev = np.random.uniform(xMin, xMax)
        yPrev = np.random.uniform(yMin, yMax)
    else:
        # Initialize positions randomly in the entire field of view
        xPrev = np.random.uniform(fieldOfView[0], fieldOfView[1])
        yPrev = np.random.uniform(fieldOfView[2], fieldOfView[3])

    xVect.append(xPrev)
    yVect.append(yPrev)
    particleIndex.append(i + tracker)


    #sample trajectory length from geometric with mean 20
    trajLength = np.random.geometric(p=1/20)


    #loop through full length of each trajectory
    for j in range(1, trajLength+1):

        #Sample diffusion
        dPoint = diffusion(xPrev, yPrev)
        sd = np.sqrt(2*dPoint*(timestep))
        xNew = np.random.normal(xPrev, sd)
        yNew = np.random.normal(yPrev, sd)

        #save new positions if particle is in field of view
        if (fieldOfView[0] <= xNew <= fieldOfView[1]) and (fieldOfView[2] <= yNew <= fieldOfView[3]):
            #this part should be considered a new trajectory
            xVect.append(xNew)
            yVect.append(yNew)
            if flag:
                tracker += 1
                particleIndex.append(i+tracker)
                flag = False
            else:
                particleIndex.append(i+tracker)
        else:
            flag = True

        #update positions
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