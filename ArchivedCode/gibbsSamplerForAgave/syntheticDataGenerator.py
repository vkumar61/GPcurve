'''
When this file is run, it takes a given diffusion coefficient map, defined in the def diffusion(x, y),
and outputs randomized protien trajectories. The ground truth diffusion map is changed by changing the
function inside of diffusion(point), where point = (x,y) is a set of coordinates. There are some other
variables that the user may choose to tune, such as average trajectory length, field of view, etc. all 
of those variables are at the begining of the code and commented conveniantly so the user can changed 
conveniantly at initialization. Once all the variables are set the code simulates randomized trajectories
using a random walk.
'''

#Necassary imports
import numpy as np
import inspect
import os
from datetime import datetime

#function that saves diffusion function along with syntheticData
def saveFunction(function, file_path):
    """Save the source code of a function to a text file."""
    source_code = inspect.getsource(function)
    with open(file_path, 'w') as file:
        file.write(source_code)

#Define function that establishes form of diffusion coefficient through space use (nm^2)/s as units
def diffusion(x, y):
    value = 50000 + 35000*(np.sin((x/10000)) + np.sin(y/2500) + np.sin((x+y)/5000) + np.sin(x*y/50000000))
    return value

#initial constants
fieldOfView = [0, 20000, 0, 20000] #[Xmin, Xmax, Ymin, Ymax] in nm for field of view
averageTrajLength = 20     #mean length of each trajectory
averageNumOfTraj = 5000   #mean for the number of trajectories as a multiple of 10
timestep = 1/30            #temporal resolution (microscope frequency) in Hz

#The exact number of trajectories to be generated
#Note: this might not match the final trajectory count due to particle that leave field of view and return
nTraj = int(averageNumOfTraj + (np.random.binomial(2*averageNumOfTraj/10, 0.5) - averageNumOfTraj/10))

#empty lists that will save data
xVect = []
yVect = []
particleIndex = []


#particle return tracker, this makes sure to count a particle leaving and 
#then returning in the field of view as two separate particles
tracker = 0
flag = False

for i in range(1, nTraj+1):
    #initialinitialize positions
    xPrev = np.random.uniform(fieldOfView[0], fieldOfView[1])
    yPrev = np.random.uniform(fieldOfView[2], fieldOfView[3])
    xVect.append(xPrev)
    yVect.append(yPrev)
    particleIndex.append(i+tracker)


    #tample trajectory length Binomial(n, p) ~ Normal(n*p, sqrt(n*p*(1-p)))
    trajLength = np.random.binomial(2*averageTrajLength, 0.5)


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