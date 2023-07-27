import os
import numpy as np

# Assuming you have the data file path and other variables defined before this point
dataPath = 'C:/Users/vkuma/PresseLab/Research/Data/CleanData/movie001.txt'
scaleFactor = 1
timeStep = 1/30
maxDataPoints = 25e4
# Read the CSV file, considering the header
data = np.genfromtxt(dataPath, delimiter=', ', skip_header=1)

# Separate columns into individual arrays
dataVectIndex = data[:, 0]
dataVect = data[:, 1:]

# Make pixel adjustment to nanometers
dataVect = dataVect[::] * scaleFactor
dataVectIndex = dataVectIndex[::]

# Put time step manually as unavailable from the data file
deltaT = timeStep

# Get unique trajectory indices
unique_indices = np.unique(dataVectIndex)

# Calculate the number of subsets required
num_subsets = int(np.ceil(len(dataVectIndex)/ maxDataPoints))

# Randomly shuffle the trajectory indices
np.random.shuffle(unique_indices)

# Divide the shuffled trajectory indices into subsets
subsets = np.array_split(unique_indices, num_subsets)

# Create a directory to save the subsets if it doesn't exist
subset_dir = "subsets"
if not os.path.exists(subset_dir):
    os.makedirs(subset_dir)

for i, subset_indices in enumerate(subsets):
    # Filter the data to keep only the trajectories in the current subset
    mask = np.isin(dataVectIndex, subset_indices)
    selected_dataVect = dataVect[mask]
    selected_dataVectIndex = dataVectIndex[mask]

    # Save the subset to a CSV file
    subset_filename = os.path.join(subset_dir, f"subset_{i + 1}.csv")
    subset_data = np.column_stack((selected_dataVectIndex, selected_dataVect / scaleFactor))
    np.savetxt(subset_filename, subset_data, delimiter=', ', header='Trajectory Index, X Position, Y Position', comments='')