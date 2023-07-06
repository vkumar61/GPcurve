import csv
import numpy as np
import matplotlib.pyplot as plt
import functions
import objects
from types import SimpleNamespace
from matplotlib import cm
import ast
import pickle
import matplotlib
import h5py


file = open("C:/Users/vkuma/Research/2000 4084.5799477079127 2380.1035500000003variables.pkl", "rb")
variables = pickle.load(file)
file = open("C:/Users/vkuma/Research/2000 4084.5799477079127 2380.1035500000003data.pkl", "rb")
data = pickle.load(file)

file = "C:/Users/vkuma/Research/2000(4084.5799477079127 2380.1035500000003).h5"
f = h5py.File(file, 'r')
dVect= f['samples'][()]
pVect = f['prob'][()]

#This function generates a plot of the MAP as a contour plot
def plots(variables, dVect, pVect, data):

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #sample with maximum probability
    unshapedMap = cInduFine.T @ (cInduInduInv @ dVect[pVect.index(max(pVect))])
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.figure()
    mapPlot = plt.contour(shapedX, shapedY, shapedMap, levels = 25)
    plt.clabel(mapPlot, inline=1, fontsize=10)
    plt.scatter(trajectories[:,0], trajectories[:,1], alpha = 0.01)

    return fig

#This function plots the probability
def probPlot(pVect):

    #generate plot
    fig = plt.figure()
    plt.plot(pVect)
    plt.title("Probability per Sample")
    plt.xlabel("Iteration/Sample")
    plt.ylabel("Log Probability")

    return fig

#This function plots the mean of all dVect samples
def meanPlot(variables, dVect, data):

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #take mean of all samples
    unshapedMap = cInduFine.T @ (cInduInduInv @ np.mean(dVect, 0))
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.figure()
    mapPlot = plt.contour(shapedX, shapedY, shapedMap, levels = 25, cmap = cm.autumn)
    plt.clabel(mapPlot, inline=1, fontsize=10)
    plt.scatter(trajectories[:,0], trajectories[:,1], alpha = 0.01, c = "black")

    return fig

def plotThreeD(variables, dVect, data):

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #take mean of all samples
    unshapedMap = cInduFine.T @ (cInduInduInv @ np.mean(dVect, 0))
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.axes(projection='3d')
    fig.plot_surface(shapedX, shapedY, shapedMap, cmap=cm.coolwarm)
    fig.scatter3D(trajectories[:,0], trajectories[:,1], 0, color = "green", alpha = 0.1, label = "Particle Data")
    
    fig.set_xlabel(r"X ($\mu m$)")
    fig.set_ylabel(r"Y ($\mu m$)")
    fig.set_zlabel(r"Diff. Coefficient ($\mu m$)")
    fig.set_title('Learned Diffusion Map')

    fig.legend()
    return fig