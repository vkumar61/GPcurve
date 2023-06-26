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

plt.plot(pVect)
plt.show()