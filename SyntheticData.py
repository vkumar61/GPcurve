#importing useful libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math

def dataGenerator(length, mean, sigma):
        
        def function(x, mean, sigma):

            y = 1000*(math.e**((-1*(x - mean)**2)/(2*sigma**2)))/(math.sqrt(2*math.pi)*sigma)
            return y

        xground = np.linspace(-10, 10, length)
        yground = function(xground, mean, sigma)

        #vmatrix = np.diag(np.full(len(yground), var))
        data = np.random.gamma(10, yground/10, len(yground))
        return xground, yground, data
