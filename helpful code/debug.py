import numpy as np
import numba as nb
import pickle as pkl
import h5py

file = open("C:/Users/vkuma/Research/200 4084.5799477079127 2380.1035500000003variables.pkl", "rb")
variables = pkl.load(file)
file = open("C:/Users/vkuma/Research/200 4084.5799477079127 2380.1035500000003data.pkl", "rb")
data = pkl.load(file)

file = "C:/Users/vkuma/Research/200(4084.5799477079127 2380.1035500000003).h5"
f = h5py.File(file, 'r')
dVect= f['samples'][()]
pVect = f['prob'][()]


#@nb.jit(nopython=True)
def gradient_descent(inducing_points, cDataIndu, priorMean, cInduInduInv, samples, means, deltaT, learning_rate=0.01, num_iterations=100):
    """
    Performs gradient descent to find the maximum of a function.

    Args:
        posterior (function): The posterior function to maximize.
        inducing_points (ndarray): The initial inducing points.
        learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.
        num_iterations (int, optional): The number of iterations for gradient descent. Defaults to 100.

    Returns:
        ndarray: The updated inducing points.
    """
    def posterior(dIndu_):
            dData_ = cDataIndu @ dIndu_
            # Prior
            diff = dIndu_ - priorMean
            prior = -0.5*(diff.T @ (cInduInduInv @ diff))
            
            #Likelihood of that data
            lhood = 0
            for i in range(samples.shape[0]):
                for j in range(samples.shape[1]):
                    lhood += (
                        -.5 * (samples[i, j] - means[i, j])**2 / (2*dData_[i]*deltaT)
                        -.5 * np.log(2*np.pi*2*dData_[i]*deltaT)
                    )
            prob = lhood + prior

            return prob
    num_points = inducing_points.shape[0]

    for _ in range(num_iterations):
        gradients = np.zeros_like(inducing_points)
        diff = np.zeros_like(inducing_points)
        posterior_value = posterior(inducing_points)

        # Compute the gradient for each inducing point
        for i in range(num_points):
            diff[i] = inducing_points[i]*0.01
            gradients[i] = (posterior(diff) - posterior_value)

        # Update the inducing points using the gradients
        inducing_points += learning_rate * gradients

    return inducing_points


# Initial inducing points
initial_points = variables.priorMean
cDataIndu = variables.cDataIndu
priorMean = variables.priorMean
cInduInduInv = variables.cInduInduInv
samples = variables.sampleCoordinates
means = variables.dataCoordinates
deltaT = data.deltaT


# Call gradient_descent function
updated_points = gradient_descent(initial_points, cDataIndu, priorMean, cInduInduInv, samples, means, deltaT, learning_rate=0.1, num_iterations=1)

print("Initial inducing points:")
print(initial_points)
print("Updated inducing points:")
print(updated_points)