
import numba as nb
import numpy as np

# Kernel function
@nb.njit(cache=True)
def kernel_FRET(x1, x2, sig, ell):

    # Get constants
    num_1, num_dims = x1.shape
    num_2, ________ = x2.shape
    sig2 = sig**2
    ell2 = ell**2

    # Create kernel
    K = np.zeros((num_1, num_2))
    for i in range(num_1):
        for j in range(num_2):
            K[i, j] = sig2 * np.exp(-.5 * np.sum((x1[i, :] - x2[j, :]) ** 2) / ell2)

    return K
