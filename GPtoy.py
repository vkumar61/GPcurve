
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace

PARAMETERS = {

    # Variables
    'P': None,
    'alpha': None,
    'f_indu': None,

    # Priors
    'ell': 1,
    'sig': 10,
    'f_indu_mean': 0,
    'alpha_shape': 2,
    'alpha_scale': None,
    'x_data': None,
    'x_indu': None,
    'x_grid': None,

    # Covariance matrices
    'C_indu_indu': None,
    'C_indu_data': None,
    'C_indu_grid': None,
    'C_indu_indu_inv': None,

    # Numbers
    'n_data': None,
    'n_grid': 1000,
    'n_indu': 50,

    # Sampler parameters
    'epsilon': 1e-3,
    'f_indu_prop_shape': .1,
}


def initialize_variables(data, parameters):


    variables = SimpleNamespace(**parameters)

    # extract variables
    x_data = variables.x_data
    x_grid = variables.x_grid
    f_indu = variables.f_indu
    n_data = variables.n_data
    n_grid = variables.n_grid
    n_indu = variables.n_indu
    ell = variables.ell
    sig = variables.sig
    a_noise = variables.a_noise
    b_noise = variables.b_noise
    C_indu_indu = variables.C_indu_indu
    C_indu_data = variables.C_indu_data
    f_indu_prop_width = variables.f_indu_prop_width

    x_indu = np.linspace(np.min(variables.x_data), np.max(variables.x_data), variables.n_indu)
    variables.x_grid = np.linspace(np.min(variables.x_data), np.max(variables.x_data), variables.n_grid)
    variables.f_indu = np.ones(variables.n_indu) * variables.f_indu_mean
    #...
    C_indu_indu_inv = np.linalg.inv(variables.C_indu_indu + epsilon*np.eye(variables.n_indu))

    if alpha_scale is None:
        alpha_scale = np.mean(data)/alpha_shape
    variables.alpha_scale = alpha_scale


    # Initialize covariance matrices
    # ...

    return variables


def sample_f_indu(data, variables):

    # extract variables
    x_data = variables.x_data
    x_grid = variables.x_grid
    f_indu = variables.f_indu
    n_data = variables.n_data
    n_grid = variables.n_grid
    n_indu = variables.n_indu
    ell = variables.ell
    sig = variables.sig
    a_noise = variables.a_noise
    b_noise = variables.b_noise
    C_indu_indu = variables.C_indu_indu
    C_indu_data = variables.C_indu_data
    f_indu_prop_width = variables.f_indu_prop_width


    # Propose new f_indu
    f_indu_old = f_indu.copy()
    f_indu_new = f_indu_old + np.random.rand(n_indu) * f_indu_prop_width

    # Calculate probabilities
    def probability(f_indu_):


        # Prior
        prior = stats.mvnormal.logpdf(f_indu_, loc=f_indu_mean, cov=C_indu_indu)

        f_data = C_indu_data.T @ C_indu_indu_inv @ f_indu_
        lhood = 0
        for n in range(n_data):
            lhood += stats.gamma.logpdf(data[n], a_noise, scale=f_data[n] / a_noise)
        lhood = np.sum(stats.gamma.logpdf(data, a=a_noise, scale=f_data/a_noise))

        prob = lhood + prior

        return prob


    # Accept or reject
    P_old = probability(f_indu_old)
    P_new = probability(f_indu_new)
    acc_prob = (
        P_new - P_old
    )
    if np.log(np.random.rand()) < acc_prob:
        f_indu = f_indu_new

    # Update
    variables.f_indu = f_indu

    return variables


def Gibbs_sampler(data, parameters, num_iterations=10000):

    parameters = {**PARAMETERS, **parameters}

    # Initialize variables
    variables = initialize_variables(data, parameters)

    # Initialize Saves
    MAP = copy.deepcopy(variables)
    samplecache = {
        'f_indu': [variables.f_indu],
        'alpha': [variables.alpha],
    }


    # Gibbs sampler
    for iteration in range(num_iterations):

        # Sample f_indu
        variables = sample_f_indu(data, variables)

        # Sample a_noise
        variables = sample_a_noise(data, variables)

        variables.P = calculate_posterior(variables)


        # Save samples
        samplecache['f_indu'].append(variables.f_indu)
        samplecache['alpha'].append(variables.alpha)
        if variables.P >= MAP.P:
            MAP = copy.deepcopy(variables)





