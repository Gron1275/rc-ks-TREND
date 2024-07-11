import generate_reservoir
import ks_integration
import prediction_analysis
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
# import predict_single_output_layer


square_nodes = False
time_step = 0.25
prediction_steps = 1000

discard_length = 500
batch_len = 1000

num_grid_points = 512
periodicity_length = 200
num_reservoirs = 64
simulation_length = 50000
p = [0.6, 3, .1] #1] # spectral radius, degree, input scaling
resparams = {
    'N': 2000,
    'num_inputs': num_grid_points,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 38000,
    'beta': 0.0001,
    'bias': 0.8,
    'overlap': 12,
    'inputs_per_reservoir': 16
}

reservoir = generate_reservoir.erdos_renyi_reservoir(
        size = resparams['N'],
        degree = resparams['degree'],
        radius = resparams['radius'],
        seed = 1
        )
W_in = generate_reservoir.generate_W_in(
        num_inputs = resparams['inputs_per_reservoir'] + 2 * resparams['overlap'],
        res_size = resparams['N'],
        sigma = resparams['sigma']
        )


def reservoir_layer(A, W_in, input, resparams, batch_len, discard_length):
    N = resparams['N']
    train_length = resparams['train_length']
    g = resparams['nonlinear_func']
    bias = resparams['bias']

    res_states = np.zeros((N, input.shape[1]))

    for i in range(input.shape[1] - 1):
        res_states[:, i + 1] = g(A @ res_states[:, i] + W_in @ input[:, i] + bias)
    return res_states[:, discard_length:].copy()


def chop_data(data, IPR, overlap, step):
    index = np.arange(data.shape[0])
    return data[np.roll(index, -IPR * step + overlap)[0: IPR + 2 * overlap], :].copy()

