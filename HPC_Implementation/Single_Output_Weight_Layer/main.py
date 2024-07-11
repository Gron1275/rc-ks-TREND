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


def fit_output_weight_chunk(chunk_id):
    global N, IPR, overlap, discard_length, train_length, X, batch_len
    loc = chop_data(X, IPR, overlap, chunk_id)
    data = loc[overlap: -overlap, :train_length]

    V_batches = np.split(data[:, batch_len + discard_length: train_length], np.arange(0, data[:, batch_len + discard_length: train_length].shape[1], batch_len), axis=1)
    S_batches = np.split(loc[:, batch_len + discard_length: train_length], np.arange(0, loc[:, batch_len + discard_length: train_length].shape[1], batch_len), axis=1)

    first_S = loc[:, :batch_len + discard_length]

    S_batches[0] = first_S
    V_batches[0] = data[:, discard_length: batch_len + discard_length]

    res_state = reservoir_layer(reservoir, W_in, first_S, resparams, batch_len=batch_len, discard_length=discard_length)

    # SS_T += res_state @ res_state.T
    # VS_T += V_batches[0] @ res_state.T


    return None







def shit(i):
    print(i)
    return(i)


if __name__ == "__main__":
    # N = resparams['N']
    # train_length = resparams['train_length']
    # IPR = resparams['inputs_per_reservoir']
    # num_inputs = resparams['num_inputs']
    # overlap = resparams['overlap']
    # beta = resparams['beta']
    #
    # X = ks_integration.int_plot(
    #     time_range=simulation_length,
    #     periodicity_length=periodicity_length,
    #     num_grid_points=num_grid_points,
    #     time_step=time_step,
    #     plot=False
    # )
    #
    # SS_T = np.zeros((N, N))
    # VS_T = np.zeros((IPR, N))

    with mp.Pool() as pool:
        ar = pool.map_async(shit, range(10000))
        print(ar.get())
    # if ar.ready():
    #     print(ar.get())
    #print(ar.get(timeout=10))



