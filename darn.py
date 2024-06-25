import generate_reservoir
import ks_integration
import rc
import prediction_analysis
import numpy as np
import matplotlib.pyplot as plt
from ks_integration import periodicity_length, num_grid_points, time_step
import parallel_predict
import multiprocessing as mp
import time

def chop_data(data, n, m, step):
    index = np.arange(data.shape[0])
    # do this so it doesnt hafta query data. prolly should just have the parameter as num_inputs
    return data[np.roll(index, -n*step + m)[0:n+2*m], :]

def train_parallel(input_data, reservoirs_states, out_weights_list,i, inputs_per_reservoir, overlap,train_length, reservoir, in_weight,resparams,idk):
    loc = chop_data(input_data, inputs_per_reservoir, overlap, i)

    '''
    reservoirs_states.append(rc.reservoir_layer(reservoirs[i], in_weights[i],loc,resparams))
    out_weights.append(rc.fit_output_weights(resparams, reservoirs_states[i], loc))
    '''
    reservoirs_states.append(rc.reservoir_layer(reservoir, in_weight, loc, resparams))
    out_weights_list.append(rc.fit_output_weights(resparams=resparams, res_states=reservoirs_states[i], data=loc[overlap:inputs_per_reservoir + overlap, :train_length]))
    idk.append(i)


if __name__ == "__main__":
    p = [0.6, 3, 0.1]  # spectral radius, degree, input scaling
    resparams = {
        'N': 1000,  # 6.08497359577532 at 1500 w degree 3, 5.0048007681229 at degree 10
        # for some reason sometimes increasing the num of nodes decreases accuracy. weird.
        'num_inputs': 64,
        'radius': p[0],
        'degree': p[1],
        'nonlinear_func': np.tanh,
        'sigma': p[2],
        'train_length': 9000,
        # when time was at 9000 in the ss it got to like 10 lambdas. with 15000 its only getting 5.28. why? with 29000 it gets 5.928
        'beta': 0.01,
        'bias': 1.2,
        'overlap': 10,
        # 10 # nates thought with issues arising from overlap being too big is prob right. what is correlation function between points
        'inputs_per_reservoir': 8   # 4

    }
    num_inputs = resparams['num_inputs']
    degree = resparams['degree']
    radius = resparams['radius']
    sigma = resparams['sigma']
    train_length = resparams['train_length']
    nodes_num = resparams['N']
    inputs_per_reservoir = resparams['inputs_per_reservoir']
    overlap = resparams['overlap']


    reservoir = generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=1)
    in_weight = generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir + 2 * overlap, res_size=nodes_num,sigma=sigma)
    X = ks_integration.int_plot(False)
    start = time.time()
    with mp.Manager() as manager:
        reservoir_states = manager.list()
        out_weights = manager.list()
        idk = manager.list()
        p = list()
        for i in range(int(num_inputs / inputs_per_reservoir)):
            p.append(mp.Process(target=train_parallel, args=(X, reservoir_states, out_weights, i, inputs_per_reservoir,overlap,train_length, reservoir, in_weight,resparams,idk)))
            p[i].start()
        # for i in range(int(num_inputs / inputs_per_reservoir)):
        #     p[i].join()
        end = time.time()
        print(f'Time for {resparams["N"]}: {end - start}')
        print(out_weights)
        print(reservoir_states)


print(__name__)