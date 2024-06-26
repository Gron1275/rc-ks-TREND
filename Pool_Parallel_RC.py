import generate_reservoir
#import ks_integration
import rc
import numpy as np
#from ks_integration import periodicity_length, num_grid_points, time_step
import parallel_predict
import multiprocessing as mp
import time

resparams = {
        'N': 2000,  # 6.08497359577532 at 1500 w degree 3, 5.0048007681229 at degree 10
        # for some reason sometimes increasing the num of nodes decreases accuracy. weird.
        'num_inputs': 512,
        'radius': 0.6,
        'degree': 3,
        'nonlinear_func': np.tanh,
        'sigma': 0.1,
        'train_length': 69000,
        # when time was at 9000 in the ss it got to like 10 lambdas. with 15000 its only getting 5.28. why? with 29000 it gets 5.928
        'beta': 0.001,
        'bias': 1.2,
        'overlap': 10,
        # 10 # nates thought with issues arising from overlap being too big is prob right. what is correlation function between points
        'inputs_per_reservoir': 32   # 4

    }
nodes_num = resparams['N']
num_inputs = resparams['num_inputs']
radius = resparams['radius']
degree = resparams['degree']
g = resparams['nonlinear_func']
sigma = resparams['sigma']
train_length = resparams['train_length']

overlap = resparams['overlap']
inputs_per_reservoir = resparams['inputs_per_reservoir']

resparams['reservoir'] = generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius,seed=1)
resparams['input_weight'] = generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir + 2 * overlap,res_size=nodes_num, sigma=sigma)
# X = ks_integration.int_plot(False)
X = np.load('X_seed0.npy')
resparams['data'] = X
reservoir = generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=1)
in_weight = generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir + 2 * overlap, res_size=nodes_num,sigma=sigma)
# def chop_data(resparams, step):
def chop_data(data, n, m, step):
#     data = resparams['data']
#     n = resparams['inputs_per_reservoir']
#     m = resparams['overlap']

    index = np.arange(data.shape[0])
    # do this so it doesnt hafta query data. prolly should just have the parameter as num_inputs
    return data[np.roll(index, -n*step + m)[0:n+2*m], :]


def train_parallel(i):
    # global resparams
    # global reservoir
    # global in_weight
    global resparams
    global reservoir
    global in_weight
    # print(resparams)
    data = resparams['data']
    n = resparams['inputs_per_reservoir']
    m = resparams['overlap']
    loc = chop_data(data, n, m, i)

    '''
    reservoirs_states.append(rc.reservoir_layer(reservoirs[i], in_weights[i],loc,resparams))
    out_weights.append(rc.fit_output_weights(resparams, reservoirs_states[i], loc))
    '''
    print(f'Starting number: {i}')
    res_state = rc.reservoir_layer(reservoir, in_weight, loc, resparams)
    out_weights = rc.fit_output_weights(resparams=resparams, res_states=res_state, data=loc[overlap:inputs_per_reservoir + overlap, :train_length])
    print(f'Finished number: {i}')
    return res_state

if __name__ == "__main__":
    # p = [0.6, 3, 0.1]  # spectral radius, degree, input scaling
    # resparams = {
    #     'N': 2000,  # 6.08497359577532 at 1500 w degree 3, 5.0048007681229 at degree 10
    #     # for some reason sometimes increasing the num of nodes decreases accuracy. weird.
    #     'num_inputs': 64,
    #     'radius': 0.6,
    #     'degree': 3,
    #     'nonlinear_func': np.tanh,
    #     'sigma': 0.1,
    #     'train_length': 9000,
    #     # when time was at 9000 in the ss it got to like 10 lambdas. with 15000 its only getting 5.28. why? with 29000 it gets 5.928
    #     'beta': 0.001,
    #     'bias': 1.2,
    #     'overlap': 10,
    #     # 10 # nates thought with issues arising from overlap being too big is prob right. what is correlation function between points
    #     'inputs_per_reservoir': 16   # 4
    #
    # }
    # nodes_num = resparams['N']
    # num_inputs = resparams['num_inputs']
    # radius = resparams['radius']
    # degree = resparams['degree']
    # g = resparams['nonlinear_func']
    # sigma = resparams['sigma']
    # train_length = resparams['train_length']
    #
    # overlap = resparams['overlap']
    # inputs_per_reservoir = resparams['inputs_per_reservoir']
    #
    # resparams['reservoir'] = generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius,seed=1)
    # resparams['input_weight'] = generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir + 2 * overlap,res_size=nodes_num, sigma=sigma)
    # # X = ks_integration.int_plot(False)
    # X = np.load('X_seed11000.npy')
    # resparams['data'] = X
    # reservoir = generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=1)
    # in_weight = generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir + 2 * overlap, res_size=nodes_num,sigma=sigma)
    start = time.time()

    with mp.Pool() as pool:
        reservoir_states = pool.map(train_parallel, range(int(num_inputs / inputs_per_reservoir)))


    print(f'Time taken: {time.time() - start}')
