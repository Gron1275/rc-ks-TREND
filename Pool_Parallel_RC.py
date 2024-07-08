import generate_reservoir
import prediction_analysis
#import ks_integration
import rc
import numpy as np
#from ks_integration import periodicity_length, num_grid_points, time_step
import parallel_predict
import multiprocessing as mp
import time
import sys
import matplotlib.pyplot as plt

resparams = {
        'N': 3000,  # 6.08497359577532 at 1500 w degree 3, 5.0048007681229 at degree 10
        # for some reason sometimes increasing the num of nodes decreases accuracy. weird.
        'num_inputs': 512,
        'radius': 0.6,
        'degree': 3,
        'nonlinear_func': np.tanh,
        'sigma': 0.5,
        'train_length': 20000,
        # when time was at 9000 in the ss it got to like 10 lambdas. with 15000 its only getting 5.28. why? with 29000 it gets 5.928
        'beta': 0.01,
        'bias': 1.2,
        'overlap': 50,
        # 10 # nates thought with issues arising from overlap being too big is prob right. what is correlation function between points
        'inputs_per_reservoir': 64   # 4

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
X = np.load('X_seed0_L200_Q512_T100000_NOWWORKING.npy')
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
    discard_length = 2
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
    res_state = rc.reservoir_layer(reservoir, in_weight, loc, resparams, discard_length)
    #out_weights = rc.fit_output_weights(resparams=resparams, res_states=res_state, data=loc[overlap:inputs_per_reservoir + overlap, :train_length])
    tupool = (res_state, rc.fit_output_weights(resparams=resparams, res_states=res_state, data=loc[overlap:inputs_per_reservoir + overlap, discard_length:train_length]),i)
    print(f'Finished number: {i}')
    return tupool

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
    print(sys.argv)
    start = time.time()

    with mp.Pool() as pool:
        res_output_tuple = pool.map(train_parallel, range(int(num_inputs / inputs_per_reservoir)))


    print(f'Time taken to train: {time.time() - start}')

    # for thing in res_output_tuple:
    #     print(thing)
    resy, outy, ii = list(zip(*res_output_tuple))
    print(f'In order?: {ii}')
    dt = 0.25
    prediction_steps = 1000

    predictions = parallel_predict.parallel_predict(out_weights=outy,reservoir=reservoir,in_weight=in_weight,training_res_states=resy, time_steps=prediction_steps,resparams=resparams)
    print(f'Time to train and predict: {time.time() - start}')

    np.save(f"Predictions_N{nodes_num}_Q{num_inputs}_L200_T{train_length}",predictions)
    actual = X[:, resparams['train_length']: resparams['train_length'] + prediction_steps]
    t_pred = np.linspace(0, prediction_steps * dt - dt, prediction_steps)
    valid_time = prediction_analysis.valid_time(predictions, actual, t_pred)
    print(valid_time)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title("Kursiv_Overlay")
    # x = np.arange((predictions-real).shape[1]) * time_step / 20.83
    x = np.arange((predictions - actual).shape[1]) * 0.25
    y = np.arange((predictions - actual).shape[0]) * (200 / 512)
    x, y = np.meshgrid(x, y)
    # pcm = ax.pcolormesh(x, y, np.abs(predictions-real))
    pcm = ax.pcolormesh(x, y, predictions - actual)
    ax.set_ylabel("$x$")
    ax.set_xlabel("$t$")
    fig.colorbar(pcm, ax=ax, label="$overlay(x, t)$")
    plt.show()

    # HOPEFULLY COMING FROM SOMETHING IN PREDICTION STAGE. OTHERWISE, WHY IS IT ALL BASICALLY GREEN
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title("Kursiv_Predict")
    # x = np.arange(predictions.shape[1]) * time_step / 20.83
    x = np.arange(predictions.shape[1]) * 0.25
    y = np.arange(predictions.shape[0]) * (200 / 512)
    x, y = np.meshgrid(x, y)
    pcm = ax.pcolormesh(x, y, predictions)
    ax.set_ylabel("$x$")
    ax.set_xlabel("$t$")
    fig.colorbar(pcm, ax=ax, label="$pred(x, t)$")
    plt.show()
    #print(f"first name: {resy}\nlmast name: {outy} \nage: {ii}")
