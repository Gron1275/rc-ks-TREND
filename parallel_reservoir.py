import generate_reservoir
import ks_integration
import rc
import prediction_analysis
import numpy as np
import matplotlib.pyplot as plt
from ks_integration import periodicity_length, num_grid_points, time_step
import parallel_predict


def train_parallel_rc(nodes_num: int, overlap: int, input_data: np.ndarray, resparams: dict, inputs_per_reservoir: int):
    # any way to make any of these lists into arrays and then do vectorizing stuff?
    # nate said that might not be efficient? why?
    # reservoirs = list()
    # in_weights = list()
    out_weights = list()
    reservoirs_states = list()

    num_inputs = resparams['num_inputs']
    degree = resparams['degree']
    radius = resparams['radius']
    sigma = resparams['sigma']
    train_length = resparams['train_length']

    print(num_inputs / inputs_per_reservoir)

    '''
    for i in range(int(num_inputs / inputs_per_reservoir)):
    # flag that determines if reservoirs are identical or not. If diff, seed=i, if not, seed=k for fixed k
    # if same reservoir for all could just make shallow copies.
    # all input weight layers should be the same as well. can use shallow copies
    # just need one res one input weight then attach diff input ranges to have the diff spatial things

        reservoirs.append(generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=i))
        in_weights.append(generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir+2*overlap, res_size=nodes_num, sigma=sigma))
    '''
    reservoir = generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=1)
    in_weight = generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir + 2 * overlap, res_size=nodes_num, sigma=sigma)
    for i in range(int(num_inputs / inputs_per_reservoir)):
        # i think that this could be done with multiprocessing. the fit output weights only differs by the parameter i
        # prob try to use multiprocessing.Pool here
        loc = chop_data(input_data, inputs_per_reservoir, overlap, i)
        '''
        reservoirs_states.append(rc.reservoir_layer(reservoirs[i], in_weights[i],loc,resparams))
        out_weights.append(rc.fit_output_weights(resparams, reservoirs_states[i], loc))
        '''
        reservoirs_states.append(rc.reservoir_layer(reservoir, in_weight, loc, resparams))
        out_weights.append(rc.fit_output_weights(resparams=resparams, res_states=reservoirs_states[i], data=loc[overlap:inputs_per_reservoir+overlap,:train_length]))


    return out_weights, in_weight, reservoirs_states, reservoir
    # for taking the output data, maybe just roll back the same way I do but compensate for the left padding then only
    # index to [0:n]


    # reservoirs_states.append(rc.reservoir_layer()) maybe put this in a different for loop so you dont hafta regenerate everything
# run the above for loop or perhaps with ndarrays find some way to vectorize and then once all that stuff is generated then do the
# reservoir_layer() step

def chop_data(data, n, m, step):
    index = np.arange(data.shape[0])
    # do this so it doesnt hafta query data. prolly should just have the parameter as num_inputs
    return data[np.roll(index, -n*step + m)[0:n+2*m], :]


p = [0.6, 3, 0.1] # spectral radius, degree, input scaling
resparams = {
    'N': 2000, # 6.08497359577532 at 1500 w degree 3, 5.0048007681229 at degree 10
    # for some reason sometimes increasing the num of nodes decreases accuracy. weird.
    'num_inputs': 64,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 9000, # when time was at 9000 in the ss it got to like 10 lambdas. with 15000 its only getting 5.28. why? with 29000 it gets 5.928
    'beta': 0.01,
    'bias': 1.2,
    'overlap': 10, # 10 # nates thought with issues arising from overlap being too big is prob right. what is correlation function between points
    'inputs_per_reservoir': 8 # 4

} # according to Solvable Model of Spatiotemporal Chaos (1993), for a system d < 3, correlation func should be exponential.
    # so maybe for exp(-d), set overlay (d) st exp(-d) < error for some error threshold? Also, prolly needs to factor in_per_res somehow maybe like inputs_per_res/64?
import time

X = ks_integration.int_plot(False)

start = time.time()
print(X.shape)
out_weights, in_weight, reservoirs_states, reservoir = train_parallel_rc(resparams['N'], resparams['overlap'], X, resparams, resparams['inputs_per_reservoir'])
end = time.time()
dt = 0.25
prediction_steps = 1000

predictions = parallel_predict.parallel_predict(out_weights=out_weights, reservoir=reservoir, in_weight=in_weight, training_res_states=reservoirs_states, time_steps=prediction_steps, resparams=resparams)
actual = X[:, resparams['train_length']: resparams['train_length'] + prediction_steps]

t_pred = np.linspace(0, prediction_steps * dt - dt, prediction_steps)
t_pred /= 20.83 # Lyapunov time for L=22



real = actual


valid_time = prediction_analysis.valid_time(predictions, real, t_pred)
print(f'Valid time: {valid_time}')
print(f'real shape {real.shape}')
print(f'predict shape {predictions.shape}')
fig, ax = plt.subplots(constrained_layout = True)
ax.set_title("Kursiv_Actual")
x = np.arange(real.shape[1]) * time_step / 20.83
y = np.arange(real.shape[0]) * periodicity_length / num_grid_points
x, y = np.meshgrid(x, y)
pcm = ax.pcolormesh(x, y, real)
ax.set_ylabel("$x$")
ax.set_xlabel("$t$")
fig.colorbar(pcm, ax = ax, label = "$u(x, t)$")
plt.show()

fig, ax = plt.subplots(constrained_layout = True)
ax.set_title("Kursiv_Predict")
x = np.arange(predictions.shape[1]) * time_step / 20.83
y = np.arange(predictions.shape[0]) * periodicity_length / num_grid_points
x, y = np.meshgrid(x, y)
pcm = ax.pcolormesh(x, y, predictions)
ax.set_ylabel("$x$")
ax.set_xlabel("$t$")
fig.colorbar(pcm, ax = ax, label = "$pred(x, t)$")
plt.show()

fig, ax = plt.subplots(constrained_layout = True)
ax.set_title("Kursiv_Overlay")
x = np.arange((predictions-real).shape[1]) * time_step / 20.83
y = np.arange((predictions-real).shape[0]) * periodicity_length / num_grid_points
x, y = np.meshgrid(x, y)
# pcm = ax.pcolormesh(x, y, np.abs(predictions-real))
pcm = ax.pcolormesh(x, y, predictions-real)
ax.set_ylabel("$x$")
ax.set_xlabel("$t$")
fig.colorbar(pcm, ax = ax, label = "$overlay(x, t)$")
plt.show()


print(f'Time for {resparams["N"]}: {end - start}')
