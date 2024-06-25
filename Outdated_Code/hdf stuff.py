import h5py
import numpy as np

import lorenz
import generate_reservoir
import rc
import prediction_analysis

with h5py.File('test2.hdf', 'r') as f:
    for key in f.keys():
        print(key)
    data = f['data']
    print(data)
reader = h5py.File('test2.hdf','r')
X = np.array(reader['data'][:]).T
print(X.shape)
print(X)
def train_vanilla_RC(resparams, data):
    """TODO"""
    N = resparams['N']
    radius = resparams['radius']
    degree = resparams['degree']
    num_inputs = resparams['num_inputs']
    sigma = resparams['sigma']
    train_length = resparams['train_length']

    A = generate_reservoir.erdos_renyi_reservoir(N, degree, radius)
    W_in = generate_reservoir.generate_W_in(num_inputs, N, sigma)

    res_states = rc.reservoir_layer(A, W_in, data, resparams)

    W_out = rc.fit_output_weights(resparams, res_states, data[:train_length])

    return W_out, A, W_in, res_states
# Specify reservoir parameters
p = [0.9, 2.6667, 0.5] # spectral radius, degree, input scaling
approx_res_size = 500
resparams = {
    'num_inputs': 128,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 800,
    'beta': 0.00003,
    'bias': 0.5
}
resparams['N'] = int(np.floor(approx_res_size / resparams['num_inputs'])) * resparams['num_inputs']
# Training
W_out, A, W_in, res_states = train_vanilla_RC(resparams, X[:resparams['train_length']])
# Prediction
prediction_steps = 50
predictions = rc.predict(W_out, A, W_in, res_states, prediction_steps, resparams)
actual = X[:, resparams['train_length']: resparams['train_length'] + prediction_steps]
dt = 0.05
t_pred = np.linspace(0, prediction_steps * dt - dt, prediction_steps)
t_pred /= lorenz.lyapunov_time  # scale to Lyapunov times

valid_time = prediction_analysis.valid_time(predictions, actual, t_pred)