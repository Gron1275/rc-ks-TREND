import ks_integration
import generate_reservoir
import rc
import prediction_analysis
import numpy as np
import h5py


def train_vanilla_RC(resparams, data):
    """TODO"""
    N = resparams['N']
    radius = resparams['radius']
    degree = resparams['degree']
    num_inputs = resparams['num_inputs']
    sigma = resparams['sigma']
    train_length = resparams['train_length']

    A = generate_reservoir.erdos_renyi_reservoir(N, degree, radius,0)
    W_in = generate_reservoir.generate_W_in(num_inputs, N, sigma)

    res_states = rc.reservoir_layer(A, W_in, data, resparams)

    W_out = rc.fit_output_weights(resparams, res_states, data[:, :train_length])
    return W_out, A, W_in, res_states
'''In the Pathak paper, the hyperparameters (minus a few details) are: 
    reservoir size 5000 nodes, 70000 training time-steps (if this is taking ages, try fewer - but still 10000 plus),
    spectral radius of .6, input coupling strength of one, bias strength of zero, 
    leakage of one, and regularization strength of 10^-4.'''
# Specify reservoir parameters
p = [0.6, 10, 0.1] # spectral radius, degree, input scaling
approx_res_size = 1000
resparams = {
    'num_inputs': 512,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 9000,
    'beta': 0.0001,
    'bias': 1.1
}
resparams['N'] = int(np.floor(approx_res_size / resparams['num_inputs'])) * resparams['num_inputs']

# X = ks_integration.int_plot(False)
X = np.load("X_seed11000.npy")

W_out, A, W_in, res_states = train_vanilla_RC(resparams, X[:, :resparams['train_length']])

# Prediction
dt = 0.25
prediction_steps = 1000
predictions = rc.predict(W_out, A, W_in, res_states, prediction_steps, resparams)
actual = X[:, resparams['train_length']: resparams['train_length'] + prediction_steps]

t_pred = np.linspace(0, prediction_steps * dt - dt, prediction_steps)
t_pred /= 20.83 # Lyapunov time for L=22

valid_time = prediction_analysis.valid_time(predictions, actual, t_pred)

np.savez("test", actual=actual, predictions=predictions)
