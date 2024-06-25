
import matplotlib.pyplot as plt
import h5py
import generate_reservoir
import rc
import prediction_analysis
import ks_etdrk4
import ks_integration
from ks_integration import periodicity_length, num_grid_points, time_step
import numpy as np
# def train_vanilla_RC(resparams, data, transient_length):
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

    # W_out = rc.fit_output_weights(resparams, res_states[:, transient_length:], data[:, transient_length:train_length])
    W_out = rc.fit_output_weights(resparams, res_states, data[:, :train_length])
    return W_out, A, W_in, res_states
'''In the Pathak paper, the hyperparameters (minus a few details) are: 
    reservoir size 5000 nodes, 70000 training time-steps (if this is taking ages, try fewer - but still 10000 plus),
    spectral radius of .6, input coupling strength of one, bias strength of zero, 
    leakage of one, and regularization strength of 10^-4.'''
# Specify reservoir parameters
p = [0.6, 10, 0.1] # spectral radius, degree, input scaling
approx_res_size = 7000
resparams = {
    'num_inputs': 64,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 69000,
    'beta': 0.0001,
    'bias': 1.1
}
resparams['N'] = int(np.floor(approx_res_size / resparams['num_inputs'])) * resparams['num_inputs']

X = ks_integration.int_plot(False)

# Training
# W_out, A, W_in, res_states = train_vanilla_RC(resparams, X[:, :resparams['train_length']],1000)
W_out, A, W_in, res_states = train_vanilla_RC(resparams, X[:, :resparams['train_length']])
print(X[:, :resparams['train_length']].shape)
print(f'W in shape {W_in.shape}')
print(f'X shape {X.shape}')

# Prediction
dt = 0.25
prediction_steps = 1000
predictions = rc.predict(W_out, A, W_in, res_states, prediction_steps, resparams)
actual = X[:, resparams['train_length']: resparams['train_length'] + prediction_steps]
# real = X[:, 1000:-1]
real = actual
# t_pred = np.linspace(0, prediction_steps * dt - dt, prediction_steps)
# t_pred = np.linspace(0, resparams['train_length'] * dt - dt, resparams['train_length'])
t_pred = np.linspace(0, prediction_steps * dt - dt, prediction_steps)
t_pred /= 20.83 # Lyapunov time for L=22

valid_time = prediction_analysis.valid_time(predictions, real, t_pred)
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
