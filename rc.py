import numpy as np


def reservoir_layer(A, W_in, input, resparams):
    """TODO"""
    N = resparams['N']
    train_length = resparams['train_length']
    g = resparams['nonlinear_func']
    bias = resparams['bias']

    # Open loop
    res_states = np.zeros((N, train_length))
    for i in range(train_length - 1):
        res_states[:, i + 1] = g(A @ res_states[:, i] + W_in @ input[:, i] + bias)
    return res_states
    # return np.concatenate((res_states, res_states**2), axis = 0)


def fit_output_weights(resparams, res_states, data, discard=0):
    """TODO"""
    beta = resparams['beta']
    N = resparams['N']

    # Tikhonov ridge regression using normal equations
    W_out = data @ res_states.T @ np.linalg.pinv(res_states @ res_states.T + beta * np.identity(N))
    return W_out


def predict(W_out, A, W_in, training_res_states, time_steps, resparams):
    #predict_length parameter
    """Closed loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    final_res_state = training_res_states[:, -1]
    # final_res_state = training_res_states[:, 1000]

    # predictions = np.zeros((W_out.shape[0], resparams['train_length']-1000))
    predictions = np.zeros((W_out.shape[0], time_steps))
    rt = final_res_state
    # for i in range(resparams['train_length']-1000):
    #     predictions[:, i] = W_out @ rt
    #     rt = g(A @ rt + W_in @ predictions[:, i] + bias)
    for i in range(time_steps):
        predictions[:, i] = W_out @ rt
        rt = g(A @ rt + W_in @ predictions[:, i] + bias)

    return predictions

def open_train_fit(W_out, A, W_in, training_res_states, time_steps, resparams,X):
    """Open loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    beginning_res_state = training_res_states[:,0]
    print(f'Beginning res shape{beginning_res_state.shape}')
    print(f'A shape {A.shape}')
    predictions = np.zeros((W_out.shape[0], resparams['train_length']))
    attempt = np.zeros((W_out.shape[0], resparams['train_length']))
    print(f'Attempt shape {attempt.shape}')
    rt = beginning_res_state
    for i in range(resparams['train_length']):
        predictions[:,i] = W_out @ rt
        attempt[:,i] = W_out @ g(A @ rt + W_in @ predictions[:,i] + bias)
        rt = training_res_states[:,i + 1]
    return predictions, attempt

def open_predict(W_out, A, W_in, input_signal, initial_state, time_steps, resparams,X):
    """Open loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    beginning_res_state = initial_state
    print(f'Beginning res shape{beginning_res_state.shape}')
    print(f'A shape {A.shape}')
    predictions = np.zeros((W_out.shape[0], resparams['train_length']))
    attempt = np.zeros((W_out.shape[0], resparams['train_length']))
    print(f'Attempt shape {attempt.shape}')
    rt = beginning_res_state
    for i in range(resparams['train_length']-1):
        predictions[:,i] = W_out @ rt
        rt = g(A @ rt + W_in @ input_signal[:, i] + bias)
        attempt[:,i] = W_out @ g(A @ rt + W_in @ predictions[:,i] + bias)
    return predictions, attempt
