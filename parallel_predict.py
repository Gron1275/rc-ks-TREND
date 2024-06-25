import numpy as np


def predict(W_out, A, W_in, training_res_states, time_steps, resparams):
    #predict_length parameter
    """Closed loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    final_res_state = training_res_states[:, -1]

    predictions = np.zeros((W_out.shape[0], time_steps))
    rt = final_res_state

    for i in range(time_steps):
        predictions[:, i] = W_out @ rt
        rt = g(A @ rt + W_in @ predictions[:, i] + bias)

    return predictions


def chop_data(data, n, m, step):
    index = np.arange(data.shape[0])
    # do this so it doesnt hafta query data. prolly should just have the parameter as num_inputs
    return data[np.roll(index, -n*step + m)[0:n+2*m], :]


def parallel_predict(out_weights, reservoir, in_weight, training_res_states, time_steps, resparams, alpha=1):
    g = resparams['nonlinear_func']
    bias = resparams['bias']

    final_res_states = list()
    # create a (num_reservoirs X Out weights X time step) matrix
    predictions_par = np.zeros((len(out_weights), out_weights[0].shape[0], time_steps))

    for res_state in training_res_states:
        final_res_states.append(res_state[:, -1])

    for dt in range(time_steps):
        for i in range(len(out_weights)):
            # print(f'out_weights dim: {out_weights[i].shape}')
            # print(f'in_weight dim: {in_weight.shape}')
            # print(f'predictions_par dim: {predictions_par[i].shape}')
            # print(f'FULL predictions_par dim: {predictions_par.shape}')
            # print(f'final_res_states dim: {final_res_states[i].shape}')
            # print(f'reservoir dim: {reservoir.shape}')
            predictions_par[i, :, dt] = out_weights[i] @ final_res_states[i]
        whole_preds = predictions_par.copy()
        whole_preds = whole_preds.reshape((resparams['num_inputs'],time_steps))
        for i in range(len(out_weights)):
            # final_res_states[i] = (1 - alpha) * final_res_states[i] + alpha * g(reservoir @ final_res_states[i] + in_weight @ predictions_par[i, :, dt] + bias)
            final_res_states[i] = (1 - alpha) * final_res_states[i] + alpha * g(reservoir @ final_res_states[i] + in_weight @ chop_data(whole_preds, resparams['inputs_per_reservoir'], resparams['overlap'], i)[:,dt] + bias)

            # i dont think this code is doing any overlapping after the initial data thats coming in
        #for i in range(len(out_weights)):

    return whole_preds
