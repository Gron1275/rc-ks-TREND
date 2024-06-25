import generate_reservoir
import rc
import prediction_analysis
import numpy as np
# def train_parallel_rc(reservoir_nums: int, nodes_num: int, overlap: int, input_data: np.ndarray, resparams: dict, inputs_per_reservoir=None):

def train_parallel_rc(nodes_num: int, overlap: int, input_data: np.ndarray, resparams: dict, inputs_per_reservoir: int):

    reservoirs = list()
    in_weights = list()
    out_weights = list()
    num_inputs = resparams['num_inputs']
    degree = resparams['degree']
    radius = resparams['radius']
    sigma = resparams['sigma']
    reservoirs_states = list()
    # i dont think i can arbitrarily define the number of reservoirs. I think what will depend on how many inputs I want per reservoir and
    # the number of input data points. In fact, I think it should be a function of those two things
    # num_inputs/inputs_per_reservoir: this will be the number of reservoirs with output layer equal to nodes_num. The input number would be nodes_num + 2*overlap
    print(num_inputs / inputs_per_reservoir)
    for i in range(int(num_inputs / inputs_per_reservoir)):
        # reservoirs.append(generate_reservoir.erdos_renyi_reservoir(size=nodes_num+overlap, degree=degree, radius=radius, seed=i))
        reservoirs.append(generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=i))
        in_weights.append(generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir+2*overlap, res_size=nodes_num, sigma=sigma))
        # in_weights.append(generate_reservoir.generate_W_in(num_inputs=num_inputs+overlap, res_size=nodes_num+overlap, sigma=sigma))
    for i in range(int(num_inputs / inputs_per_reservoir)):
        # print(chop_data(input_data, inputs_per_reservoir, overlap, i).shape)
        # print(in_weights[i] @ chop_data(input_data, inputs_per_reservoir, overlap, i))
        loc = chop_data(input_data, inputs_per_reservoir, overlap, i)
        print(loc.shape)
        print(in_weights[i].shape)
        print((in_weights[i] @ loc).shape)
        reservoirs_states.append(rc.reservoir_layer(reservoirs[i], in_weights[i],loc,resparams))
        out_weights.append(rc.fit_output_weights(resparams, reservoirs_states[i],loc))
    return out_weights, reservoirs, in_weights, reservoirs_states
    # for taking the output data, maybe just roll back the same way I do but compensate for the left padding then only
    # index to [0:n]


    # reservoirs_states.appent(rc.reservoir_layer()) maybe put this in a different for loop so you dont hafta regenerate everything
# run the above for loop or perhaps with ndarrays find some way to vectorize and then once all that stuff is generated then do the
# reservoir_layer() step
g = np.arange(1280).reshape(64,20)
# in this case, i = m, i think. and [0:4] is n + m?
# 5 i think is input_len / n
# for i in range(5):
#     print(g[:,np.roll(np.arange(10),1-2*i)[0:4]])
#     print()

def chop_data(data, n, m, step):
    index = np.arange(data.shape[0])
    return data[np.roll(index,-n*step + m)[0:n+2*m],:]
    # for i in range(int(data.shape[0] / n)):
    #     print(data[:, np.roll(fart,-n*i+m)[0:n+2*m]])
    #     print()

#make_da_pizza(g, 2,1)
#chop_data(g, 2, 1)
# will need to find out the range(?) function. That needs to be better or it just needs to be stated that n needs to divide input size
# i think n has to divide data.shape
# the issue seems to be coming from the index
# 2 - 2*m*i works for n = 4, m = 1
p = [0.6, 10, 0.1] # spectral radius, degree, input scaling
approx_res_size = 100
resparams = {
    'num_inputs': 64,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 20,
    'beta': 0.0001,
    'bias': 1.2
}
# resparams['N'] = int(np.floor(approx_res_size / resparams['num_inputs'])) * resparams['num_inputs']
resparams['N'] = 1000
train_parallel_rc(1000, 5, g, resparams, 16)
# need to transpose g and then prolly its needed to edit the chopping function. For now I will just transpose at the reservoir layer stage