import generate_reservoir
import ks_integration
import rc_single_output_weight as rc
import prediction_analysis
import numpy as np
import matplotlib.pyplot as plt
import time

#from ks_integration import periodicity_length, num_grid_points, time_step

#from memory_profiler import profile
import predict_single_output_layer


#@profile
def train_parallel_rc(
        nodes_num: int,
        overlap: int,
        input_data: np.ndarray,
        resparams: dict,
        batch_len: int = 1000,
        discard_length: int = 1000,
        square_nodes: bool = True
        ):

    num_inputs = resparams['num_inputs']
    degree = resparams['degree']
    radius = resparams['radius']
    sigma = resparams['sigma']
    train_length = resparams['train_length']
    inputs_per_reservoir = resparams["inputs_per_reservoir"]

    #print(num_inputs / inputs_per_reservoir)

    '''
    for i in range(int(num_inputs / inputs_per_reservoir)):
    # flag that determines if reservoirs are identical or not. If diff, seed=i, if not, seed=k for fixed k
    # if same reservoir for all could just make shallow copies.
    # all input weight layers should be the same as well. can use shallow copies
    # just need one res one input weight then attach diff input ranges to have the diff spatial things

        reservoirs.append(generate_reservoir.erdos_renyi_reservoir(size=nodes_num, degree=degree, radius=radius, seed=i))
        in_weights.append(generate_reservoir.generate_W_in(num_inputs=inputs_per_reservoir+2*overlap, res_size=nodes_num, sigma=sigma))
    '''
    reservoir = generate_reservoir.erdos_renyi_reservoir(
        size = nodes_num,
        degree = degree,
        radius = radius,
        seed = 1
        )
    in_weight = generate_reservoir.generate_W_in(
        num_inputs = inputs_per_reservoir + 2 * overlap,
        res_size = nodes_num,
        sigma = sigma
        )
    W_out, reservoirs_states = rc.fit_output_weight(resparams=resparams, input=input_data, reservoir=reservoir, W_in=in_weight, discard=discard_length, batch_len=batch_len, square_nodes=square_nodes)
    # for i in range(num_inputs // inputs_per_reservoir):
    #     # i think that this could be done with multiprocessing. the fit output weights only differs by the parameter i
    #     # prob try to use multiprocessing.Pool here
    #     loc = chop_data(input_data, inputs_per_reservoir, overlap, i)
    #     print(f'Loc shape: {loc.shape}')
    #     # print(loc)
    #     '''
    #     reservoirs_states.append(rc.reservoir_layer(reservoirs[i], in_weights[i],loc,resparams))
    #     out_weights.append(rc.fit_output_weights(resparams, reservoirs_states[i], loc))
    #     '''
    #     #reservoirs_states.append(rc.reservoir_layer(reservoir, in_weight, loc, resparams))
    #     VS_T, SS_T, res_state = rc.fit_output_weights(
    #         resparams = resparams,
    #         W_in = in_weight,
    #         reservoir = reservoir,
    #         data = loc[overlap: -overlap, :train_length],
    #         batch_len = batch_len,
    #         loc = loc,
    #         discard = discard_length,
    #         square_nodes = square_nodes
    #         )
    #     print(f'Training stage {(i + 1) / (num_inputs // inputs_per_reservoir) * 100} % done')
    #     reservoirs_states.append(res_state)


        # out_weights.append(rc.fit_output_weights(resparams=resparams, W_in=in_weight, reservoir=reservoir, data=loc[overlap:inputs_per_reservoir+overlap,discard_length:train_length], batch_len=batch_len))

    return W_out, in_weight, reservoirs_states, reservoir


def chop_data(data, n, m, step):
    index = np.arange(data.shape[0])
    # do this so it doesnt hafta query data. prolly should just have the parameter as num_inputs
    return data[np.roll(index, - n * step + m)[0: n + 2 * m], :].copy() # added .copy() hopefully to improve if there was mem leak


square_nodes = False
time_step = 0.25
prediction_steps = 1000
transient_length = 500
num_grid_points = 256
periodicity_length = 100
num_reservoirs = 16
p = [0.6, 3, 0.1] #1] # spectral radius, degree, input scaling
resparams = {
    'N': 1000, # 6.08497359577532 at 1500 w degree 3, 5.0048007681229 at degree 10
    # for some reason sometimes increasing the num of nodes decreases accuracy. weird.
    'num_inputs': num_grid_points,
    'radius': p[0],
    'degree': p[1],
    'nonlinear_func': np.tanh,
    'sigma': p[2],
    'train_length': 35000, # when time was at 9000 in the ss it got to like 10 lambdas. with 15000 its only getting 5.28. why? with 29000 it gets 5.928
    'beta': 0.0001,
    'bias': 1.3, #0
    'overlap': 6, #20, #10 # nates thought with issues arising from overlap being too big is prob right. what is correlation function between points
    'inputs_per_reservoir' : 16
}
resparams["num_inputs"] = num_grid_points

X = ks_integration.int_plot(
    time_range=73000,
    periodicity_length=periodicity_length,
    num_grid_points=num_grid_points,
    time_step=time_step,
    plot=False
    )
#X = np.load('X_seed0_L200_Q512_T100000_NOWWORKING.npy')
#X = np.load('X_seed4_L22_Q64_T80000_NOWWORKING.npy')
# X = ks_integration.int_plot(plot=False,IC_seed=0)
start = time.time()
print(X.shape)
out_weight, in_weight, reservoirs_states, reservoir = train_parallel_rc(
    resparams['N'],
    resparams['overlap'],
    X,
    resparams,
    resparams['inputs_per_reservoir'],
    discard_length = transient_length,
    square_nodes = square_nodes
    )
print(f"out weight shape: {out_weight.shape}")
end = time.time()

"""
Conducting a test where instead of using the generated single output layer, 
I use ones that give good VPTS from parallel_2
"""

test_out_weight = np.load("/Users/grennongurney/PycharmProjects/pythonProject16/Parallel_2/out_weights_XSEED11000_1000_B1.3_sig0.1_trans500_TL35000_IPR16.npy")
valid_times = list()
for i in range(8):
    predictions = predict_single_output_layer.parallel_predict(
        out_weight = test_out_weight[i],
        reservoir = reservoir,
        in_weight = in_weight,
        final_res_states = reservoirs_states,
        time_steps = prediction_steps,
        resparams = resparams,
        square_nodes = square_nodes
        )
    actual = X[:, resparams['train_length']: resparams['train_length'] + prediction_steps]
    print(f'Predict shape :{predictions.shape}')
    print(f'X shape: {X.shape}')
    t_pred = np.linspace(0, prediction_steps * time_step - time_step, prediction_steps)
    t_pred /= 11.11  # Lyapunov time for L=200

    real = actual

#     valid_time = prediction_analysis.valid_time(predictions, real, t_pred)
#     valid_times.append(valid_time)
#     print(f'Valid time for weight layer {i}: {valid_time}')
# valid_times_Array = np.array(valid_times)
# np.save("valid_times_arrayHARDER", valid_times_Array)

    fig, ax = plt.subplots(constrained_layout = True)
    ax.set_title(f"Kursiv_Overlay")
    # x = np.arange((predictions-real).shape[1]) * time_step / 20.83
    x = np.arange((predictions-real).shape[1]) * time_step / 11.11
    y = np.arange((predictions-real).shape[0]) * periodicity_length / num_grid_points
    x, y = np.meshgrid(x, y)
    # pcm = ax.pcolormesh(x, y, np.abs(predictions-real))
    pcm = ax.pcolormesh(x, y, predictions-real)
    ax.set_ylabel("$x$")
    ax.set_xlabel("$t$")
    fig.colorbar(pcm, ax = ax, label = "$overlay(x, t)$")
    plt.savefig(f"Images/Test_Graphs/num{i}overlay_{resparams['N']}_B{resparams['bias']}_sig{resparams['sigma']}_trans{transient_length}.png")


    print(f'real shape {real.shape}')
    print(f'predict shape {predictions.shape}')
    fig, ax = plt.subplots(constrained_layout = True)
    ax.set_title(f"Kursiv_Actual")
    # x = np.arange(real.shape[1]) * time_step / 20.83
    x = np.arange(real.shape[1]) * time_step / 11.11
    # x = np.arange(real.shape[1]) * time_step
    y = np.arange(real.shape[0]) * periodicity_length / num_grid_points
    x, y = np.meshgrid(x, y)
    pcm = ax.pcolormesh(x, y, real)
    ax.set_ylabel("$x$")
    ax.set_xlabel("$t$")
    fig.colorbar(pcm, ax = ax, label = "$u(x, t)$")
    plt.savefig(f"Images/Test_Graphs/num{i}actual_{resparams['N']}_B{resparams['bias']}_sig{resparams['sigma']}_trans{transient_length}.png")
    fig, ax = plt.subplots(constrained_layout = True)
    ax.set_title(f"Kursiv_Predict")
    # x = np.arange(predictions.shape[1]) * time_step / 20.83
    x = np.arange(predictions.shape[1]) * time_step / 11.11
    # x = np.arange(predictions.shape[1]) * time_step

    y = np.arange(predictions.shape[0]) * periodicity_length / num_grid_points
    x, y = np.meshgrid(x, y)
    pcm = ax.pcolormesh(x, y, predictions)
    ax.set_ylabel("$x$")
    ax.set_xlabel("$t$")
    fig.colorbar(pcm, ax = ax, label = "$pred(x, t)$")
    plt.savefig(f"Images/Test_Graphs/num{i}predict_{resparams['N']}_B{resparams['bias']}_sig{resparams['sigma']}_trans{transient_length}.png")

print(f'Time for {resparams["N"]}: {end - start}')
