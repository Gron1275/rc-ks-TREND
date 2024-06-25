# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 02:16:31 2024

@author: decla
"""

import numba
import ks_etdrk4 as ks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
num_grid_points = 512
time_step = .25
periodicity_length = 200
time_range = 70000
IC_bounds = [-.6, .6]
L_22_Lyap_Time = 20.83
IC_seed = 11000
params = np.array([[], []], dtype = np.complex128)

font_size = 15.


def int_plot(plot=True, IC_seed = IC_seed):
    u_arr, new_params = ks.kursiv_predict(
        u0 = np.random.default_rng(IC_seed).uniform(IC_bounds[0], IC_bounds[1], size = num_grid_points),
        N = num_grid_points,
        tau = time_step,
        T = time_range,
        params = params,
        noise = np.zeros((1, 1), dtype = np.double)
        )
    mean = np.mean(u_arr,axis=1)
    stdev = np.std(u_arr,axis=1)
    u_arr = (u_arr - mean.reshape((-1,1))) / stdev.reshape((-1,1))
    if plot:
        with mpl.rc_context({"font.size" : font_size}):
            fig, ax = plt.subplots(constrained_layout = True)
            ax.set_title("Kursiv_Predict")
            x = np.arange(u_arr.shape[1]) * time_step
            y = np.arange(u_arr.shape[0]) * periodicity_length / num_grid_points
            x, y = np.meshgrid(x, y)
            pcm = ax.pcolormesh(x, y, u_arr)
            ax.set_ylabel("$x$")
            ax.set_xlabel("$t$")
            fig.colorbar(pcm, ax = ax, label = "$u(x, t)$")
            plt.show()
    return u_arr
