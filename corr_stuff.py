import ks_integration
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
"""Caution must be applied when using cross correlation for nonlinear systems. 
In certain circumstances, which depend on the properties of the input, cross correlation 
between the input and output of a system with nonlinear dynamics can be completely blind to 
certain nonlinear effects.[14] This problem arises because some quadratic moments can equal zero 
and this can incorrectly suggest that there is little "correlation" (in the sense of statistical dependence) 
between two signals, when in fact the two signals are strongly related by nonlinear dynamics."""
# X = ks_integration.int_plot(False)
# Y = ks_integration.int_plot(False, 0)
# print(np.mean(np.correlate(X[0,:10000],X[3,:10000],"same")))

correlations = list()
other = list()
Yother = list()
shift = 0
grid_points = 64
time_range = 10000
#ks_integration.int_plot(True, 0)
init_conditions = list()
for i in range(1):
    init_conditions.append(ks_integration.int_plot(False, i))
glist = list()
for j in range(len(init_conditions)):
    temp = list()
    for i in range(grid_points):

    #correlations.append(np.mean(np.correlate(X[shift,:10000],X[(i + shift) % 64,:10000],"same")))

        temp.append(scp.correlate(init_conditions[j][int(grid_points/2),:time_range],init_conditions[j][(i + shift) % grid_points,:time_range],'valid')/(time_range))
    glist.append(temp)
        # Yother.append(np.correlate(Y[shift, :70000], Y[(i + shift) % 64, :70000]))
# plt.plot(np.arange(64),correlations)
print(glist[0])
# for i in range(len(init_conditions)):
#     plt.plot(np.arange(grid_points), glist[i][0:time_range])
# plt.plot(np.arange(grid_points), np.zeros((grid_points,)))
# plt.title("Correlation vs. Length")
# plt.xlabel("x")
# plt.ylabel("Correlation strength")
# plt.show()
"""Partial correlation
Main article: Partial correlation
If a population or data-set is characterized by more than two variables, a partial correlation coefficient measures the strength 
of dependence between a pair of variables that is not accounted for by the way in which they both change in response to variations 
in a selected subset of the other variables.
IMPORTANT!?!?
"""
print(__name__)