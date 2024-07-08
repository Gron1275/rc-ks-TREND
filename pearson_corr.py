import numpy as np
import matplotlib.pyplot as plt

X = np.load('X_seed0_L500_Q1024_T300000_NOWWORKING.npy')
# Y = np.load('X_seed1_L200_Q512_T300000.npy')
# Z = np.load('X_seed2_L200_Q512_T300000.npy')
# L 22
# two paper setups
# correlation as a function of distance between two points
# loop thru all points
"""
"""
Xcorrelation_matrix = np.corrcoef(X, X)
# Ycorrelation_matrix = np.corrcoef(Y, Y)
# Zcorrelation_matrix = np.corrcoef(Z, Z)
print('matrix made')
ran = 1024
Xmidpoint_line = Xcorrelation_matrix[int(ran/2), :ran]
# Ymidpoint_line = Ycorrelation_matrix[256, :512]
# Zmidpoint_line = Zcorrelation_matrix[256, :512]
plt.plot(np.arange(ran), Xmidpoint_line)
# plt.plot(np.arange(512), Ymidpoint_line)
# plt.plot(np.arange(512), Zmidpoint_line)
plt.plot(np.arange(ran),np.zeros((ran,)))
plt.title("Pearson Correlation vs. Length")
plt.xlabel("Q (grid points)")
plt.ylabel("Correlation strength")
plt.show()