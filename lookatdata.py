import matplotlib.pyplot as plt
import numpy as np

"""
fig, ax = plt.subplots(constrained_layout = True)
ax.set_title("Kursiv_Predict")
# x = np.arange(predictions.shape[1]) * time_step / 20.83
x = np.arange(predictions.shape[1]) * time_step
y = np.arange(predictions.shape[0]) * periodicity_length / num_grid_points
x, y = np.meshgrid(x, y)
pcm = ax.pcolormesh(x, y, predictions)
ax.set_ylabel("$x$")
ax.set_xlabel("$t$")
fig.colorbar(pcm, ax = ax, label = "$pred(x, t)$")
plt.show()
"""
predictions = np.load('Overlay_N2000_Q512_L200_T69000.npy')
print(predictions.shape)
print(predictions)
fig, ax = plt.subplots(constrained_layout = True)
ax.set_title("Kursiv_Predict")
print('here')
x = np.arange(predictions.shape[1]) * 0.25
y = np.arange(predictions.shape[0]) * (200 / 512)
x, y = np.meshgrid(x,y,predictions)
pcm = ax.pcolormesh(x, y, predictions)
ax.set_ylabel("$x$")
ax.set_xlabel("$t$")
fig.colorbar(pcm, ax = ax, label = "$pred(x, t)$")
plt.show()
