"""
Created on Fri Apr 26 13:55:25 2024

@author: PJD
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import rc
import os
dir = os.path.dirname(os.path.abspath(__file__))
rc('text', usetex=True)

### Read files
data = np.loadtxt(os.path.join(dir,"./ECSIM_simulation.txt"))
energy = np.loadtxt(os.path.join(dir,"./ECSIM_energy.txt"))

### Set parameters
Np = 10000
Nx = 128

x = data[:Np,: ]
v = data[Np:4*Np, :]
E = data[4*Np: 4*Np + 3*Nx,:]
B = data[4*Np + 3*Nx:,:]
print(B.shape)
print(E.shape)

NT = x.shape[1]-1
dt = 0.125
ts = np.linspace(0, NT*dt, NT+1)

### Plot data at this time step
time_step = 250

### Velocity along Y
x_t = x[:,time_step].reshape(Np,1)
v_t = v[:, time_step].reshape(Np, 3)
negative = np.where(v_t[:, 1] < 0)[0]
positive = np.where(v_t[:, 1] >= 0 )[0]

### ========================================================================== ####

### Plots

fig = plt.figure(figsize = (10.4, 6.5), dpi = 300)

### ----------------------------------- ###

plt.subplot(2, 1, 1)

plt.scatter(x_t[positive], v_t[positive, 0], c = "blue", marker = ".")
plt.scatter(x_t[negative], v_t[negative, 0], c = "red", marker = ".")

plt.xlim(0, 2*np.pi)
plt.yticks([-0.10, -0.05, 0.00, 0.05, 0.10])

plt.title("Phase Space", fontsize = 24)
plt.xlabel("X (m)", fontsize = 22)
plt.ylabel("Velocity along X\n(m/s) ", fontsize = 22)

plt.tick_params(axis = 'x', which = 'major', labelsize = 18, length = 8)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)

### ----------------------------------- ###

plt.subplot(2, 1, 2)

plt.semilogy(ts[:time_step],abs(energy[:time_step]-energy[0])/energy[0], "b-")

plt.title(r"\textit{Exact} energy conservation", fontsize = 24)
plt.xlabel("Simulation time (s)", fontsize = 22)
plt.ylabel("Error in energy", fontsize = 22)

plt.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 8)
plt.tick_params(axis = 'y', which = 'minor', labelsize = 10, length = 4)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)

plt.xlim(-0.2, 31.3)


### ----------------------------------- ###

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./Sim_plots_1.png")

### ========================================================================== ####