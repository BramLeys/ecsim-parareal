"""
Created on Fri Apr 26 13:55:25 2024

@author: PJD

Adjusted by Bram Leys
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import rc

rc('text', usetex=True)

### Read files
data = np.loadtxt("../../../../paper/Results/simulation_result1D.txt")
energy = np.loadtxt("../../../../paper/Results/simulation_energy1D.txt")

### Set parameters
Np = 10000
Nx = 5000

print(data.shape)
print(energy.shape)

x = data[:Np,: ]
v = data[Np:2*Np, :]
E = data[2*Np: 2*Np + 1*Nx,:]
B = data[2*Np + 1*Nx:,:]
print(B.shape)
print(E.shape)

NT = energy.shape[0]-1
dt = 0.125
ts = np.linspace(0, NT*dt, NT+1)

fig = plt.figure(figsize = (12, 10), dpi = 400)
plt.subplot(3, 1, 1)

# initial conditions
time_step = 0
### Velocity along Y
x_t = x[:,time_step].reshape(Np,1)
v_t = v[:, time_step].reshape(Np, 1)
# negative = np.where(v_t[:, 1] < 0)[0]
# positive = np.where(v_t[:, 1] >= 0 )[0]

### ========================================================================== ####

plt.scatter(x_t, v_t, c = "blue", marker = ".")
# plt.scatter(x_t[negative], v_t[negative, 0], c = "red", marker = ".")

plt.xlim(0, 2*np.pi)
# plt.yticks([-0.10, -0.05, 0.00, 0.05, 0.10])

plt.title("Phase space at t = 0s", fontsize = 30)
plt.xlabel("X (m)", fontsize = 25)
plt.ylabel("Velocity \n(m/s)", fontsize = 25)

plt.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 8)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)

### ----------------------------------- ###
plt.subplot(3, 1, 2)

# initial conditions
time_step = 1
### Velocity along Y
x_t = x[:,time_step].reshape(Np,1)
v_t = v[:, time_step].reshape(Np, 1)
# negative = np.where(v_t[:, 1] < 0)[0]
# positive = np.where(v_t[:, 1] >= 0 )[0]

### ========================================================================== ####

plt.scatter(x_t, v_t, c = "blue", marker = ".")
# plt.scatter(x_t[positive], v_t[positive, 0], c = "blue", marker = ".")
# plt.scatter(x_t[negative], v_t[negative, 0], c = "red", marker = ".")

plt.xlim(0, 2*np.pi)
# plt.yticks([-0.10, -0.05, 0.00, 0.05, 0.10])

plt.title(f"Phase space at t = {dt*NT}s", fontsize = 30)
plt.xlabel("X (m)", fontsize = 25)
plt.ylabel("Velocity \n(m/s) ", fontsize = 25)

plt.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 8)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)
### ----------------------------------- ###

plt.subplot(3, 1, 3)

plt.semilogy(ts[:],abs(energy[:]-energy[0])/energy[0], "b-")

plt.title(r"\textit{Exact} energy conservation", fontsize = 30)
plt.xlabel("Simulation time (s)", fontsize = 25)
plt.ylabel("Error in energy", fontsize = 25)

plt.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 8)
plt.tick_params(axis = 'y', which = 'minor', labelsize = 20, length = 4)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)

plt.xlim(-0.2, NT*dt + 0.3)


### ----------------------------------- ###

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/Sim_plots_1D.eps")
plt.savefig("./figures/png/Sim_plots_1D.png")

### ========================================================================== ####