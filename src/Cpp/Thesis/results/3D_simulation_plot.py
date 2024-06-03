import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import rc

rc('text', usetex=True)

### Read files
data = np.loadtxt("../../../../paper/Results/simulation_result3D.txt")
energy = np.loadtxt("../../../../paper/Results/simulation_energy3D.txt")

### Set parameters
Np = 10000
Nx = 128

print(data.shape)
print(energy.shape)

x = data[:Np,: ]%(2*np.pi)
v = data[Np:4*Np, :]
E = data[4*Np: 4*Np + 3*Nx,:]
B = data[4*Np + 3*Nx:,:]
print(B.shape)
print(E.shape)

NT = energy.shape[0]-1
dt = 0.5
ts = np.linspace(0, NT*dt, NT+1)

fig = plt.figure(figsize = (12, 5), dpi = 300)

# initial conditions
time_step = 0
### Velocity along Y
x_t = x[:,time_step].reshape(Np,1)
v_t = v[:, time_step].reshape(Np, 3)
negative = np.where(v_t[:, 1] < 0)[0]
positive = np.where(v_t[:, 1] >= 0 )[0]

### ========================================================================== ####

plt.scatter(x_t[negative], v_t[negative, 0], c = "red", marker = ".", label=r"$v_y < 0$")
plt.scatter(x_t[positive], v_t[positive, 0], c = "blue", marker = ".", label=r"$v_y \ge 0$")

plt.xlim(0, 2*np.pi)
plt.ylim(-0.016,0.016)
# plt.yticks([-0.10, -0.05, 0.00, 0.05, 0.10])

plt.xlabel("X (m)", fontsize = 25)
plt.ylabel("Velocity along X\n(m/s)", fontsize = 25)

plt.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 8)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)
plt.legend( fontsize  = 20, ncol=2)
fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/Sim_plots_3D_init.eps")
plt.savefig("./figures/png/Sim_plots_3D_init.png")

### ----------------------------------- ###
fig = plt.figure(figsize = (12, 5), dpi = 300)

# initial conditions
time_step = 1
### Velocity along Y
x_t = x[:,time_step].reshape(Np,1)
v_t = v[:, time_step].reshape(Np, 3)
negative = np.where(v_t[:, 1] < 0)[0]
positive = np.where(v_t[:, 1] >= 0 )[0]

### ========================================================================== ####

# plt.scatter(x_t, v_t, c = "blue", marker = ".")
plt.scatter(x_t[negative], v_t[negative, 0], c = "red", marker = ".", label=r"$v_y < 0$")
plt.scatter(x_t[positive], v_t[positive, 0], c = "blue", marker = ".", label=r"$v_y \ge 0$")

plt.xlim(0, 2*np.pi)
# plt.yticks([-0.10, -0.05, 0.00, 0.05, 0.10])

plt.xlabel("X (m)", fontsize = 25)
plt.ylabel("Velocity along X\n(m/s) ", fontsize = 25)

plt.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 8)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 8)
plt.legend( fontsize  = 20, ncol=2)
fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/Sim_plots_3D_end.eps")
plt.savefig("./figures/png/Sim_plots_3D_end.png")
### ----------------------------------- ###

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.semilogy(ts[:],abs(energy[:]-energy[0])/energy[0], "b-")

plt.xlabel("Simulation time (s)", fontsize = 25)
plt.ylabel("Error in energy", fontsize = 25)

plt.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 8)

plt.xlim(-0.2, NT*dt + 0.3)


### ----------------------------------- ###

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/Sim_plots_3D_energy.eps")
plt.savefig("./figures/png/Sim_plots_3D_energy.png")

### ========================================================================== ####