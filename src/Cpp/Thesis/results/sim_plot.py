import numpy as np
import math
import matplotlib.pyplot as plt

plt.ion()
file = "./results/ECSIM_simulation.txt"
data = np.loadtxt(file)
file = "./results/ECSIM_energy.txt"
energy = np.loadtxt(file)

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
ts = np.linspace(0,NT*dt, NT+1)

particle = Np//2
grid_point = Nx//2
fig, axs = plt.subplots(2, 1)
# line_neg, =axs[0].plot([],[],"r.",label='v.x < 0')
# line_pos, =axs[0].plot([],[],"b.",label='v.x > 0')
line_pos, = axs[0].plot([],[],"r.",label='v.y > 0')
line_neg, = axs[0].plot([],[],"b.",label='v.y < 0')
axs[0].set_xlabel('position on x-axis (m)', fontsize = 24)
axs[0].set_ylabel('velocity along x-axis (m/s)', fontsize = 24)
axs[0].legend(fontsize='large')
# axs[1].set_xlabel('x')
# axs[1].set_ylabel('v.x')
axs[0].set_xlim(0, 2*math.pi)

line_energ, = axs[1].semilogy([],[])
axs[1].set_xlabel("time (s)", fontsize = 24)
axs[1].set_ylabel(r"$\frac{|E(t)-E(0)|}{E(0)}$", fontsize = 24)
# axs[1].set_xlim(0, 2*math.pi)
# axs[1].legend()
for time_step in [250]:
# time_step = 500
    v_t = v[:,time_step].reshape(Np,3)
    x_t = x[:,time_step].reshape(Np,1)

    negative = np.where(v_t[:,0] < 0)[0]
    positive = np.where(v_t[:,0] >= 0 )[0]
    # line_neg.set_data(x_t[negative],v_t[negative,1])
    # line_pos.set_data(x_t[positive],v_t[positive,1])
    # axs[0].relim()
    # axs[0].autoscale_view(None,False,True)
    # axs[0].legend()

    negative = np.where(v_t[:,1] < 0)[0]
    positive = np.where(v_t[:,1] >= 0 )[0]
    line_neg.set_data(x_t[negative],v_t[negative,0])
    line_pos.set_data(x_t[positive],v_t[positive,0])
    axs[0].relim()
    axs[0].autoscale_view(None,False,True)
    axs[0].set_title("Phase space", fontsize = 24)

    line_energ.set_data(ts[:time_step],abs(energy[:time_step]-energy[0])/energy[0])
    axs[1].relim()
    axs[1].autoscale_view(None,True,True)
    axs[1].set_title("Energy conservation", fontsize = 24)
    # plt.suptitle(f"t = {time_step*dt}", ha='center', fontsize = 32)
    plt.tight_layout()
    plt.pause(0.01)
    plt.draw()
plt.ioff()
plt.show()


# axs[0].plot(ts,x[particle,:], markersize = 10)
# axs[0].set_title("Position")
# axs[0].set_ylabel("x (m)", fontsize = 16)
# axs[0].set_xlabel("t (s)", fontsize = 16)

# axs[1].plot(ts,v[3*particle,:], markersize = 10)
# axs[1].set_title("Velocity")
# axs[1].set_ylabel("v.x (m/s)", fontsize = 16)
# axs[1].set_xlabel("t (s)", fontsize = 16)

# axs[2].plot(ts,E[3*grid_point,:], markersize = 10)
# axs[2].set_title("Electric field")
# axs[2].set_ylabel("E.x (V/m)", fontsize = 16)
# axs[2].set_xlabel("t (s)", fontsize = 16)

# axs[3].plot(ts,B[3*grid_point + 1,:], markersize = 10)
# axs[3].set_title("Magnetic field")
# axs[3].set_ylabel("B.y (T)", fontsize = 16)
# axs[3].set_xlabel("t (s)", fontsize = 16)

# Show the plot
# plt.xlabel(r"$\frac{\Delta t_\mathrm{Coarse}}{\Delta t_\mathrm{Fine}}$", fontsize = 16)