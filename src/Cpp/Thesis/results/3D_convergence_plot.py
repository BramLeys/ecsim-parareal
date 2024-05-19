
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc

# rc('text', usetex=True)

# Data loading and selection
filename = "../../../../paper/Results/ecsim3d_convergence_errors.txt"
data = np.loadtxt(filename)

ts = data[:,0]
err_x = data[:,1]
err_v = data[:,2]
err_E = data[:,3]
err_B = data[:,4]

max_err = np.max(data[:,1:])

# guides
second_order = ts**2*1000
first_order = ts/100

# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.loglog(ts, err_x, marker='o', color='blue', markersize = 10, label="Position")
plt.loglog(ts, err_v, marker='^', color='red', markersize = 10, label="Velocity")
plt.loglog(ts, err_E, marker='H', color='green', markersize = 15, label="Electric field")
plt.loglog(ts, err_B, marker='s', color='orange', markersize = 10, label="Magnetic field")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta t^2)$")
plt.loglog(ts, first_order, linestyle="dashed", color="brown",markersize = 10, label =r"$\mathcal{O}(\Delta t)$")
plt.ylabel("Error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Timestep size", fontsize = 20)
# fig.legend(bbox_to_anchor = (0.95, 0.999), ncol = 3, prop = {'size': 18})
plt.legend(fontsize=20,loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"Convergence of 1D3V using ECSIM", fontsize=25)

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/ECSIM3DConvergence.eps")
plt.savefig("./figures/png/ECSIM3DConvergence.png")

