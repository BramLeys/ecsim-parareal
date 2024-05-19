
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


# Data loading and selection
filename = "../../../../paper/Results/tol_parareal_check.txt"
data = np.loadtxt(filename)

its = data[:,0]
sim_level_est = data[:,1]
sim_level_act = data[:,2]
step_level_est = data[:,3]
step_level_act = data[:,4]


# Plots

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12, 5), dpi = 300, sharex=True)
fig.subplots_adjust(top=0.8) 
ax1.semilogy(its, sim_level_est, marker='o', color='blue', markersize = 15, label=r"Simulation level tolerance")
ax1.semilogy(its, step_level_est, marker='^', color='red', markersize = 10, label=r"Timestep level tolerance")
ax1.set_ylabel("Parareal error", fontsize = 20)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax1.set_xticks(range(1,its.shape[0]+1))
ax1.set_xlabel("Iteration", fontsize = 20)
ax1.set_title(r"Error during parareal",fontsize = 25)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height*1.2])

ax2.semilogy(its, sim_level_act, marker='o', color='blue', markersize = 15)
ax2.semilogy(its, step_level_act, marker='^', color='red', markersize = 10)
ax2.set_ylabel("Error", fontsize = 20)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax2.set_xticks(range(1,its.shape[0]+1))
ax2.set_xlabel("Iteration", fontsize = 20)
ax2.set_title(r"Error w.r.t serial",fontsize = 25)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width, box.height*1.2])
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=20)


# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/Tolerance_parareal_check.eps")
plt.savefig("./figures/png/Tolerance_parareal_check.png")

