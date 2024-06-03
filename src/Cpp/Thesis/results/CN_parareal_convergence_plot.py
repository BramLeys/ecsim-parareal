
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


# Data loading and selection
filename = "../../../../paper/Results/cn_parareal_convergence.txt"
data = np.loadtxt(filename)

its = data[:,0]
default_random = data[:,1]
random_time_fix = data[:,2]
random_space_fix = data[:,3]
smooth = data[:,4]
smooth_expensive = data[:,5]


# Plots

fig, ax = plt.subplots(1,1,figsize = (15, 7), dpi = 300, sharex=True)

ax.semilogy(its, random_space_fix, marker='H', color='orange', markersize = 15, label=r"Random ($\Delta t_G=10^{-2}, N_x=10$)")
ax.semilogy(its, default_random, marker='o', color='blue', markersize = 10, label=r"Random ($\Delta t_G=10^{-2}, N_x=100$)")
ax.semilogy(its, random_time_fix, marker='^', color='red', markersize = 10, label=r"Random ($\Delta t_G=10^{-3}, N_x=100$)")
ax.semilogy(its, smooth, marker='>', color='green', markersize = 10, label=r"Smooth ($\Delta t_G=10^{-2}, N_x=100$)")
ax.semilogy(its, smooth_expensive, marker='s', color='black', markersize = 10, label=r"Smooth ($\Delta t_G=10^{-2}, N_x=1000$)")
# ax.hlines(10**(-8), 0, np.max(its), linestyle="dashed")
ax.set_ylabel("Parareal error", fontsize = 20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax.set_xlabel("Iteration", fontsize = 20)
# plt.legend(loc='best', bbox_to_anchor=(0.65, 0., 0.35, 0.35), framealpha=1,fontsize=20)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*1.2])
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fontsize=20)

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/CN_parareal_Convergence.eps")
plt.savefig("./figures/png/CN_parareal_Convergence.png")

