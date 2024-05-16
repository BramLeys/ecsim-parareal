
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

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.semilogy(its, default_random, marker='o', color='blue', markersize = 10, label=r"Random ($\Delta t_G=10^{-1}, N_x=100$)")
plt.semilogy(its, random_time_fix, marker='^', color='red', markersize = 10, label=r"Random ($\Delta t_G=10^{-3}, N_x=100$)")
plt.semilogy(its, random_space_fix, marker='+', color='orange', markersize = 10, label=r"Random ($\Delta t_G=10^{-1}, N_x=10$)")
plt.semilogy(its, smooth, marker='p', color='green', markersize = 10, label=r"Smooth ($\Delta t_G=10^{-1}, N_x=100$)")
plt.semilogy(its, smooth_expensive, marker='s', color='black', markersize = 10, label=r"Smooth ($\Delta t_G=10^{-2}, N_x=1000$)")
plt.ylabel("State Change", fontsize = 16, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.yticks(color = "blue")
plt.xlabel("Iteration", fontsize = 16)
plt.legend(loc='best', bbox_to_anchor=(0.65, 0., 0.35, 0.35), framealpha=1)
# plt.legend(ncol = 3)
plt.title(r"Parareal convergence of CASE II using CN")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./CN_parareal_Convergence.png")

