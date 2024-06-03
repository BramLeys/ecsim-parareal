
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


# Data loading and selection
filename = "../../../../paper/Results/3d_parareal_convergence.txt"
data = np.loadtxt(filename)

its = data[:,0]
bad_conv = data[:,1]
good_conv = data[:,2]


# Plots

fig = plt.figure(figsize = (8,6), dpi = 300)

plt.semilogy(its, bad_conv, marker='o', color='blue', markersize = 10, label=r"$N_x = 5000$")
plt.semilogy(its, good_conv, marker='^', color='red', markersize = 10, label=r"$N_x = 512$")
plt.ylabel("Parareal error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel("Iteration", fontsize = 20)
plt.legend(loc='lower left',  framealpha=1, fontsize=20)
# plt.legend(ncol = 3)

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/3D_parareal_Convergence.eps")
plt.savefig("./figures/png/3D_parareal_Convergence.png")

