
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

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.semilogy(its, bad_conv, marker='o', color='blue', markersize = 10, label=r"$N_x = 5000$")
plt.semilogy(its, good_conv, marker='^', color='red', markersize = 10, label=r"$N_x = 512$")
plt.ylabel("State Change", fontsize = 16, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.yticks(color = "blue")
plt.xlabel("Iteration", fontsize = 16)
plt.legend(loc='best', bbox_to_anchor=(0.65, 0., 0.35, 0.35), framealpha=1)
# plt.legend(ncol = 3)
plt.title(r"Parareal convergence of CASE IV using ECSIM")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./3D_parareal_Convergence.png")

