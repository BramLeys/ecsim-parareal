import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# load data
dir = "../../../../paper/Results/"
# dir = "./"
filename = "time_step_test.txt"
data = np.loadtxt(dir + filename)

# select data
ref = data[:,0]
time_ser = data[:,1]
time_par = data[:,2]
it = data[:,3]
err = data[:,4]
speedup = data[:,5]
	

# Plot

fig = plt.figure(figsize = (8, 6), dpi = 200)

# plt.subplot(1, 2, 1)
plt.semilogx(ref, time_par/1000,marker='o', color='blue', markersize = 10)
plt.ylabel("Parareal solver (s)", fontsize = 20, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.yticks(color = "blue")
plt.xlabel(r"$\Delta t_\mathrm{Coarse}/\Delta t_\mathrm{Fine}$", fontsize = 20)


ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.set_ylabel("Serial solver (s)", fontsize = 20, color = "red")
ax2.semilogx(ref, time_ser/1000, marker='d', color='red', markersize = 10)
plt.tick_params(axis = 'y', which = 'major', labelsize = 20, color = "red")
plt.yticks(color = "red")
# plt.title("Computational runtime", fontsize=25)
fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/time_step_constant_coarse_time.eps")
plt.savefig("./figures/png/time_step_constant_coarse_time.png")

fig = plt.figure(figsize = (8, 6), dpi = 200)
bound = 1/it * 96
bound = 1./(it*(1./96 + (1./it+1.)/ref))
# plt.subplot(1, 2, 2)
plt.semilogx(ref, speedup, marker='H', color='black', markersize = 10, label="Speedup")
plt.semilogx(ref, bound, linestyle="dashed", color='red', markersize = 10, label="Theoretical bound")
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"$\Delta t_\mathrm{Coarse}/\Delta t_\mathrm{Fine}$", fontsize = 20)
plt.ylabel("Speedup (with parareal)", fontsize = 20)
# plt.title("Speedup", fontsize=25)
plt.legend(fontsize=20)

# save fig

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/time_step_constant_coarse_single_speedup.eps")
plt.savefig("./figures/png/time_step_constant_coarse_single_speedup.png")
