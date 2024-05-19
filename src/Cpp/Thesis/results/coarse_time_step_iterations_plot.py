import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

rc('text', usetex=True)

# load data
dir = "../../../../paper/Results/"
# dir = "./"
filename = "coarse_time_step_test.txt"
data = np.loadtxt(dir + filename)

# select data
ref = data[:,0]
time_ser = data[:,1]
time_par = data[:,2]
it = data[:,3]
err = data[:,4]
speedup = data[:,5]
	

# Plot

fig = plt.figure(figsize = (8,6), dpi = 200)

plt.semilogx(ref, it,marker='o', color='blue', markersize = 10)
plt.ylabel("Parareal iterations", fontsize = 20)
plt.yticks(np.arange(0,20))
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"$\Delta t_\mathrm{Coarse}/\Delta t_\mathrm{Fine}$", fontsize = 20)
plt.title(r"Parareal iterations for $\Delta t_\mathrm{Fine} = 10^{-5}$", fontsize=25)



# save fig

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/time_step_constant_fine_iterations.eps")
plt.savefig("./figures/png/time_step_constant_fine_iterations.png")
