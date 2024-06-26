import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# load data
dir = "../../../../paper/Results/"
# dir = "./"
filename = "time_step_speedup.txt"
data = np.loadtxt(dir + filename)

coarse_steps = [r"$\Delta t_\mathrm{Coarse} = 10^{-2}$",r"$\Delta t_\mathrm{Coarse} = 10^{-3}$",r"$\Delta t_\mathrm{Coarse} = 10^{-4}$" ]
combs = np.array([["o", "blue"], ["^", "green"], ["p", "red"] , ["H", "orange"], ["s", "black"]])

# select data
ref = data[:,0]
# Plot

fig = plt.figure(figsize = (8,6), dpi = 200)

for i in range(1,data.shape[1]):
    plt.semilogx(ref, data[:,i], marker=combs[i-1][0], color=combs[i-1][1], markersize = 10, label=coarse_steps[i-1])
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"$\Delta t_\mathrm{Coarse}/\Delta t_\mathrm{Fine}$", fontsize = 20)
plt.ylabel("Speedup (with parareal)", fontsize = 20)
# plt.title("Speedup for different coarse step sizes", fontsize=25)
plt.legend(fontsize=20)

# save fig

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/time_step_constant_coarse_speedup.eps")
plt.savefig("./figures/png/time_step_constant_coarse_speedup.png")


