
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# Data loading and selection
filename = "../../../../paper/Results/cn_convergence_errors.txt"
data = np.loadtxt(filename)

ts = data[:,0]
err_para = data[:,1]
err_ser = data[:,2]

max_err = np.max(data[:,1:])

# guides
second_order = ts**2

# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.loglog(ts, err_ser, marker='^', color='red', markersize = 15, label="Serial")
plt.loglog(ts, err_para, marker='o', color='blue', markersize = 10, label="Parareal")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta t^2)$")
plt.ylabel("Error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Timestep size ($\Delta t$)", fontsize = 20)
# fig.legend(bbox_to_anchor = (0.95, 0.999), ncol = 3, prop = {'size': 18})
plt.legend(fontsize=20)

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/CNConvergence.eps")
plt.savefig("./figures/png/CNConvergence.png")

