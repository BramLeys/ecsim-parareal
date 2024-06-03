
import numpy as np
import matplotlib.pyplot as plt

# Data loading and selection
filename = "../../../../paper/Results/ecsim_spatial_convergence_errors.txt"
data = np.loadtxt(filename)

ts = data[:,0]
err_x = data[:,1]
err_v = data[:,2]
err_E = data[:,3]
err_B = data[:,4]

max_err = np.max(data[:,1:])

# guides
second_order = ts**2

# Plots

fig = plt.figure(figsize = (8,6), dpi = 300)

plt.loglog(ts, err_E, marker='H', color='green', markersize = 15, label="Electric field")
plt.loglog(ts, err_B, marker='s', color='orange', markersize = 10, label="Magnetic field")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta x^2)$")
plt.ylabel("Error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Grid cell size ($\Delta x$)", fontsize = 20)
# fig.legend(bbox_to_anchor = (0.95, 0.999), ncol = 3, prop = {'size': 18})
plt.legend(fontsize=20)

for i in range(1,len(ts)):
    print('delta t', ts[i])
    print(f"electric field convergence: {np.log10(err_E[i]/err_E[i-1])/np.log10(ts[i]/ts[i-1])}")
    print(f"magnetic field convergence: {np.log10(err_B[i]/err_B[i-1])/np.log10(ts[i]/ts[i-1])}")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/ECSIMSpatialConvergence.eps")
plt.savefig("./figures/png/ECSIMSpatialConvergence.png")

