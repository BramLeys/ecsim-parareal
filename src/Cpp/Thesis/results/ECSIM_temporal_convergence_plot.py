
import numpy as np
import matplotlib.pyplot as plt

# Data loading and selection
filename = "../../../../paper/Results/ecsim_time_convergence_errors.txt"
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

plt.loglog(ts, err_x, marker='^', color='blue', markersize = 16, label="Position")
plt.loglog(ts, err_v, marker='o', color='red', markersize = 10, label="Velocity")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta t^2)$")
plt.ylabel("Error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Timestep size ($\Delta t$)", fontsize = 20)
plt.legend(fontsize = 20)
fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/ECSIMTemporalConvergence_particles.eps")
plt.savefig("./figures/png/ECSIMTemporalConvergence_particles.png")

fig = plt.figure(figsize = (8,6), dpi = 300)
plt.loglog(ts, err_E, marker='H', color='green', markersize = 15, label="Electric field")
plt.loglog(ts, err_B, marker='s', color='orange', markersize = 10, label="Magnetic field")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta t^2)$")
plt.ylabel("Error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Timestep size ($\Delta t$)", fontsize = 20)
# fig.legend(bbox_to_anchor = (0.95, 0.999), ncol = 3, prop = {'size': 18})
plt.legend(fontsize = 20)
# plt.legend(fontsize=20,loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/ECSIMTemporalConvergence_fields.eps")
plt.savefig("./figures/png/ECSIMTemporalConvergence_fields.png")

for i in range(1,len(ts)):
    print('delta t', ts[i])
    print(f"position convergence: {np.log10(err_x[i]/err_x[i-1])/np.log10(ts[i]/ts[i-1])}")
    print(f"velocity convergence: {np.log10(err_v[i]/err_v[i-1])/np.log10(ts[i]/ts[i-1])}")
    print(f"electric field convergence: {np.log10(err_E[i]/err_E[i-1])/np.log10(ts[i]/ts[i-1])}")
    print(f"magnetic field convergence: {np.log10(err_B[i]/err_B[i-1])/np.log10(ts[i]/ts[i-1])}")

