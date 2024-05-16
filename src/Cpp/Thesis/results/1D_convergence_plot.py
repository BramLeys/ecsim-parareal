
import numpy as np
import matplotlib.pyplot as plt

# Data loading and selection
filename = "./ecsim1d_convergence_errors.txt"
data = np.loadtxt(filename)


ts = data[:,0]
err_x = data[:,1]
err_v = data[:,2]
err_E = data[:,3]
err_B = data[:,4]

max_err = np.max(data[:,1:])

# guides
second_order = ts**2* ts[0]**2/max_err

# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.loglog(ts, err_x, marker='o', color='blue', markersize = 10, label="Position")
plt.loglog(ts, err_v, marker='x', color='red', markersize = 10, label="Velocity")
plt.loglog(ts, err_E, marker='+', color='green', markersize = 10, label="Electric field")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta t^2)$")
plt.ylabel("Error", fontsize = 16, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.yticks(color = "blue")
plt.xlabel(r"Timestep size", fontsize = 16)
# fig.legend(bbox_to_anchor = (0.95, 0.999), ncol = 3, prop = {'size': 18})
plt.legend()
plt.title(r"Convergence of CASE III using ECSIM")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./ECSIM1DConvergence.png")

