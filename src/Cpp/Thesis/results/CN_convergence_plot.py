
import numpy as np
import matplotlib.pyplot as plt

# Data loading and selection
filename = "./cn_convergence_errors.txt"
data = np.loadtxt(filename)

ts = data[:,0]
err_para = data[:,1]
err_ser = data[:,2]

max_err = np.max(data[:,1:])

# guides
second_order = ts**2* ts[0]**2/max_err

# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.loglog(ts, err_para, marker='o', color='blue', markersize = 10, label="Parareal")
plt.loglog(ts, err_ser, marker='x', color='red', markersize = 15, label="Serial")
plt.loglog(ts, second_order, linestyle="dashed", color="black",markersize = 10, label =r"$\mathcal{O}(\Delta t^2)$")
plt.ylabel("Error", fontsize = 16, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.yticks(color = "blue")
plt.xlabel(r"Timestep size", fontsize = 16)
# fig.legend(bbox_to_anchor = (0.95, 0.999), ncol = 3, prop = {'size': 18})
plt.legend()
plt.title(r"Convergence of CASE II using CN")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./CNConvergence.png")

