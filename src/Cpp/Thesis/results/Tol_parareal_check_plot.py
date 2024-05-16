
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


# Data loading and selection
filename = "../../../../paper/Results/tol_parareal_check.txt"
data = np.loadtxt(filename)

its = data[:,0]
sim_level_est = data[:,1]
sim_level_act = data[:,2]
step_level_est = data[:,3]
step_level_act = data[:,4]


# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.semilogy(its, sim_level_est, marker='o', color='blue', markersize = 10, label=r"Estimated error for simulation level tolerance")
plt.semilogy(its, sim_level_act, marker='^', color='red', markersize = 15, label=r"Actual error for simulation level tolerance")
plt.semilogy(its, step_level_est, marker='+', color='orange', markersize = 10, label=r"Estimated error for timestep level tolerance")
plt.semilogy(its, step_level_act, marker='p', color='green', markersize = 10, label=r"Actual error for timestep level tolerance")
plt.ylabel("Error", fontsize = 16, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.yticks(color = "blue")
plt.xticks(range(1,its.shape[0]+1))
plt.xlabel("Iteration", fontsize = 16)
plt.legend()
plt.title(r"Estimated and actual errors incurred for different tolerance strategies")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./Tolerance_parareal_check.png")

