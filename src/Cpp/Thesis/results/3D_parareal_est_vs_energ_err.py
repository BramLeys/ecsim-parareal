
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


# Data loading and selection
filename = "../../../../paper/Results/parareal_iteration_information.txt"
data = np.loadtxt(filename)

its = data[:,0]
err_est = data[:,1]
err_act = data[:,3]

# Plots

fig = plt.figure(figsize = (8,6), dpi = 300)

plt.semilogy(its, err_est, marker='o', color='blue', markersize = 10, label=r"Parareal error")
if(np.any(err_act > 0)):
    plt.semilogy(its, err_act, marker='^', color='red', markersize = 14, label=r"Energy error")
plt.ylabel("Error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20, )
plt.xticks(range(1,its.shape[0]+1))
plt.xlabel("Iteration", fontsize = 20)
plt.legend( fontsize = 20)
plt.title(r"State variable and energy error during parareal", fontsize = 23)

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/est_vs_energy_err_parareal_check.eps")
plt.savefig("./figures/png/est_vs_energy_err_parareal_check.png")

