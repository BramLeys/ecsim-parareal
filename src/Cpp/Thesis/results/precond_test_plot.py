import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mticker

rc('text', usetex=True)

# Data loading and selection
filename = "./preconditioner_timings.txt"
data = np.loadtxt(filename)

# select data
ts = data[:,0]
id = data[:,1]
jac = data[:,2]
lu = data[:,3]


# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.loglog(ts, id/1000,marker='o', color='blue', markersize = 10, label="Identity")
plt.loglog(ts, jac/1000,marker='s', color='red', markersize = 10, label="Jacobi")
plt.loglog(ts, lu/1000,marker='^', color='green', markersize = 10, label="Incomplete LU")
plt.ylabel("Time (s)", fontsize = 16)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Time step size", fontsize = 16)
plt.legend()

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./precond_test.png")

