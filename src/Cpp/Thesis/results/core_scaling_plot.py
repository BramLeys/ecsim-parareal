import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mticker

rc('text', usetex=True)

# Data loading and selection
filename = "./core_scaling_test.txt"
data = np.loadtxt(filename)

cores = data[:,0]
ser_time = data[:,1]
par_time = data[:,2]
its = data[:,3]
err = data[:,4]
speedup = data[:,5]

# Plots


fig = plt.figure(figsize = (12, 5), dpi = 300)
plt.subplot(1, 2, 1)
plt.plot(cores, speedup, marker='o', color='blue', markersize = 10)
plt.ylabel("Speedup", fontsize = 16)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Number of cores", fontsize = 16)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(12))
plt.title(r"Speedup for increasing core usage")

plt.subplot(1, 2, 2)
plt.plot(cores, speedup/cores, marker='o', color='blue', markersize = 10)
plt.ylabel("Parallel efficiency", fontsize = 16)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Number of cores", fontsize = 16)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(12))
plt.title(r"Parallel efficiency for increasing core usage")

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./core_scaling_test.png")

