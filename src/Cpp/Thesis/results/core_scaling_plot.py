import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mticker

rc('text', usetex=True)

# Data loading and selection
dir = "../../../../paper/Results/"
filename = "./core_scaling_test.txt"
data = np.loadtxt(dir + filename)

cores = data[:,0]
ser_time = data[:,1]
par_time = data[:,2]
its = data[:,3]
err = data[:,4]
speedup = data[:,5]

# Plots

guide_speedup = 1/its*cores
fig = plt.figure(figsize = (12, 5), dpi = 300)
plt.subplot(1, 2, 1)
plt.plot(cores, speedup, marker='o', color='blue', markersize = 10, label  ="Speedup")
plt.plot(cores,guide_speedup,linestyle="dashed", color="red", markersize =10, label = "Theoretical bound")
plt.ylabel("Speedup", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"Number of cores", fontsize = 20)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(12))
plt.title(r"Speedup for increasing core usage", fontsize = 20)
plt.legend(fontsize=20)

plt.subplot(1, 2, 2)
plt.plot(cores, speedup/cores, marker='o', color='blue', markersize = 10, label  ="Parallel efficiency")
plt.plot(cores,guide_speedup/cores,linestyle="dashed", color="red", markersize =10, label = "Theoretical bound")
plt.ylabel("Parallel efficiency", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"Number of cores", fontsize = 20)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(12))
plt.title(r"Parallel efficiency for increasing core usage", fontsize = 20)
plt.legend(fontsize=20)

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./core_scaling_test.eps")

