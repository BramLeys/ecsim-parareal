import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# load data
filename = "./subcycling_test.txt"
data = np.loadtxt(filename)

# select data
nsub = data[:,0]
time_ser = data[:,1]
time_par = data[:,2]
it = data[:,3]
err = data[:,4]
speedup = data[:,5]
	

# Plot

fig = plt.figure(figsize = (12, 5), dpi = 200)

# plt.subplot(1, 2, 1)
# plt.semilogx(nsub, time_par/1000,marker='o', color='blue', markersize = 10)
# plt.ylabel("Parareal solver (s)", fontsize = 16, color = "blue")
# plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
# plt.yticks(color = "blue")
# plt.xlabel(r"Number of subcycles", fontsize = 16)


# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax2.set_ylabel("Serial solver (s)", fontsize = 16, color = "red")
# ax2.semilogx(nsub, time_ser/1000, marker='d', color='red', markersize = 8)
# plt.tick_params(axis = 'y', which = 'major', labelsize = 14, color = "red")
# plt.yticks(color = "red")


# plt.subplot(1, 2, 2)
plt.plot(nsub, speedup, marker='H', color='black', markersize = 10)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"Number of subcycles", fontsize = 16)
plt.ylabel("Speedup (with parareal)", fontsize = 16)

# save fig

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./subcycle_speedup.png")