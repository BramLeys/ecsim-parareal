import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# load data
dir = "../../../../paper/Results/"
# dir = "./"
filename = "subcycling_test.txt"
data = np.loadtxt(dir + filename)

# select data
nsub = data[:,0]
time_ser = data[:,1]
time_par = data[:,2]
it = data[:,3]
err = data[:,4]
speedup = data[:,5]
	

# Plot

fig = plt.figure(figsize = (8, 6), dpi = 200)

# plt.subplot(1, 2, 2)
plt.plot(nsub, speedup, marker='H', color='black', markersize = 10)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"Number of subcycles", fontsize = 20)
plt.ylabel("Speedup (with parareal)", fontsize = 20)
# plt.title("Speedup using subcycling", fontsize=25)

# save fig

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./figures/eps/subcycle_speedup.eps")
plt.savefig("./figures/png/subcycle_speedup.png")
