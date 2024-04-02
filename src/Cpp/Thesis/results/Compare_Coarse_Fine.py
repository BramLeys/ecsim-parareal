"""
Created on Sun Mar 10 19:57 2024

@author: Pranab JD
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

### Given Data Sets
file = "./src/Cpp/Thesis/results/Parareal_speedup_fine_time_steps.txt"
data = np.loadtxt(file)

### ========================================================================== ####
data_points = data.size[0]

dt = np.zeros(data_points)
T_serial = np.zeros(data_points)
T_parareal = np.zeros(data_points)


dt[0] = data[0][0]
T_serial[0] = data[0][1]
T_parareal[0] = data[0][2]

for ii in range(1, data_points):
	dt[ii] = data[ii][0]
	T_serial[ii] = data[ii][1]
	T_parareal[ii] = data[ii][2]
	
### ========================================================================== ####

### Plots

fig = plt.figure(figsize = (12, 5), dpi = 200)

### ----------------------------------- ###

plt.subplot(1, 2, 1)
plt.plot(dt, T_parareal/1000, 'bo', markersize = 10)
plt.ylabel("Parareal solver (s)", fontsize = 16, color = "blue")
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.yticks(color = "blue")
plt.xlabel(r"$\frac{\Delta t_\mathrm{Coarse}}{\Delta t_\mathrm{Fine}}$", fontsize = 16)


ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.set_ylabel("Serial solver (s)", fontsize = 16, color = "red")
ax2.plot(dt, T_serial/1000, 'rd', markersize = 8)
plt.tick_params(axis = 'y', which = 'major', labelsize = 14, color = "red")
plt.yticks(color = "red")


plt.subplot(1, 2, 2)
plt.plot(dt, T_serial/T_parareal, 'kH', markersize = 10)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.xlabel(r"$\frac{\Delta t_\mathrm{Coarse}}{\Delta t_\mathrm{Fine}}$", fontsize = 16)
plt.ylabel("Speedup (with parareal)", fontsize = 16)

### ----------------------------------- ###

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./Coarse_fine_cost_comparison.eps")

### ========================================================================== ####

"""
1. Try the same for dt_coarse/dt_fine = 2, 3, 4, 5, 6, 7, 8, 9, 10.

2. Keeping final simulation time constant, try larger values of dt_coarse. Do you, now, need more parareal iterations to converge? Hopefully, yes!

3. Save plots in ".eps" format. It's easier to resize (if needed) in latex. Png or jpg don't tend to resize well

4. Try including your plots in a latex file (format article or report). Are the plots clearly visible? Are the axes lables and legends readable? Are the ticks visible?
	If yes, great! If not, you may wish to try the figure formatting options from this python file.

"""