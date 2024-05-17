import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mticker

rc('text', usetex=True)

# Data loading and selection
dir = "../../../../paper/Results/"
para_time_name = "solver_para_timings.txt"
parallel_timings = np.loadtxt(dir + para_time_name)

ser_time_name = "solver_serial_timings.txt"
serial_timings = np.loadtxt(dir + ser_time_name)

speedup = serial_timings/parallel_timings
print(speedup)

solvers = ["LU", "GMRES", "BiCGSTAB"]
x = np.arange(len(solvers))  # the label locations
# Plots

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (12, 5), dpi = 300, sharex=True)
fig.subplots_adjust(top=0.8) 

# Plotting on the first subplot
ax1.bar(np.arange(len(solvers)) - 0.25, speedup[0, :], 0.25, label='Coarse LU')
ax1.bar(np.arange(len(solvers)), speedup[1, :], 0.25, label='Coarse GMRES')
ax1.bar(np.arange(len(solvers)) + 0.25, speedup[2, :], 0.25, label='Coarse BiCGSTAB')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax1.set_xlabel('Fine Solver', fontsize=20)
ax1.set_ylabel('Speedup', fontsize=20)
ax1.set_title('Speedup for different linear solvers', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_xticks(np.arange(len(solvers)))
ax1.set_xticklabels(solvers)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height*1.2])

# Plotting on the second subplot
ax2.bar(np.arange(len(solvers)) - 0.25, parallel_timings[0, :] / 1000, 0.25)
ax2.bar(np.arange(len(solvers)), parallel_timings[1, :] / 1000, 0.25)
ax2.bar(np.arange(len(solvers)) + 0.25, parallel_timings[2, :] / 1000, 0.25)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax2.set_xlabel('Fine Solver', fontsize=20)
ax2.set_ylabel('Time (s)', fontsize=20)
ax2.set_title('Computational runtime', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_xticks(np.arange(len(solvers)))
ax2.set_xticklabels(solvers)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width, box.height*1.2])


# Creating a single legend for both subplots
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fontsize=20)


# saving

plt.tight_layout()
plt.savefig("./solver_test.eps")

