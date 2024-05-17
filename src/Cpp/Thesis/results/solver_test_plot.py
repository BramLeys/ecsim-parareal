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

# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

plt.subplot(1, 2, 1)
solvers = ["LU", "GMRES", "BiCGSTAB"]
x = np.arange(len(solvers))  # the label locations
width = 0.25  # the width of the bars

plt.bar(x - width, speedup[0,:], width, label='Coarse LU')
plt.bar(x, speedup[0,:], width, label='Coarse GMRES')
plt.bar(x + width, speedup[0,:], width, label='Coarse BiCGSTAB')

# Add some text for labels, title, and custom x-axis tick labels, etc.
plt.xlabel('Fine Solver', fontsize = 20)
plt.ylabel('Speedup', fontsize = 20)
plt.title('Speedup for different linear solvers', fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
plt.ylim(0,np.max(speedup) + 1)
plt.xticks(x,labels = solvers)

plt.subplot(1, 2, 2)
solvers = ["LU", "GMRES", "BiCGSTAB"]
x = np.arange(len(solvers))  # the label locations
width = 0.25  # the width of the bars

plt.bar(x - width, parallel_timings[0,:]/1000, width, label='Coarse LU')
plt.bar(x, parallel_timings[1,:]/1000, width, label='Coarse GMRES')
plt.bar(x + width, parallel_timings[2,:]/1000, width, label='Coarse BiCGSTAB')

# Add some text for labels, title, and custom x-axis tick labels, etc.
plt.xlabel('Fine Solver', fontsize = 20)
plt.ylabel('Time (s)', fontsize = 20)
plt.title('Computational runtime', fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
plt.ylim(0,np.max(parallel_timings)/1000 + 1)
plt.xticks(x,labels = solvers)

# Put a legend to the right of the current axis
plt.legend(fontsize=20,loc='center left', bbox_to_anchor=(1, 0.5))


# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./solver_test.eps")

