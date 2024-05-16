import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mticker

rc('text', usetex=True)

# Data loading and selection
para_time_name = "./solver_para_timings.txt"
parallel_timings = np.loadtxt(para_time_name)

ser_time_name = "./solver_serial_timings.txt"
serial_timings = np.loadtxt(ser_time_name)

speedup = serial_timings/parallel_timings

# Plots

fig = plt.figure(figsize = (12, 5), dpi = 300)

solvers = ["LU", "GMRES", "BiCGSTAB"]
x = np.arange(len(solvers))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, speedup[0,:], width, label='Coarse LU')
rects2 = ax.bar(x, speedup[1,:], width, label='Coarse GMRES')
rects3 = ax.bar(x + width, speedup[2,:], width, label='Coarse BiCGSTAB')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Fine Solver', fontsize = 16)
ax.set_ylabel('Speedup', fontsize = 16)
ax.set_title('Speedup for different linear solvers', fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax.set_ylim(0,np.max(speedup) + 0.7)
ax.set_xticks(x)
ax.set_xticklabels(solvers)
ax.legend()

# saving

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./solver_test.png")

