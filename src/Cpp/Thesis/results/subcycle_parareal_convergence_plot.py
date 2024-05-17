import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# load data
dir = "../../../../paper/Results/"
filename = "subcycle_parareal_convergence.txt"
data = np.loadtxt(dir +filename).transpose()

k = data.shape[1]-1

combs = np.array([["o", "blue"], ["^", "green"], ["p", "red"] , ["H", "orange"], ["s", "black"]])
	

# Plot

fig = plt.figure(figsize = (8,6), dpi = 200)

its = range(1,k+1)
for i in range(data.shape[0]):
    plt.semilogy(its, data[i,1:],marker=combs[i][0], color=combs[i][1], markersize = 10, label = r"$\nu = $"+str(int(data[i,0])))

plt.ylabel("Parareal error", fontsize = 20)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
plt.xlabel(r"Parareal iteration", fontsize = 20)
plt.title("Convergence of parareal using subcycling", fontsize=25)
plt.legend(fontsize=20)

# save fig

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig("./subcycle_parareal_convergence.png")
