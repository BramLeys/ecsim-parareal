import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy as sp

mat = sp.io.loadmat("src/Cpp/Thesis/results/maxwell.mat")
matMax = mat["ans"]

CppMax = np.genfromtxt("src/Cpp/Thesis/results/maxwellCpp.mat", dtype = np.double)

diff = matMax - CppMax
for i in range(diff.shape[0]):
    if any(abs(diff[i,:]) > 1e-15):
        print(f"row {i} has problem: matlab row = {matMax[i,:]}, Cpp row = {CppMax[i,:]}, diff = {diff[i,:]}")

