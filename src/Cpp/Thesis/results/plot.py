import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

def find_files_with_preamble(folder_path, preamble):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder path '{folder_path}' does not exist.")
        return []

    # Get all files in the folder with the given preamble
    print("looking for: ", os.path.join(folder_path, f'{preamble}*'))
    print("found: ", glob.glob(os.path.join(folder_path, f'{preamble}*')))
    files_with_preamble = [file for file in glob.glob(os.path.join(folder_path, f'{preamble}*')) if os.path.isfile(file)]

    return files_with_preamble


preamble = "src\Cpp\Thesis\\results\Parareal_states_error_iteration"
title = "Errors for each parareal iteration using threshold at 1e-8"

files = find_files_with_preamble("./", preamble)
print(files)
fig, ax = plt.subplots()
ax.set_title(title)
for i in range(len(files)):
    print(files[i])
    X = np.genfromtxt(files[i], dtype = np.double).T
    NT = X.shape[0]
    print("time length",NT)
    # T = np.genfromtxt(args.xvalues, dtype = np.double).T
    T = np.arange(NT)
    #Visualize
    # plt.semilogy(T,np.power(np.full(NT, 10,dtype=np.double),-args.guide*np.arange(NT)), label="O(dt^"+str(args.guide)+")")
    ax.semilogy(T,X, label="iteration "+str(i))
    # plt.plot(T,X)
ax.set_xlabel("n")
ax.set_ylabel(r'$\frac{\|X^{k}-X^{k+1}\|_2}{\|X^{k+1}\|_2}$')
plt.legend()
plt.pause(0.5)
plt.show()

