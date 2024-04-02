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

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xvalues', help="filename of x-axis values")
parser.add_argument('-y', '--yvalues', help="filename of y-axis values")
parser.add_argument('-t', '--title', help="title of plot")
parser.add_argument('-s', '--semilog', help="makes a semilog plot instead of normal plot", action='store_true')
parser.add_argument('-g', '--guide', help="add guide line",type=int)

args = parser.parse_args()

if args.yvalues is None:
    args.yvalues = "src\Cpp\Thesis\\results\Parareal_states_error_iteration"
if args.title is None:
    args.title = "Errors for each parareal iteration"
args.semilog = True

files = find_files_with_preamble("./", args.yvalues)
print(files)
for i in range(len(files)):
    print(files[i])
    X = np.genfromtxt(files[i], dtype = np.double).T
    NT = X.shape[0]
    print("time length",NT)
    if args.xvalues is not None:
        T = np.genfromtxt(args.xvalues, dtype = np.double).T
    else:
        T = np.arange(NT)
    #Visualize
    plt.figure()
    plt.title(args.title + " iteration "+ str(i))
    if args.semilog:
        if args.guide is not None:
            plt.semilogy(T,np.power(np.full(NT, 10,dtype=np.double),-args.guide*np.arange(NT)), label="O(dt^"+str(args.guide)+")")
        plt.semilogy(T,X)
    else:
        plt.plot(T,X)
    plt.xlabel("n")
    plt.ylabel("|X^k - X^(k-1)|_2/|X^k|_2")
    plt.show(block=False)
    plt.legend()
    plt.pause(0.5)
plt.show()

