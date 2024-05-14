import numpy as np
import math
import time
import scipy as sp
import Solvers
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parareal convergence test')
    parser.add_argument('-c', '--coarse_dt', type=float, default=1e-2,
                        help='Set the value of coarse solver dt (default: 1e-2)')
    parser.add_argument('-f', '--fine_dt', type=float, default=1e-4,
                        help='Set the value of fine solver dt (default: 1e-4)')
    parser.add_argument('-tr', '--thresh', type=float, default=1e-8,
                        help='Set the value of threshold (default: 1e-8)')
    parser.add_argument('-N', '--N', type=int, default=10,
                        help='Set the value of number of gridpoints (default: 10)')
    parser.add_argument('-t', '--T', type=float, default=0,
                        help='Set the time length, 0 becomes 12*coarse dt (default: 0)')
    return parser.parse_args()

L = 2 * math.pi
args = parse_arguments()
args.T = 12 * args.coarse_dt if args.T == 0 else args.T

dx = L / args.N

def second_order(tn,xn):
    yn = np.empty(xn.shape)
    for i in range(yn.shape[0]):
        yn[i] = (xn[(i - 1 + yn.shape[0]) % yn.shape[0]] - 2 * xn[i] + xn[(i + 1) % yn.shape[0]]) / (dx * dx)
    return yn

B = np.zeros((args.N, args.N))
for i in range(args.N):
	B[i, (i + 1) % args.N] = 1 / (dx * dx)
	B[i, (i ) % args.N]  = -2 / (dx * dx)
	B[i, (i - 1 + args.N) % args.N]  = 1 / (dx * dx)
B = sp.sparse.csr_array(B)


NT = (int)(args.T / args.coarse_dt)
ts = np.linspace(0, args.T,NT + 1)

F = Solvers.CrankNicholson(B, args.fine_dt, args.thresh/100)
G = Solvers.CrankNicholson(B, args.coarse_dt, args.thresh/100)
ref_solver = Solvers.CrankNicholson(B, args.fine_dt/pow(2,3), 1e-15)

parareal_solver = Solvers.PararealSolver(50, F, G, it_threshold=args.thresh)

Xn = np.sin(2*np.linspace(0,L,args.N,endpoint=False)) + 5
# Xn = np.random.rand(args.N)
Yn_ref = ref_solver.Step(Xn, 0, args.T,True)
X_para = parareal_solver.Solve(Xn, ts)
Y_ser = F.Step(Xn,0,args.T,True)

print(Y_ser.shape)
refinement = round(args.coarse_dt/args.fine_dt)
ser_steps = list(range(0,Y_ser.shape[0],refinement))
ref_steps = list(range(0,Yn_ref.shape[0],refinement*8))
print(ser_steps)

k = X_para.shape[0]
max_errors = np.empty(k)
for i in range(k):
    max_errors[i] = np.max(np.linalg.norm((Yn_ref[ref_steps,:] - X_para[i,:,:]),axis=-1))
print(max_errors)
print(f"max fine error: {np.max(np.linalg.norm((Yn_ref[ref_steps,:] - Y_ser[ser_steps,:]),axis=-1))}")
print(f"error compared to serial: {np.linalg.norm((X_para[-1,:,:] - Y_ser[ser_steps,:]),axis=-1)}")

