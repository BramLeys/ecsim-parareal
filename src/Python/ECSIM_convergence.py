import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import scipy as sp
import Solvers
import common

if __name__ == '__main__':
    Nx=128 # number of grid cells
    L=2*math.pi # Size of position space
    NT=10 # number of time steps
    Np=10000 # number of particles
    graphics = True # should graphics be shown
    NTOUT=NT//2 # How many times should updates be shown
    dx=L/Nx # length of each grid cell
    Nsub = 1

    qom=-1 #charge mass density of particles

    theta=0.5 #theta of field and particle mover
    mode=5    # mode of perturbation sin

    Vx=dx*np.ones(Nx) # volumes of each of the grid cells
    xdim = 1
    vdim = 1
    E0 = np.zeros((Nx,vdim))#initialization of electric field in each of the grid cells

    Bc = np.zeros((Nx,vdim))#initialization of magnetic field in each of the grid cell centers
    Bc[:,0] = np.ones(Nx)/10

    VT = 0.02 # thermic velocity of particles in each direction
    V0 = 0.1
    V1 = .1*V0

    xp=np.linspace(0,L-L/Np,Np)
    vp=VT*np.random.randn(Np,vdim)
    # vp = (VT*np.linspace(0,1,Np)).reshape(Np,vdim)

    pm=1+ np.arange(Np)
    pm=1-2*(pm%2)
    vp[:,0] += pm*V0

    vp[:,0] += V1*np.sin(2*math.pi*xp/L*mode)
    # vel = sp.io.loadmat("src/test.mat")
    # vp[:,:] = vel["vel"]

    ix=np.floor(xp/dx).astype(int); # cell of the particle, first cell is cell 0, first node is 0 last node Nx
    frac1 = 1-(xp/dx-ix)
    ix2=(ix+1)%Nx # second cell of influence due to extended local support of b-spline


    M=np.zeros((Nx,Nx))
    for ip in range(Np):
        M[ix[ip],ix[ip]] += frac1[ip]**2/2; # divided by 2 for the symmetry added later
        M[ix2[ip],ix[ip]] += frac1[ip]*(1-frac1[ip])
        M[ix2[ip],ix2[ip]] += (1-frac1[ip])**2/2
    M=M+M.T

    rhotarget=-1*Vx
    rhotildeV=sp.sparse.linalg.spsolve(sp.sparse.csr_array(M),rhotarget)

    qp = np.empty(Np)
    rho = np.zeros(Nx)
    for ip in range(Np):
        qp[ip]=rhotildeV[ix[ip]]*frac1[ip]+rhotildeV[ix2[ip]]*(1-frac1[ip])
        rho[ix[ip]] += frac1[ip]*qp[ip]
        rho[ix2[ip]] += (1-frac1[ip])*qp[ip]

    rho=rho/Vx

    dt_base=1e-1 #time step length

    solver = Solvers.ECSIM(Np,Nx,Nsub,dt_base,qp, 1)
    Xn = np.hstack((np.reshape(xp,(Np*xdim)),np.reshape(vp,(Np*vdim)),np.reshape(E0,(Nx*vdim)),np.reshape(Bc,(Nx*vdim))))
    T = dt_base*20
    refinements = 4
    solver.dt = dt_base/pow(2,refinements-1) / 50
    print("Calculating reference")
    time_start = time.perf_counter()
    X_ref = solver.Step(Xn,0,T)
    time_end = time.perf_counter()
    print(f"Reference simulation of {T/solver.dt} timesteps took {time_end-time_start:0.4f}s")
    sol = np.empty((refinements,Xn.shape[0]))
    errors = np.empty((refinements,5))
    for i in range(refinements):
        solver.dt = dt_base/pow(2,i)
        errors[i,0] = solver.dt
        time_start = time.perf_counter()
        sol[i,:] = solver.Step(Xn,0,T)
        time_end = time.perf_counter()
        print(f"Simulation of {T/solver.dt} timesteps took {time_end-time_start:0.4f}s")

    xp_ref = X_ref[:xdim*Np]
    vp_ref = X_ref[xdim*Np:xdim*Np+vdim*Np]
    E0_ref = X_ref[xdim*Np+vdim*Np:xdim*Np+vdim*(Np + Nx)]
    Bc_ref = X_ref[xdim*Np+vdim*(Np + Nx):xdim*Np+vdim*(Np + 2*Nx)]
    convergence = np.empty((refinements,4))
    for i in range(refinements):
        xp = sol[i,:xdim*Np]
        vp = sol[i,xdim*Np:xdim*Np+vdim*Np]
        E0 = sol[i,xdim*Np+vdim*Np:xdim*Np+vdim*(Np + Nx)]
        Bc = sol[i,xdim*Np+vdim*(Np + Nx):xdim*Np+vdim*(Np + 2*Nx)]
        errors[i,1:] = [np.linalg.norm(xp-xp_ref)/np.linalg.norm(xp_ref),np.linalg.norm(vp-vp_ref)/np.linalg.norm(vp_ref),
                        np.linalg.norm(E0-E0_ref)/np.linalg.norm(E0_ref),np.linalg.norm(Bc-Bc_ref)/np.linalg.norm(Bc_ref)]
        if i > 0:
            convergence[i,:] = errors[i,1:]/errors[i-1,1:]
    print(f"errors = {errors}")
    print(f"convergence = {convergence}")




