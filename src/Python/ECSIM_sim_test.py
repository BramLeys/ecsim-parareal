import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import scipy as sp
import Solvers
import common

if __name__ == '__main__':
    plt.ion()

    Nx=500 # number of grid cells
    L=2*math.pi # Size of position space
    NT=2**8 # number of time steps
    Np=10000 # number of particles
    graphics = True # should graphics be shown
    NTOUT=NT//2 # How many times should updates be shown
    dt=0.125 #time step length
    dx=L/Nx # length of each grid cell
    Nsub = 1

    qom=-1 #charge mass density of particles

    theta=0.5 #theta of field and particle mover
    mode=5    # mode of perturbation sin

    xv=np.linspace(0,L,Nx+1) # position of each of the grid cell vertices
    xc=.5*(xv[1:]+xv[:-1]) # position of each of the grid cell centers
    Vx=dx*np.ones(Nx) # volumes of each of the grid cells
    xdim = 1
    vdim = 1
    E0 = np.zeros((Nx,vdim))#initialization of electric field in each of the grid cells

    Bc = np.zeros((Nx,vdim))#initialization of magnetic field in each of the grid cell centers
    # Bc[:,0] = np.ones(Nx)/10

    VT = 0.02 # thermic velocity of particles in each direction
    V0 = 0.1
    V1 = .1*V0

    xp=np.linspace(0,L-L/Np,Np)
    vp=VT*np.random.randn(Np,vdim)
    # vp = (VT*np.linspace(0,1,Np)).reshape(Np,vdim)
    sigma_x=.01

    pm=1+ np.arange(Np)
    pm=1-2*(pm%2)
    vp[:,0] += pm*V0

    vp[:,0] += V1*np.sin(2*math.pi*xp/L*mode)

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

    histEnergy = np.empty((NT))

    solver = Solvers.ECSIM(Np,Nx,1,dt,qp, 1)
    Xn = np.hstack((np.reshape(xp,(Np*xdim)),np.reshape(vp,(Np*vdim)),np.reshape(E0,(Nx*vdim)),np.reshape(Bc,(Nx*vdim))))
    # Xn = fine_ECSIM.Step(Xn,0,dt_fine)
    Ek,Ee,Eb = solver.Energy(Xn)
    print(f"kinetic energy = {Ek}, electric energy = {Ee}, magnetic energy = {Eb}")
    
    # time_start = time.perf_counter()
    # X1 = solver.Step(Xn,0,(NT)*dt)
    # time_end = time.perf_counter()
    # print(f"Serial simulation of {NT*dt}s took {time_end-time_start:0.4f}s")


    ax = plt.subplots(2,figsize=(8, 6), dpi=100)
    line, = ax[1][0].plot([],[],"b.")
    ax[1][0].set_xlim(0, L)
    line_Etot, = ax[1][1].semilogy([],[],label="difference in total energy")
    ax[1][1].set_xlim(0, NT*dt)
    X1 = Xn
    for it in range(NT):
            X1 = solver.Step(X1,dt*it, dt*(it+1))
            xp = np.reshape(X1[:xdim*Np],(Np))%L
            vp = np.reshape(X1[xdim*Np:xdim*Np+vdim*Np], (Np,vdim))
            Ek1, Ee1,Eb1 = solver.Energy(X1)
            histEnergy[it] = Ek1 + Ee1 + Eb1
            print(f"kinetic energy = {Ek1}, electric energy = {Ee1}, magnetic energy = {Eb1}")
            if((it%round(NT/NTOUT)==0) and graphics):
                ax[1][0].set_title(f"Fine solution")
                line.set_data(xp,vp)
                ax[1][0].set_xlabel('x')
                ax[1][0].set_ylabel('vx')
                ax[1][0].relim()
                ax[1][0].autoscale_view(None,False,True)
                line_Etot.set_data(np.arange(it+1)*dt,abs((histEnergy[:it+1] - histEnergy[0])/histEnergy[0]))
                ax[1][1].set_xlabel('t')
                ax[1][1].set_ylabel(r'\frac{|E(t) - E(0)|}{E(0)}')
                ax[1][1].relim()
                ax[1][1].autoscale_view(None,False,True)
                plt.pause(0.01)
                plt.draw()

    plt.ioff()
    plt.show()

