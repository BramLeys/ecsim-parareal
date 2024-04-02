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

    Nx=128 # number of grid cells
    L=2*math.pi # Size of position space
    NT=250 # number of time steps
    Np=10000 # number of particles
    graphics = True # should graphics be shown
    NTOUT=NT//2 # How many times should updates be shown
    dt=.125 #time step length
    dx=L/Nx # length of each grid cell
    Nsub = 1

    qom=-1 #charge mass density of particles

    theta=0.5 #theta of field and particle mover
    mode=3 # mode of perturbation sin

    xv=np.linspace(0,L,Nx+1) # position of each of the grid cell vertices
    xc=.5*(xv[1:]+xv[:-1]) # position of each of the grid cell centers
    Vx=dx*np.ones(Nx) # volumes of each of the grid cells
    xdim = 1
    vdim = 3
    E0 = np.zeros((Nx,vdim))#initialization of electric field in each of the grid cells

    B = np.zeros((Nx,vdim))

    Bc = np.ones((Nx,vdim))#initialization of magnetic field in each of the grid cell centers
    # Bc[:,0] = np.ones(Nx)/10

    VT = np.ones(vdim)*0.01 # thermic velocity of particles in each direction
    V0 = 0.2
    V1 = 0

    xp=np.linspace(0,L-L/Np,Np)
    vp=np.transpose(VT[:,None]*np.random.randn(vdim,Np))
    sigma_x=.01

    pm=1+ np.arange(Np)
    pm=1-2*(pm%2)
    vp[:,1] += pm*V0

    vp[:,1] += V1*np.sin(2*math.pi*xp/L*mode)
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

    histEnergy = np.empty((NT))
    # histEnergyB = np.empty((NT))
    # histEnergyE = np.empty((NT))
    # histEnergyK = np.empty((NT))
    # histMomentum = np.empty((NT))
    # histFmode = np.empty(NT)
    # histxvel = np.empty((NT,Np))

    # spectrumE=np.empty((NT,Nx,vdim))
    # spectrumB=np.empty((NT,Nx,vdim))
    time_matrix = 0

    dt_coarse = dt
    dt_fine = dt_coarse/2
    max_para_iterations = 6
    wp_E = 1e-15
    wp_P = 1e-6

    coarse_ECSIM = Solvers.ECSIM(Np,Nx,Nsub,dt_coarse,qp)
    fine_ECSIM = Solvers.ECSIM(Np,Nx,Nsub,dt_fine,qp)
    para = Solvers.PararealSolver(max_para_iterations, fine_ECSIM,coarse_ECSIM, wp_E, wp_P)
    Xn = np.hstack((np.reshape(xp,(Np*xdim)),np.reshape(vp,(Np*vdim)),np.reshape(E0,(Nx*vdim)),np.reshape(Bc,(Nx*vdim))))
    Xn = fine_ECSIM.Step(Xn,0,dt_fine)
    Ek,Ee,Eb = fine_ECSIM.Energy(Xn)

    time_start = time.perf_counter()
    X1 = fine_ECSIM.Step(Xn,0,(NT)*dt)
    time_end = time.perf_counter()
    print(f"Serial simulation of {NT} timesteps took {time_end-time_start:0.4f}s")

    time_start = time.perf_counter()
    X1 = para.Solve(Xn,np.arange(NT+1)*dt)
    time_end = time.perf_counter()
    print(f"Parallel simulation of {NT} timesteps took {time_end-time_start:0.4f}s")

    axes = []
    for k in range(X1.shape[0]):
        # plt.rc('text', usetex=True)
        axes.append(plt.subplots(3,figsize=(8, 6), dpi=100))
        line_pos, = axes[k][1][0].plot([],[],"b.", label='vy > 0')
        line_neg, = axes[k][1][0].plot([],[],"r.", label='vy < 0')
        line_Etot, = axes[k][1][1].semilogy([],[],label="difference in total energy")
        axes[k][1][0].legend()
        line_posy, = axes[k][1][2].plot([],[],"b.", label='vx > 0')
        line_negy, = axes[k][1][2].plot([],[],"r.", label='vx < 0')
        axes[k][1][2].legend()
        axes[k][1][0].set_xlim(0, L)
        axes[k][1][1].set_xlim(0, NT*dt)
        axes[k][1][2].set_xlim(0, L)
        for it in range(NT):
            Xn = X1[k,it+1,:]
            # Xn = coarse_ECSIM.Step(Xn,dt*it, dt*(it+1))
            xp = np.reshape(Xn[:xdim*Np],(Np))%L
            vp = np.reshape(Xn[xdim*Np:xdim*Np+vdim*Np], (Np,vdim))
            Ekold=Ek; Ebold=Eb; Eeold=Ee
            Ek,Ee,Eb = fine_ECSIM.Energy(Xn)
            Ediff = Ek - Ekold + Ee - Eeold + Eb - Ebold
            histEnergy[it] = Ek + Ee + Eb
            if((it%round(NT/NTOUT)==0) and graphics):
                ii = np.where(vp[:,1]>0)[0]; jj = np.where(vp[:,1]<0)[0]
                axes[k][1][0].set_title(f"k = {k}, t = {(it+1)*dt}")
                line_pos.set_data(xp[ii],vp[ii,0])
                line_neg.set_data(xp[jj],vp[jj,0])
                axes[k][1][0].set_xlabel('x')
                axes[k][1][0].set_ylabel('vx')
                axes[k][1][0].relim()
                axes[k][1][0].autoscale_view(None,False,True)
                ii = np.where(vp[:,0]>0)[0]; jj = np.where(vp[:,0]<0)[0]
                axes[k][1][2].set_title(f"k = {k}, t = {(it+1)*dt}")
                line_posy.set_data(xp[ii],vp[ii,1])
                line_negy.set_data(xp[jj],vp[jj,1])
                axes[k][1][2].set_xlabel('x')
                axes[k][1][2].set_ylabel('vy')
                axes[k][1][2].relim()
                axes[k][1][2].autoscale_view(None,False,True)
                line_Etot.set_data(np.arange(it+1)*dt,abs((histEnergy[:it+1] - histEnergy[0])/histEnergy[0])) # how does this prove energy conservation?
                axes[k][1][1].set_xlabel('t')
                axes[k][1][1].set_ylabel('|E(t) - E(0)|/E(0)')
                axes[k][1][1].relim()
                axes[k][1][1].autoscale_view(None,False,True)
                plt.pause(0.01)
                plt.draw()
    plt.ioff()
    plt.show()

