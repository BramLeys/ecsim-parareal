import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy as sp
import Solvers

if __name__ == '__main__':
    plt.ion()

    Nx=128 # number of grid cells
    L=2*math.pi # Size of position space
    Np=10000 # number of particles
    graphics = True # should graphics be shown
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
    Bc[:,0] = np.ones(Nx)/10

    VT = np.ones(vdim)*0.02 # thermic velocity of particles in each direction
    V0 = 0.1
    V1 = .1*V0

    xp=np.linspace(0,L-L/Np,Np)
    vp=np.transpose(VT[:,None]*np.random.randn(vdim,Np))
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

    dt_coarse = 1e-2
    dt_fine = dt_coarse/10

    Nsub_fine = Nsub
    Nsub_coarse = Nsub_fine

    wp_E = 1e-15
    wp_P = 1e-15

    coarse_ECSIM = Solvers.ECSIM(Np,Nx,Nsub_coarse,dt_coarse,qp, 1)
    fine_ECSIM = Solvers.ECSIM(Np,Nx,Nsub_fine,dt_fine,qp,1)
    Xn = np.hstack((np.reshape(xp,(Np*xdim)),np.reshape(vp,(Np*vdim)),np.reshape(E0,(Nx*vdim)),np.reshape(Bc,(Nx*vdim))))
    # get into regime
    Xn = fine_ECSIM.Step(Xn,0,dt_fine)
    Ek,Ee,Eb = fine_ECSIM.Energy(Xn)
    Etot = Ek+Ee+Eb

    k_max = 5
    NT_max = 100

    state_errors = np.zeros((k_max, NT_max))
    energy_errors = np.zeros((k_max, NT_max))
    para = Solvers.PararealSolver(k_max, fine_ECSIM,coarse_ECSIM,wp_E,wp_P)
    for NT in range(1,NT_max):
        print(f"Simulating with fine dt = {dt_fine}, coarse dt = {dt_coarse} and {NT} steps.")
        parareal_histEnergy = np.empty((NT))

        time_start = time.perf_counter()
        parareal_test_solution = para.Solve(Xn,np.arange(NT+1)*dt_coarse)
        time_end = time.perf_counter()

        print(f"Parareal simulation of {NT} steps and {parareal_test_solution.shape[0]-1} iterations took {time_end-time_start:0.4f}s")
        

        for k in range(1,min(k_max+1,parareal_test_solution.shape[0])):
            parareal_histEnergy[:] = list(map(lambda x: np.sum(fine_ECSIM.Energy(x)), parareal_test_solution[k,:,:]))
            energy_errors[k-1,NT] = np.max(np.abs(parareal_histEnergy-Etot))/Etot
            state_errors[k-1,NT] = np.max(np.linalg.norm(parareal_test_solution[k,:,:]-parareal_test_solution[k-1,:,:],axis=1)/np.linalg.norm(parareal_test_solution[-k,:,:],axis=1))
            print(f"k = {k}, NT = {NT}: max state error = {state_errors[k-1,NT]}, max energy error = {energy_errors[k-1,NT]}")
        for k in range(parareal_test_solution.shape[0],k_max+1):
            energy_errors[k-1,NT] = 0
            state_errors[k-1,NT] = 0

    np.savetxt("state_error.txt", state_errors, fmt='%.6e', delimiter='\t')
    np.savetxt("energy_error.txt", energy_errors, fmt='%.6e', delimiter='\t')
    ax = plt.subplots(2,figsize=(8, 6), dpi=100)
    for k in range(1,1+k_max):
        ax[1][0].loglog(dt_coarse*np.arange(k+1,NT_max),state_errors[k-1,k+1:],label=f"k = {k}")
    ax[1][0].set_xlim(0, NT_max*dt_coarse)
    ax[1][0].set_xlabel('t')
    ax[1][0].set_ylabel('max(|X^k - X^(k-1)|_2/|X^k|_2)')
    ax[1][0].relim()
    ax[1][0].set_title(f"State Errors, F_dt = {dt_fine} G_dt = {dt_coarse}")
    plt.legend()

    for k in range(1,1+k_max):
        ax[1][1].loglog(dt_coarse*np.arange(k+1,NT_max),energy_errors[k-1,k+1:],label=f"k = {k}")
    ax[1][1].set_xlim(0, NT_max*dt_coarse)
    ax[1][1].set_xlabel('t')
    ax[1][1].set_ylabel('max(|E^k(t) - E^0(0)|/E^0(0))')
    ax[1][1].relim()
    ax[1][1].set_title(f"Energy Errors, F_dt = {dt_fine} G_dt = {dt_coarse}")
    plt.legend()
    plt.pause(0.1)
    plt.draw()
    plt.ioff()
    plt.show()


