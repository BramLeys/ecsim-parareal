import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import scipy as sp
import Solvers

if __name__ == '__main__':
    plt.ion()

    Nx=128 # number of grid cells
    L=2*math.pi # Size of position space
    T = 1
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

    dt = 1e-3
    dt_coarse = 1e-2

    NT = int(T/dt_coarse) +1
    print(f"Simulating with fine dt = {dt}, coarse dt = {dt_coarse} and {NT} steps.")

    Nsub_fine = Nsub
    Nsub_coarse = Nsub_fine
    max_para_iterations = 50

    wp_E = 1e-15
    wp_P = 1e-15

    coarse_ECSIM = Solvers.ECSIM(Np,Nx,Nsub_coarse,dt_coarse,qp, 1)
    fine_ECSIM = Solvers.ECSIM(Np,Nx,Nsub_fine,dt,qp,1)
    para = Solvers.PararealSolver(max_para_iterations, fine_ECSIM,coarse_ECSIM,wp_E,wp_P)
    Xn = np.hstack((np.reshape(xp,(Np*xdim)),np.reshape(vp,(Np*vdim)),np.reshape(E0,(Nx*vdim)),np.reshape(Bc,(Nx*vdim))))

    # get into regime
    Xn = fine_ECSIM.Step(Xn,0,dt)
    Ek,Ee,Eb = fine_ECSIM.Energy(Xn)

    fine_test_solution = np.empty((NT,Xn.shape[0]))
    fine_histEnergy = np.empty((NT))
    fine_histEnergy[0] = Ek + Ee + Eb

    coarse_test_solution = np.empty((NT,Xn.shape[0]))
    coarse_histEnergy = np.empty((NT))
    coarse_histEnergy[0] = Ek + Ee + Eb

    parareal_test_solution = np.empty((NT,Xn.shape[0]))
    parareal_histEnergy = np.empty((NT))
    parareal_histEnergy[0] = Ek + Ee + Eb

    ax = plt.subplots(3,figsize=(8, 6), dpi=100)
    
    time_start = time.perf_counter()
    fine_test_solution = np.empty((NT,Xn.shape[0]))
    fine_test_solution[0,:] = Xn
    for it in range(1,NT):
        fine_test_solution[it,:] = fine_ECSIM.Step(fine_test_solution[it-1,:],dt_coarse*(it-1), dt_coarse*(it))
        fine_histEnergy[it] = np.sum(fine_ECSIM.Energy(fine_test_solution[it,:]))
    time_end = time.perf_counter()
    print(f"Fine simulation of {NT} steps took {time_end-time_start:0.4f}s")

    ax[1][0].plot(np.arange(NT)*dt_coarse,abs((fine_histEnergy[:] - fine_histEnergy[0])/fine_histEnergy[0]),label=f"Difference in total energy fine ECSIM dt = {dt}")
    ax[1][0].set_xlim(0, (NT-1)*dt_coarse)
    ax[1][0].set_xlabel('t')
    ax[1][0].set_ylabel('|E(t) - E(0)|/E(0)')
    ax[1][0].set_title(f"Fine simulation, dt = {dt}")
    ax[1][0].relim()
    ax[1][0].autoscale_view(None,False,True)
    plt.pause(0.1)
    plt.draw()

    time_start = time.perf_counter()
    coarse_test_solution = np.empty((NT,Xn.shape[0]))
    coarse_test_solution[0,:] = Xn
    for it in range(1,NT):
        coarse_test_solution[it,:] = coarse_ECSIM.Step(coarse_test_solution[it-1,:],dt_coarse*(it-1), dt_coarse*(it))
        coarse_histEnergy[it] = np.sum(coarse_ECSIM.Energy(coarse_test_solution[it,:]))
    time_end = time.perf_counter()
    print(f"Coarse simulation of {NT} steps took {time_end-time_start:0.4f}s")
    ax[1][1].plot(np.arange(NT)*dt_coarse,abs((coarse_histEnergy[:] - coarse_histEnergy[0])/coarse_histEnergy[0]),label=f"Difference in total energy coarse ECSIM dt = {dt_coarse}")
    ax[1][1].set_xlim(0, (NT-1)*dt_coarse)
    ax[1][1].set_xlabel('t')
    ax[1][1].set_ylabel('|E(t) - E(0)|/E(0)')
    ax[1][1].set_title(f"Coarse simulation, dt = {dt_coarse}")
    ax[1][1].relim()
    ax[1][1].autoscale_view(None,False,True)
    plt.pause(0.1)
    plt.draw()
    

    time_start = time.perf_counter()
    parareal_test_solution = para.Solve(Xn,np.arange(NT)*dt_coarse)[-1,:,:]
    time_end = time.perf_counter()
    print(f"Parareal simulation of {NT} steps took {time_end-time_start:0.4f}s")
    for it in range(1,NT):
        parareal_histEnergy[it] = np.sum(fine_ECSIM.Energy(parareal_test_solution[it,:]))

    ax[1][2].plot(np.arange(NT)*dt_coarse,abs((parareal_histEnergy[:] - parareal_histEnergy[0])/parareal_histEnergy[0]),label=f"Difference in total energy fine ECSIM F_dt = {dt}, G_dt = {dt_coarse}")
    ax[1][2].set_xlim(0, (NT-1)*dt_coarse)
    ax[1][2].set_xlabel('t')
    ax[1][2].set_ylabel('|E(t) - E(0)|/E(0)')
    ax[1][2].relim()
    ax[1][2].set_title(f"Parareal simulation, F_dt = {dt} G_dt = {dt_coarse}")
    ax[1][2].autoscale_view(None,False,True)
    plt.pause(0.1)
    plt.draw()

    print(f"Coarse relative L1 Error = {np.sum(abs(coarse_test_solution-fine_test_solution))/np.sum(abs(fine_test_solution))}\
                    L2 Error = {np.linalg.norm(coarse_test_solution-fine_test_solution)/np.linalg.norm(fine_test_solution)}\
                    Linf Error = {np.max(abs(coarse_test_solution-fine_test_solution))/np.max(abs(fine_test_solution))}")
    
    print(f"Parareal relative L1 Error = {np.sum(abs(parareal_test_solution-fine_test_solution))/np.sum(abs(fine_test_solution))}\
                    L2 Error = {np.linalg.norm(parareal_test_solution-fine_test_solution)/np.linalg.norm(fine_test_solution)}\
                    Linf Error = {np.max(abs(parareal_test_solution-fine_test_solution))/np.max(abs(fine_test_solution))}")
    plt.ioff()
    plt.show()


