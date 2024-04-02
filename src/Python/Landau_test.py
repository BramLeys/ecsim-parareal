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
    NT=100 # number of time steps
    Np=10000 # number of particles
    NTOUT=NT//2 # How many times should updates be shown
    dt=0.125 #time step length
    dx=L/Nx # length of each grid cell
    Nsub = 1

    qom=-1 # charge mass density of particles q/m

    theta=0.5 #theta of field and particle mover
    
    f = 5
    omega = f*2*math.pi
    k = 2*math.pi

    xv=np.linspace(0,L,Nx+1)[:-1] # position of each of the grid cell vertices
    xc=.5*(xv[1:]+xv[:-1]) # position of each of the grid cell centers
    Vx=dx*np.ones(Nx) # volumes of each of the grid cells
    xdim = 1
    vdim = 1
    E0 = 0.1*np.sin(k*xv).reshape((Nx,vdim))#initialization of electric field in each of the grid cells

    Bc = np.zeros((Nx,vdim))#initialization of magnetic field in each of the grid cell centers
    Bc[:,0] = np.ones(Nx)/10

    VT = np.ones(vdim)*0.02 # thermic velocity of particles in each direction
    V0 = omega/k

    xp = np.random.uniform(0,L-L/Np,Np)
    vp = np.random.normal(V0, 0.1, Np)

    ix=np.floor(xp/dx).astype(int); # cell of the particle, first cell is cell 0, first node is 0 last node Nx
    frac1 = 1-(xp/dx-ix)
    ix2=(ix+1)%Nx # second cell of influence due to extended local support of b-spline


    M=np.zeros((Nx,Nx))
    for ip in range(Np):
        M[ix[ip],ix[ip]] += frac1[ip]**2/2; # divided by 2 for the symmetry added later
        M[ix2[ip],ix[ip]] += frac1[ip]*(1-frac1[ip])
        M[ix2[ip],ix2[ip]] += (1-frac1[ip])**2/2
    M=M+M.T

    rhotarget=-Vx
    rhotildeV=sp.sparse.linalg.spsolve(sp.sparse.csr_array(M),rhotarget)

    qp = np.empty(Np)
    for ip in range(Np):
        qp[ip]=rhotildeV[ix[ip]]*frac1[ip]+rhotildeV[ix2[ip]]*(1-frac1[ip])

    histEnergy = np.empty((NT))

    dt_coarse = dt
    dt_fine = dt_coarse/2
    Nsub_fine = Nsub
    Nsub_coarse = Nsub_fine
    max_para_iterations = 50
    wp_E = 1e-15
    wp_P = 1e-15

    coarse_ECSIM = Solvers.ECSIM(Np,Nx,Nsub_coarse,dt_coarse,qp, 1)
    fine_ECSIM = Solvers.ECSIM(Np,Nx,Nsub_fine,dt_fine,qp,1)
    para = Solvers.PararealSolver(max_para_iterations, fine_ECSIM,coarse_ECSIM,wp_E,wp_P)
    Xn = np.hstack((np.reshape(xp,(Np*xdim)),np.reshape(vp,(Np*vdim)),np.reshape(E0,(Nx*vdim)),np.reshape(Bc,(Nx*vdim))))
    Xn = fine_ECSIM.Step(Xn,0,dt_fine)
    Ek,Ee,Eb = fine_ECSIM.Energy(Xn)

    # vps = np.empty((NT+1,Np))
    # Es = np.empty((NT+1,Nx))
    # vps[0,:] = np.reshape(Xn[xdim*Np:xdim*Np+vdim*Np], Np)
    # Es[0,:] = np.reshape(Xn[xdim*Np+vdim*Np:xdim*Np+vdim*(Np + Nx)], Nx)
    # fft_E = np.fft.fft(Es, axis=1)
    # # Compute the frequencies
    # sampling_freq = 1 / dx  # Sampling frequency
    # freqs = np.fft.fftfreq(Nx, 1 / sampling_freq)
    # # Plot the FFT output
    # fig,ax = plt.subplots(1,figsize=(8, 6))
    # line, = ax.plot(freqs, np.abs(fft_E[0]))
    # plt.vlines([f,-f],0,20, "k")
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('FFT Electric field')
    # plt.grid(True)
    # plt.pause(0.1)
    # plt.draw()
    # time_start = time.perf_counter()
    # _,X1 = fine_ECSIM.Step(Xn,0,NT*dt_coarse, True)
    # time_end = time.perf_counter()
    # print(f"Serial simulation of {NT} steps took {time_end-time_start:0.4f}s")
    # for i in range(1,NT+1):
    #     vps[i,:] = np.reshape(X1[i,xdim*Np:xdim*Np+vdim*Np], Np)
    #     Es[i,:] = np.reshape(X1[i,xdim*Np+vdim*Np:xdim*Np+vdim*(Np + Nx)], Nx)
    #     fft_E = np.fft.fft(Es[i,:])
    #     line.set_data(freqs, np.abs(fft_E))
    #     plt.pause(0.1)
    #     plt.draw()

    # fft_vp = np.fft.fft(vps, axis=0)
    # fft_vp = np.mean(fft_vp,axis=1)
    # # Compute the frequencies
    # sampling_freq = 1 / dt_coarse  # Sampling frequency
    # freqs = np.fft.fftfreq(NT+1, 1 / sampling_freq)
    # # Plot the FFT output
    # plt.figure(figsize=(8, 6))
    # plt.plot(freqs, np.abs(fft_vp))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('average FFT of velocity')
    # plt.grid(True)
    # plt.show()


    ax = plt.subplots(2,figsize=(8, 6), dpi=100)
    line, = ax[1][0].plot([],[],"b.")
    ax[1][0].set_xlim(0, L)
    line_Etot, = ax[1][1].semilogy([],[],label="difference in total energy")
    ax[1][1].set_xlim(0, NT*dt)
    X1 = Xn
    for it in range(NT):
            X1 = fine_ECSIM.Step(X1,dt*it, dt*(it+1))
            xp = np.reshape(X1[:xdim*Np],(Np))%L
            vp = np.reshape(X1[xdim*Np:xdim*Np+vdim*Np], (Np,vdim))
            histEnergy[it] = np.sum(fine_ECSIM.Energy(X1))
            if((it%round(NT/NTOUT)==0)):
                ax[1][0].set_title(f"Fine solution")
                line.set_data(xp,vp)
                ax[1][0].set_xlabel('x')
                ax[1][0].set_ylabel('vx')
                ax[1][0].relim()
                ax[1][0].autoscale_view(None,False,True)
                line_Etot.set_data(np.arange(it+1)*dt,abs((histEnergy[:it+1] - histEnergy[0])/histEnergy[0]))
                ax[1][1].set_xlabel('t')
                ax[1][1].set_ylabel('|E(t) - E(0)|/E(0)')
                ax[1][1].relim()
                ax[1][1].autoscale_view(None,False,True)
                plt.pause(0.01)
                plt.draw()


    time_start = time.perf_counter()
    X1 = para.Solve(Xn,np.arange(NT+1)*dt_coarse)
    time_end = time.perf_counter()
    print(f"Parallel simulation of {NT} steps took {time_end-time_start:0.4f}s")

    axes = []
    for k in range(X1.shape[0]):
        axes.append(plt.subplots(2,figsize=(8, 6), dpi=100))
        line, = axes[k][1][0].plot([],[],"b.")
        axes[k][1][0].set_xlim(0, L)
        line_Etot, = axes[k][1][1].semilogy([],[])
        axes[k][1][1].set_xlim(0, NT*dt_coarse)
        for it in range(NT):
            Xn = X1[k,it+1,:]
            # Xn = coarse_ECSIM.Step(Xn,dt*it, dt*(it+1))
            xp = np.reshape(Xn[:xdim*Np],(Np))%L
            vp = np.reshape(Xn[xdim*Np:xdim*Np+vdim*Np], (Np,vdim))
            histEnergy[it] = np.sum(fine_ECSIM.Energy(Xn))
            if((it%round(NT/NTOUT)==0)):
                axes[k][1][0].set_title(f"k = {k}, t = {(it+1)*dt_coarse}")
                line.set_data(xp,vp)
                axes[k][1][0].set_xlabel('x')
                axes[k][1][0].set_ylabel('vx')
                axes[k][1][0].relim()
                axes[k][1][0].autoscale_view(None,False,True)
                line_Etot.set_data(np.arange(it+1)*dt_coarse,abs((histEnergy[:it+1] - histEnergy[0])/histEnergy[0]))
                axes[k][1][1].set_xlabel('t')
                axes[k][1][1].set_ylabel('|E(t) - E(0)|/E(0)')
                axes[k][1][1].relim()
                axes[k][1][1].autoscale_view(None,False,True)
                plt.pause(0.01)
                plt.draw()
    plt.ioff()
    plt.show()

