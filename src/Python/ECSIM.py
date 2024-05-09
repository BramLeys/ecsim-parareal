import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy as sp

def alpha(beta,B):
    # I = np.eye(B.shape[0])
    # IxB = np.array([np.cross(I[0,:],B),np.cross(I[1,:],B),np.cross(I[2,:],B)])
    # return (I + beta*IxB + beta**2*np.outer(B,B))/(1+np.dot(beta*B,beta*B))
    sx=B[:,0]*beta;sy=B[:,1]*beta;sz=B[:,2]*beta
    return np.transpose((np.array([[1+sx*sx,sz+sx*sy,-sy+sx*sz],\
      [-sz+sx*sy,1+sy*sy,sx+sy*sz],\
      [sy+sx*sz,-sx+sy*sz,1+sz*sz]])/(1+sx*sx+sy*sy+sz*sz)).T,axes=(0, 2, 1))

plt.ion()
# plt.rc('text', usetex=True)
fig, (ax,ax2) = plt.subplots(2,figsize=(8, 6), dpi=100)
# Get the screen width and height
# screen_width = fig.canvas.manager.window.winfo_screenwidth()
# screen_height = fig.canvas.manager.window.winfo_screenheight()

# Calculate the center position
# center_x = int((screen_width - fig.get_size_inches()[0] * fig.get_dpi()) / 2)
# center_y = int((screen_height - fig.get_size_inches()[1] * fig.get_dpi()) / 2)
# # Set the window location to center
# fig.canvas.manager.window.wm_geometry(f"+{center_x}+{center_y}")
line_pos, = ax.plot([],[],"b.", label='vy > 0')
line_neg, = ax.plot([],[],"r.", label='vy < 0')
line_Etot, = ax2.plot([],[],label="difference in total energy")
ax.legend()
ax2.set_title("Difference in total energy from the begin state")


Nx=128 # number of grid cells
L=2*math.pi # Size of position space
T = 1
dt=1e-2 #time step length
NT=int(T/dt) # number of time steps
Np=10000 # number of particles
graphics = True # should graphics be shown
NTOUT=1 # How many times should updates be shown
dx=L/Nx # length of each grid cell
Nsub = 1

ax.set_xlim(0, L)
ax2.set_xlim(0, NT*dt)

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

Bc = np.zeros((Nx,vdim))#initialization of magnetic field in each of the grid cell centers
Bc[:,0] = np.ones(Nx)/10

# curlBv_y=np.zeros(Nx)
# curlBv_z=np.zeros(Nx)
# curlEc_y=np.zeros(Nx)
# curlEc_z=np.zeros(Nx)

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

Nsm=0
if Nsm>0:
    SM = np.diagflat(np.ones(Nx-1),-1) + np.diagflat(np.ones(Nx)*2,0) + np.diagflat(np.ones(Nx-1),1)
    SM[0,-1] = 1
    SM[-1,0] = 1
    SM /= 4
    SM=np.power(SM,Nsm)
else:
    SM=np.eye(Nx)

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
rhotildeV=sp.sparse.linalg.spsolve(M,rhotarget)

qp = np.empty(Np)
rho = np.zeros(Nx)
for ip in range(Np):
    qp[ip]=rhotildeV[ix[ip]]*frac1[ip]+rhotildeV[ix2[ip]]*(1-frac1[ip])
    rho[ix[ip]] += frac1[ip]*qp[ip]
    rho[ix2[ip]] += (1-frac1[ip])*qp[ip]

rho=rho/Vx

histEnergy = np.empty((NT))
histEnergyB = np.empty((NT))
histEnergyE = np.empty((NT))
histEnergyK = np.empty((NT))
histMomentum = np.empty((NT))
histFmode = np.empty(NT)
histxvel = np.empty((NT,Np))
Ee=0

spectrumE=np.empty((NT,Nx,vdim))
spectrumB=np.empty((NT,Nx,vdim))
time_matrix = 0
calculation_time = 0

for it in range(NT):
    tic = time.perf_counter()
    ix=np.floor(xp/dx).astype(int); # cell of the particle, first cell is cell 0, first node is 0 last node Nx
    frac1 = 1-(xp/dx-ix)
    ix2=(ix+1)%Nx # second cell of influence due to extended local support of b-spline

    B[1:,:]=.5*(Bc[1:,:]+Bc[:-1,:]); B[0,:]=.5*(Bc[-1,:]+Bc[0,:]) # calculate the vertex points of the magnetic field

    Bp = np.empty((Np,vdim))
    # Find magnetic field for each particle
    Bp[:,:]=B[ix,:]*frac1[:,None]+B[ix2,:]*(1-frac1[:,None])
    alphap = alpha(qom*dt/2,Bp)
    vphat = np.einsum("ijk,ik->ij",alphap,vp)
    J0 = np.zeros((Nx,vdim))
    for i in range(vdim):
        np.add.at(J0[:,i], (ix), frac1*qp*vphat[:,i])
        np.add.at(J0[:,i], (ix2), (1-frac1)*qp*vphat[:,i])
    J0/=Vx[:,None]
    #smoothing current
    # if Nsm > 0:
    #     J0 = SM@J0
    

    # curlBv_y[1:]=-(Bc[1:,2]-Bc[:-1,2])/dx;curlBv_y[0]=-(Bc[0,2]-Bc[-1,2])/dx
    # curlBv_z[1:]=(Bc[1:,1]-Bc[:-1,1])/dx;curlBv_z[0]=(Bc[0,1]-Bc[-1,1])/dx

    # curlEc_y[1:]=-(E0[1:,2]-E0[:-1,2])/dx;curlEc_y[0]=-(E0[0,2]-E0[-1,2])/dx
    # curlEc_z[1:]=(E0[1:,1]-E0[:-1,1])/dx;curlEc_z[0]=(E0[0,1]-E0[-1,1])/dx

    time_matrix_in  = time.perf_counter()
    M=np.zeros((Nx,Nx,vdim,vdim))
    for i in range(vdim):
            for j in range(vdim):
                np.add.at(M[:,:,i,j], (ix, ix), .5*frac1**2*qp*alphap[:,i,j])
                np.add.at(M[:,:,i,j], (ix2, ix), frac1*(1-frac1)*qp*alphap[:,i,j])
                np.add.at(M[:,:,i,j], (ix2, ix2), .5*(1-frac1)**2*qp*alphap[:,i,j])
                M[:,:,i,j]=M[:,:,i,j]+M[:,:,i,j].T
                # M[:,:,i,j]=SM@M[:,:,i,j]@SM

    time_matrix_out = time.perf_counter()
    
    time_matrix += (time_matrix_out-time_matrix_in)

    AmpereX = np.eye(Nx) + qom*dt**2*theta/2*M[:,:,0,0]/dx
    AmpereY = np.eye(Nx) + qom*dt**2*theta/2*M[:,:,1,1]/dx
    AmpereZ = np.eye(Nx) + qom*dt**2*theta/2*M[:,:,2,2]/dx

    Derv= np.eye(Nx) + np.diagflat(np.ones(Nx-1),-1); Derv[0,-1] = -1; Derv/=dx
    Derc= -np.eye(Nx) + np.diagflat(np.ones(Nx-1),1); Derc[-1,0] = 1; Derc/=dx

    Faraday = np.eye(Nx)

    bKrylov = np.hstack((E0[:,0] - J0[:,0]*dt*theta, E0[:,1]-(J0[:,1])*dt*theta, E0[:,2]-(J0[:,2])*dt*theta,Bc[:,1],Bc[:,2]))

    Maxwell=np.vstack((np.hstack((AmpereX, qom*dt**2*theta/2*(M[:,:,0,1])/dx, qom*dt**2*theta/2*(M[:,:,0,2])/dx,  np.zeros((Nx,2*Nx)))),\
        np.hstack((qom*dt**2*theta/2*M[:,:,1,0]/dx, AmpereY,  qom*dt**2*theta/2*M[:,:,1,2]/dx, np.zeros((Nx,Nx)), Derv*dt*theta)),\
        np.hstack((qom*dt**2*theta/2*M[:,:,2,0]/dx, qom*dt**2*theta/2*M[:,:,2,1]/dx, AmpereZ,  -Derv*dt*theta,  np.zeros((Nx,Nx)))),\
        np.hstack((np.zeros((Nx,2*Nx)),   -Derc*dt*theta, Faraday, np.zeros((Nx,Nx)))),\
        np.hstack((np.zeros((Nx,Nx)),Derc*dt*theta, np.zeros((Nx,2*Nx)), Faraday))))
    

    xKrylov=sp.sparse.linalg.spsolve(Maxwell,bKrylov)

    E12 = np.reshape((xKrylov[:vdim*Nx]),(Nx,vdim),"F")
    Bc[:,1]=(xKrylov[vdim*Nx:4*Nx]-Bc[:,1]*(1-theta))/theta
    Bc[:,2]=(xKrylov[4*Nx:5*Nx]-Bc[:,2]*(1-theta))/theta
    Eold=E0
    E0 = (E12-E0*(1-theta))/theta
    # E12sm = SM@E12
    E12sm = E12

    xp += vp[:,0]*dt
    xp = xp%L
    vp[:,:] = np.einsum("ijk,ik->ij",(2*alphap-np.eye(vdim)[None,:,:]),vp) + dt*qom*np.einsum("ijk,ik->ij",alphap,(E12sm[ix,:]*frac1[:,None] + E12sm[ix2,:]*(1-frac1[:,None])))
    
    Ek = 0.5*np.dot(qp,(np.sum(vp**2,axis=1)))/qom
    Ee = np.sum(E0[:,0]**2 + E0[:,1]**2 + E0[:,2]**2)*dx/2
    Eb = 0.5*sum(Bc[:,1]**2+Bc[:,2]**2)*dx
    Etot = Ek + Ee + Eb

    histEnergy[it] = Etot
    histEnergyE[it] = Ee
    histEnergyB[it] = Eb
    histEnergyK[it] = Ek
    histMomentum[it] = np.dot(qp,vp[:,0])
    spectrumE[it,:,:] = E0
    spectrumB[it,:,:] = Bc
    histxvel[it,:] = vp[:,0]
    f=np.fft.fft(Bc[:,2])
    histFmode[it] = abs(f[4])

    toc = time.perf_counter()
    calculation_time += toc-tic
    # print(f"Iteration {it} took {toc-tic:0.4f}s of which {time_matrix_out-time_matrix_in:0.4f}s were spent on mass matrices")

    if((it%round(NT/NTOUT)==0) and graphics):
        ii = np.where(vp[:,1]>0)[0]; jj = np.where(vp[:,1]<0)[0]
        ax.set_title(f"Phase space for iteration {it}, t = {it*dt}")
        line_pos.set_data(xp[ii],vp[ii,0])
        line_neg.set_data(xp[jj],vp[jj,0])
        ax.set_xlabel('x')
        ax.set_ylabel('vx')
        ax.relim()
        ax.autoscale_view(None,False,True)
        line_Etot.set_data(np.arange(it+1)*dt,abs(histEnergy[:it+1] - histEnergy[0]))
        ax2.set_xlabel('t')
        ax2.set_ylabel('|E(t) - E(0)|')
        ax2.relim()
        ax2.autoscale_view(None,False,True)
        plt.pause(0.001)
        plt.draw()
print("first energy difference: ", abs(histEnergy[1] - histEnergy[0]) )
print("last energy difference: ", abs(histEnergy[-1] - histEnergy[-2]) )
print('first energy diff of diff: ', abs(abs(histEnergy[2] - histEnergy[1]) - abs(histEnergy[1] - histEnergy[0])))
print('last energy diff of diff: ', abs(abs(histEnergy[-1] - histEnergy[-2]) - abs(histEnergy[1] - histEnergy[0]))/ abs(histEnergy[1] - histEnergy[0]))
print(f"Entire simulation of {NT*dt}s took {calculation_time:0.4f}s of calculation of which the mass matrices calculations took {time_matrix:0.4f}s")
plt.ioff()
plt.show()