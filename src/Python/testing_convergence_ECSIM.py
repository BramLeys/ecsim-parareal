import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
m = 1
q = -1
L = 2*math.pi
E_func = lambda t: np.array([-m/q*math.sin(t) + m/(2*q)*math.cos(t), -m/(2*q)*math.sin(t), -m/(2*q)*math.sin(t) - 3*m/(2*q)*math.cos(t)]).reshape((1,3))
B_func = lambda t: np.array([-m/(2*q), m/(2*q),m/(2*q)]).reshape((1,3))
analytical_x = lambda t: np.array([math.sin(t)]).reshape((1,1))
analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)]).reshape((1,3))

k = 3
omega = k
analytical_E = lambda x,t: np.array([0,math.cos(omega*t)*math.sin(k*x),0]).reshape((1,3))
analytical_B = lambda x,t: np.array([0,0, -math.sin(omega*t)*math.cos(k*x)]).reshape((1,3))

t0 = 0
t_end = 1
Np = 1
Nx_base = 10
Nt_base = 10
refinements = 5

def alpha(B,dt):
    beta = q/m*dt/2
    sx=B[:,0]*beta;sy=B[:,1]*beta;sz=B[:,2]*beta
    return np.transpose((np.array([[1+sx*sx,sz+sx*sy,-sy+sx*sz],\
                                    [-sz+sx*sy,1+sy*sy,sx+sy*sz],\
                                    [sy+sx*sz,-sx+sy*sz,1+sz*sz]])/(1+sx*sx+sy*sy+sz*sz)).T,axes=(0, 2, 1))

def DecoupledStep3D(yn,tn,tn1,Nx, dt,dx, return_all_steps = False ):
        xp = np.copy(np.reshape(yn[:Np],(Np)))
        vp = np.copy(np.reshape(yn[Np:Np+3*Np], (Np,3)))
        E0 = np.copy(np.reshape(yn[Np+3*Np:Np+3*(Np + Nx)], (Nx,3)))
        Bc = np.copy(np.reshape(yn[Np+3*(Np + Nx):Np+3*(Np + 2*Nx)], (Nx,3)))
        
        nb_steps = math.ceil(abs((tn1-tn)/dt))
        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        theta = 0.5
        for n in np.arange(1,nb_steps+1):
            t = tn + dt*(n-1) # t at beginning of time interval

            J0 = np.zeros((Nx,3))
            M=np.zeros((Nx,Nx,3,3))
            alphap = alpha(B_func(t),dt)

            A = q/m*dt**2*theta/2*M/dx
            I = np.eye(Nx)
            Derv= np.eye(Nx) + np.diagflat(np.ones(Nx-1),-1); Derv[0,-1] = -1; Derv/=dx
            Derc= -np.eye(Nx) + np.diagflat(np.ones(Nx-1),1); Derc[-1,0] = 1; Derc/=dx

            bKrylov = np.hstack((E0[:,0] - J0[:,0]*dt*theta, E0[:,1]-(J0[:,1])*dt*theta, E0[:,2]-(J0[:,2])*dt*theta,Bc[:,1],Bc[:,2]))

            Maxwell=np.vstack((np.hstack((I + A[:,:,0,0], A[:,:,0,1], A[:,:,0,2],  np.zeros((Nx,2*Nx)))),\
                np.hstack((A[:,:,1,0], I + A[:,:,1,1], A[:,:,1,2],np.zeros((Nx,Nx)), Derv*dt*theta)),\
                np.hstack((A[:,:,2,0],  A[:,:,2,1], I+A[:,:,2,2],  -Derv*dt*theta,  np.zeros((Nx,Nx)))),\
                np.hstack((np.zeros((Nx,2*Nx)),   -Derc*dt*theta, I, np.zeros((Nx,Nx)))),\
                np.hstack((np.zeros((Nx,Nx)),Derc*dt*theta, np.zeros((Nx,2*Nx)), I))))
            
            xKrylov=sp.sparse.linalg.spsolve(sp.sparse.csr_array(Maxwell),bKrylov)

            E12 = np.reshape((xKrylov[:3*Nx]),(Nx,3),"F")
            Bc[:,1]=(xKrylov[3*Nx:4*Nx]-Bc[:,1]*(1-theta))/theta
            Bc[:,2]=(xKrylov[4*Nx:5*Nx]-Bc[:,2]*(1-theta))/theta

            E0 = (E12-E0*(1-theta))/theta # $E^{n+\theta} = \theta*E^{n+1} + (1-\theta)*E^{n}$

            xp += vp[:,0]*dt
            vp[:,:] = np.einsum("ijk,ik->ij",(2*alphap-np.eye(3)[None,:,:]),vp) + dt*q/m*np.einsum("ijk,ik->ij",alphap,(E_func(t+0.5*dt)))
            x[n,:] = np.hstack((np.reshape(xp[:],(Np*1)),np.reshape(vp,(Np*3)),np.reshape(E0,(Nx*3)),np.reshape(Bc,(Nx*3))))
        if return_all_steps:
            return x
        return x[-1,:]


errors = np.empty((refinements,4))
Nx = 10
dimension = 4*Np + 6*Nx
dx = L/Nx
E0 = np.empty((Nx,3))
Bc = np.empty((Nx,3))
E_analyt = np.empty((Nx,3))
Bc_analyt = np.empty((Nx,3))
for j in range(Nx):
    x = j*dx
    E0[j,:] = analytical_E(x,t0)
    Bc[j,:] = analytical_B(x+0.5*dx,t0)
    E_analyt[j,:] = analytical_E(x,t_end)
    Bc_analyt[j,:] = analytical_B(x+0.5*dx,t_end)
print("E0: ", E0)
print("Bc", Bc)
print("E_end: ", E_analyt)
print("B_end: ", Bc_analyt)
for i in range(refinements):
    Nt = Nt_base * 2**i
    ts = np.linspace(t0,t_end,Nt+1)
    dt = (t_end-t0)/Nt
    xp = analytical_x(-dt/2)
    vp = analytical_v(0)
    print("xp: ", xp)
    print("vp", vp)
    print("x_end: ",analytical_x(t_end-dt/2))
    print("v_end: ", analytical_v(t_end))
    Xn = np.hstack((np.reshape(xp,(Np*1)),np.reshape(vp,(Np*3)),np.reshape(E0,(Nx*3)),np.reshape(Bc,(Nx*3))))
    Yn = DecoupledStep3D(Xn, t0,t_end, Nx,dt,dx)
    errors[i,:] = np.array([np.linalg.norm(analytical_x(t_end-dt/2)-Yn[:Np].reshape((Np,1)))/np.linalg.norm(analytical_x(t_end-dt/2)), 
                            np.linalg.norm(analytical_v(t_end)-Yn[Np:4*Np].reshape((Np,3)))/np.linalg.norm(analytical_v(t_end)), 
                            np.linalg.norm(E_analyt-Yn[4*Np:4*Np + 3*Nx].reshape((Nx,3)))/np.linalg.norm(E_analyt), 
                            np.linalg.norm(Bc_analyt-Yn[4*Np + 3*Nx:].reshape((Nx,3)))/np.linalg.norm(Bc_analyt)])
    if (i > 0):
        print(f' time convergence: {errors[i,:]/errors[i-1,:]}')
print(f"time convergence errors = {errors}")

fig = plt.figure()
plt.plot(np.linspace(0,L,Nx),Bc_analyt[:,2], label="analyic")
plt.plot(np.linspace(0,L,Nx),Yn[4*Np + 3*Nx:].reshape((Nx,3))[:,2], label="calculated")
plt.legend()

plt.show()
fig2 = plt.figure()
plt.plot(np.linspace(0,L,Nx),E_analyt[:,1], label="analyic")
plt.plot(np.linspace(0,L,Nx),Yn[4*Np:4*Np + 3*Nx].reshape((Nx,3))[:,1], label="calculated")
plt.legend()
plt.show()

Nt = 100
ts = np.linspace(t0,t_end,Nt+1)
xp = analytical_x(-dt/2)
vp = analytical_v(0)
dt = (t_end-t0)/Nt
for i in range(refinements):
    Nx = Nx_base * 2**i
    dx = L/Nx
    dimension = 4*Np + 6*Nx
    E0 = np.empty((Nx,3))
    Bc = np.empty((Nx,3))
    E_analyt = np.empty((Nx,3))
    Bc_analyt = np.empty((Nx,3))
    for j in range(Nx):
        x = j*dx
        E0[j,:] = analytical_E(x,t0)
        Bc[j,:] = analytical_B(x+0.5*dx,t0)
        E_analyt[j,:] = analytical_E(x,t_end)
        Bc_analyt[j,:] = analytical_B(x+0.5*dx,t_end)
    Xn = np.hstack((np.reshape(xp,(Np*1)),np.reshape(vp,(Np*3)),np.reshape(E0,(Nx*3)),np.reshape(Bc,(Nx*3))))
    Yn = DecoupledStep3D(Xn, t0,t_end, Nx,dt,dx)
    errors[i,:] = np.array([np.linalg.norm(analytical_x(t_end-dt/2)-Yn[:Np].reshape((Np,1)))/np.linalg.norm(analytical_x(t_end-dt/2)), 
                            np.linalg.norm(analytical_v(t_end)-Yn[Np:4*Np].reshape((Np,3)))/np.linalg.norm(analytical_v(t_end)), 
                            np.linalg.norm(E_analyt-Yn[4*Np:4*Np + 3*Nx].reshape((Nx,3)))/np.linalg.norm(E_analyt), 
                            np.linalg.norm(Bc_analyt-Yn[4*Np + 3*Nx:].reshape((Nx,3)))/np.linalg.norm(Bc_analyt)])
    if (i > 0):
        print(f' space convergence: {errors[i,:]/errors[i-1,:]}')
print(f"space convergence errors = {errors}")

