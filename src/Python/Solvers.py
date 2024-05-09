import numpy as np
import math
import scipy as sp
import multiprocessing
import time
import inspect

class BorisSolver:
    def __init__(self, t_stop, nb_nodes, Efun, Bfun, t_start=0):
        self.q = -1.602176634e-19
        self.m = 9.1093837015e-31
        self.t_end = t_stop
        self.t0 = t_start
        self.E = Efun
        self.B = Bfun
        self.dimension = 3
        self.SetN(nb_nodes=nb_nodes)
    
    def SetN(self,nb_nodes):
        self.N = nb_nodes
        self.dt = (self.t_end-self.t0)/self.N
        self.x = np.empty((self.dimension,self.N+1))
        self.v = np.empty([self.dimension,self.N+1])
        self.t = np.linspace(self.t0,self.t_end,self.N+1)

    def Solve(self, x0=None, v0=None):
        if x0 is None:
            x0 = np.zeros(self.dimension)
        if v0 is None:
            v0 = np.zeros(self.dimension)
        self.x[:,0] = x0
        self.v[:,0] = v0
        self.v[:,0] = self.UpdateVelocity(self.x[:,0],self.v[:,0], self.t0,-0.5*self.dt)
        for n in range(1,self.N+1):
            self.x[:,n],self.v[:,n] = self.Step(self.x[:,n-1],self.v[:,n-1], self.t[n-1], self.t[n])

    def Step(self,xn,vn,tn,tn1):
        nb_steps = math.ceil(abs((tn1-tn)/self.dt))
        if nb_steps == 1:
            vn1 = self.UpdateVelocity(xn,vn, tn,self.dt)
            # in case the time range is smaller than the defined timestep in the initializer
            interpolation_factor = (tn1-tn)/self.dt
            return xn + interpolation_factor*(self.UpdatePosition(xn,vn1, tn,self.dt)-xn),vn + interpolation_factor*(vn1-vn)
        else:
        # in case time range is larger than the timestep defined in the initializer
            ts = np.array([tn + i*self.dt for i in range(nb_steps+1)])
            x = np.empty((self.dimension, nb_steps+1))
            x[:,0] = xn
            v = np.empty((self.dimension, nb_steps+1))
            v[:,0] = vn
            for i in range(1,nb_steps+1):
                v[:,i] = self.UpdateVelocity(x[:,i-1], v[:,i-1], ts[i-1], self.dt)
                x[:,i] = self.UpdatePosition(x[:,i-1], v[:,i], ts[i-1],self.dt)
            # interpolate 
            interpolation_factor = (tn1-tn)/self.dt - int((tn1-tn)/self.dt)
            if interpolation_factor == 0.0:
                interpolation_factor = 1
            return x[:,-2] + interpolation_factor*(x[:,-1]-x[:,-2]), v[:,-2] + interpolation_factor*(v[:,-1]-v[:,-2])

    def UpdateVelocity(self,xn,vn,tn, dtn):
        Bn = self.B(xn,tn)
        En = self.E(xn,tn)
        ksi = self.q*dtn/(2*self.m)
        h = ksi*Bn
        s = 2*h/(1+h**2)
        u = vn + ksi*En
        return u + np.cross((u + np.cross(u,h)),s) + ksi*En

    def UpdatePosition(self,xn,vn,tn, dtn):
        return xn + dtn*vn
    
class RK2:
    def __init__(self,f,dt):
        self.dt = dt
        self.f = f
    def Step(self,yn,tn,tn1,return_all_steps = False):
        nb_steps = math.ceil(abs((tn1-tn)/self.dt))
        ts = np.array([tn + i*self.dt for i in range(nb_steps+1)])
        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        for i in range(1,nb_steps+1):
            k1 = self.f(ts[i-1],x[i-1])
            k2 = self.f(ts[i-1] + 0.5*self.dt, x[i-1]+ 0.5*k1*self.dt)
            x[i,:] = x[i-1,:] + self.dt*(k2)

        # if interpolation factor is not a whole integer, then the method breaks down to first order
        interpolation_factor = (tn1-tn)/self.dt - int((tn1-tn)/self.dt)
        if interpolation_factor == 0.0:
            interpolation_factor = 1
        if return_all_steps:
            x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:]), x
        return x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:])

class ForwardEuler:
    def __init__(self,f,dt):
        self.dt = dt
        self.f = f
    def Step(self,yn,tn,tn1,return_all_steps = False):
        nb_steps = math.ceil(abs((tn1-tn)/self.dt))
        ts = np.array([tn + i*self.dt for i in range(nb_steps+1)])
        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        for i in range(1,nb_steps+1):
            x[i,:] = x[i-1,:] + self.dt*self.f(ts[i-1],x[i-1])
        # if interpolation factor is not a whole integer, then the method breaks down to first order
        interpolation_factor = (tn1-tn)/self.dt - int((tn1-tn)/self.dt)
        if interpolation_factor == 0.0:
            interpolation_factor = 1
        if return_all_steps:
            x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:]), x
        return x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:])

class SymplecticEuler:
    def __init__(self,f,g,dt):
        self.dt = dt
        self.f = f
        self.g = g
    def Step(self,qn, pn,tn,tn1):
        nb_steps = math.ceil(abs((tn1-tn)/self.dt))
        ts = np.array([tn + i*self.dt for i in range(nb_steps+1)])
        q = np.empty((nb_steps+1,qn.shape[0]))
        p = np.empty((nb_steps+1,pn.shape[0]))
        q[0,:] = qn
        p[0,:] = pn
        for i in range(1,nb_steps+1):
            q[i,:] = q[i-1,:] + self.dt*self.g(ts[i-1],p[i-1])
            p[i,:] = p[i-1,:] + self.dt*self.f(ts[i-1],q[i,:])

        # if interpolation factor is not a whole integer, then the method breaks down to first order
        interpolation_factor = (tn1-tn)/self.dt - int((tn1-tn)/self.dt)
        if interpolation_factor == 0.0:
            interpolation_factor = 1
        return (q[-2,:] + interpolation_factor*(q[-1,:]-q[-2,:]),p[-2,:] + interpolation_factor*(p[-1,:]-p[-2,:])), (q,p)
    
class RK4:
    def __init__(self,f,dt):
        self.dt = dt
        self.f = f
    def Step(self,yn,tn,tn1,return_all_steps = False):
        nb_steps = round(abs((tn1-tn)/self.dt))
        ts = np.array([tn + i*self.dt for i in range(nb_steps+1)])
        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        for i in range(1,nb_steps+1):
            k1 = self.f(ts[i-1],x[i-1,:])
            k2 = self.f(ts[i-1] + 0.5*self.dt, x[i-1,:]+ 0.5*k1*self.dt)
            k3 = self.f(ts[i-1] + 0.5*self.dt, x[i-1,:]+ (0.5*k2)*self.dt)
            k4 = self.f(ts[i-1] + self.dt, x[i-1,:]+ k3*self.dt)
            x[i,:] = x[i-1,:] + self.dt*(k1/6 + k2/3 + k3/3 + k4/6)

        if return_all_steps:
            x
        return x[-1,:]
    
class CrankNicholson:
    def __init__(self,A,dt,thresh=1e-10):
        self.dt = dt
        self.A = A
        self.thresh= thresh
    def Step(self,yn,tn,tn1,return_all_steps = False):
        I = np.identity(self.A.shape[0])
        nb_steps = round(abs((tn1-tn)/self.dt))
        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        for i in range(1,nb_steps+1):
            x[i,:],_ = sp.sparse.linalg.bicgstab(sp.sparse.csr_array(I - self.dt * 0.5 * self.A),x[i-1,:] + 0.5 * self.dt * self.A @ x[i-1,:],tol=self.thresh)
        if return_all_steps:
            return x
        return x[-1,:]

def watch(x):
    caller_name = "x"
    for frame in inspect.stack():
        frame_locals = frame[0].f_locals
        for var_name, var in frame_locals.items():
            if np.array_equal(var, x):
                caller_name = var_name    
    # Compute the norms
    norm_1 = np.sum(np.abs(x))
    norm_2 = np.linalg.norm(x)
    norm_inf = np.max(np.abs(x))
    
    # Print the norms and variable name
    print(f"Norm-1 of {caller_name}: {norm_1} Norm-2 of {caller_name}: {norm_2}   Norm-Inf of {caller_name}: {norm_inf}")

class ECSIM:
    def __init__(self,Np,Nx,Nsub, dt, qp, vdim=3):
        self.xdim = 1 #only xdim = 1 supported currently
        self.vdim = vdim # dimension of velocity/Magnetic field
        self.L = math.pi*2 # length of position space
        self.Np=Np # number of particles
        self.Nx = Nx # number of grid cells
        self.Nsub = Nsub # number of subcycles for each timestep
        self.dx=self.L/self.Nx # length of each grid cell
        self.dt = dt #time step length
        self.qom = -1 #charge mass density of particles (q/m)
        self.theta = 0.5 #theta of field and particle mover
        # self.Nsm=0 #smoothing factor
        # if self.Nsm>0:
        #     self.SM = np.diagflat(np.ones(self.Nx-1),-1) + np.diagflat(np.ones(self.Nx)*2,0) + np.diagflat(np.ones(self.Nx-1),1)
        #     self.SM[0,-1] = 1
        #     self.SM[-1,0] = 1
        #     self.SM /= 4
        #     self.SM=np.power(self.SM,self.Nsm)
        # else:
        #     self.SM=np.eye(self.Nx)
        self.qp = qp # charge of each particle
        self.Vx = self.dx*np.ones(self.Nx) # volume of grid (only 1D space supported)
        if (vdim == 1):
            self.Step = self.Step1D
        else:
            self.Step = self.Step3D
    
    # Calculates the kinetic, electric and magnetic energy of state Xn
    def Energy(self,Xn):
        vp = np.reshape(Xn[self.xdim*self.Np:self.xdim*self.Np+self.vdim*self.Np], (self.Np,self.vdim))
        E0 = np.reshape(Xn[self.xdim*self.Np+self.vdim*self.Np:self.xdim*self.Np+self.vdim*(self.Np + self.Nx)], (self.Nx,self.vdim))
        Bc = np.reshape(Xn[self.xdim*self.Np+self.vdim*(self.Np + self.Nx):self.xdim*self.Np+self.vdim*(self.Np + 2*self.Nx)], (self.Nx,self.vdim))
        Ek = 0.5*np.dot(self.qp,(np.sum(vp**2,axis=1)))/self.qom
        Ee = 0.5*np.sum(np.linalg.norm(E0,axis=1)**2)*self.dx
        Eb = 0.5*sum(np.linalg.norm(Bc,axis=1)**2)*self.dx
        return np.array((Ek,Ee,Eb))
    
    # calculates the rotation matrix alpha for a given magnetic field discretization
    def alpha(self,B):
        # I = np.eye(B.shape[0])
        # IxB = np.array([np.cross(I[0,:],B),np.cross(I[1,:],B),np.cross(I[2,:],B)])
        # return (I + beta*IxB + beta**2*np.outer(B,B))/(1+np.dot(beta*B,beta*B))
        # it is faster to just define the matrix than to calculate it
        beta = self.qom*self.dt/2
        sx=B[:,0]*beta;sy=B[:,1]*beta;sz=B[:,2]*beta
        return np.transpose((np.array([[1+sx*sx,sz+sx*sy,-sy+sx*sz],\
                                        [-sz+sx*sy,1+sy*sy,sx+sy*sz],\
                                        [sy+sx*sz,-sx+sy*sz,1+sz*sz]])/(1+sx*sx+sy*sy+sz*sz)).T,axes=(0, 2, 1))
    
    #yn = [xp,vp,E0,Bc] where xp = Np x 1, vp = Np x 1, E0 = Nx x 1, Bc = Nx x 1
    def Step1D(self,yn,tn,tn1,return_all_steps = False):
        xi = np.copy(np.reshape(yn[:self.xdim*self.Np],(self.Np)))
        vp = np.copy(np.reshape(yn[self.xdim*self.Np:self.xdim*self.Np+self.vdim*self.Np], (self.Np,self.vdim)))
        E0 = np.copy(np.reshape(yn[self.xdim*self.Np+self.vdim*self.Np:self.xdim*self.Np+self.vdim*(self.Np + self.Nx)], (self.Nx,self.vdim)))
        Bc = np.copy(np.reshape(yn[self.xdim*self.Np+self.vdim*(self.Np + self.Nx):self.xdim*self.Np+self.vdim*(self.Np + 2*self.Nx)], (self.Nx,self.vdim)))

        # Bookkeeping the original energy in the system
        Ekold, Eeold, Ebold = self.Energy(yn)
        
        # calculate how many steps are required to hit the desired end time tn1
        nb_steps = math.ceil(abs((tn1-tn)/(self.dt*self.Nsub)))

        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        xp = np.tile(xi,(self.Nsub,1))
        for n in np.arange(1,nb_steps+1):
            print(f"----------------iteration {n}----------------")
            # watch(xp)
            xp += vp[:,0]*self.dt/self.Nsub*np.arange(1,self.Nsub+1)[:,None]
            # watch(xp)
            # Position is periodic with period self.L (parareal can go out of bounds between solves)
            xp_view = xp%self.L
            # watch(xp_view)
            
            ix=np.floor(xp_view/self.dx).astype(int); # cell of the particle, first cell is cell 0, first node is 0 last node Nx
            frac1 = 1-(xp_view/self.dx-ix) # Wpg
            ix2=(ix+1)%self.Nx # second cell of influence due to extended local support of first order b-spline

            # watch(ix)
            # watch(frac1)
            # watch(ix2)

            fraction_p = np.zeros((self.Np,self.Nx))
            for itsub in range(self.Nsub):
                mask_ix = np.arange(self.Nx) == ix[itsub, :, np.newaxis]
                mask_ix2 = np.arange(self.Nx) == ix2[itsub, :, np.newaxis]
                fraction_p[mask_ix] += frac1[itsub, :] / self.Nsub #influence of current gridcell
                fraction_p[mask_ix2] += (1 - frac1[itsub, :]) / self.Nsub #influence of previous gridcell due to first order b-spline
            
            J0 = np.zeros((self.Nx,self.vdim))
        
            # np.add.at(J0[:,0], (ix), frac1*self.qp*vp[:,0])
            # np.add.at(J0[:,0], (ix2), (1-frac1)*self.qp*vp[:,0])

            # for ip in range(self.Np):
            #     J0[:,0] += self.qp[ip]*fraction_p[ip,:]*vp[ip,0]

            # add together the contributions of all particles to the current in each gridcell 
            J0[:,0] = np.sum(self.qp[:,None]*fraction_p*vp,axis=0)
            # watch(J0)
            J0/=self.Vx[:,None]
            # watch(J0)

            # #smoothing current
            # J0 = self.SM@J0

            # time_matrix_in  = time.perf_counter()
            # M=np.zeros((self.Nx,self.Nx))
            # # calculate the mass matrix
            # np.add.at(M[:,:], (ix, ix), .5*frac1**2*self.qp)
            # np.add.at(M[:,:], (ix2, ix), frac1*(1-frac1)*self.qp)
            # np.add.at(M[:,:], (ix2, ix2), .5*(1-frac1)**2*self.qp)
            # M[:,:]=M+M.T
            M = np.einsum('ij,ik->jk',fraction_p,fraction_p*self.qp[:,None])
            # time_matrix_out = time.perf_counter()
            # time_matrix += (time_matrix_out-time_matrix_in)

            bKrylov = E0 - J0*self.dt*self.theta
            # watch(bKrylov)

            # currently doesn't use 4pi and c in the scaling terms (same as code example)
            Maxwell=np.eye(self.Nx) +self.qom*self.dt**2*self.theta/2*M/self.dx
            
            E12=np.reshape(sp.sparse.linalg.spsolve(sp.sparse.csr_array(Maxwell),bKrylov),(self.Nx,self.vdim))
            # watch(E12)

            E0 = (E12-E0*(1-self.theta))/self.theta # $E^{n+\theta} = \theta*E^{n+1} + (1-\theta)*E^{n}$
            # watch(E0)
            # E12sm = self.SM@E12
            E12sm = E12

            vp[:,0] += np.sum(self.dt/self.Nsub*(E12sm[ix[:,:],0]*frac1[:,:]+E12sm[ix2[:,:],0]*(1-frac1[:,:]))*self.qom, axis=0)
            # watch(vp)
            # vp[:,0] += self.dt*self.qom*(E12sm[ix,0]*frac1 + E12sm[ix2,0]*(1-frac1))
            # watch(xp[-1,:])
            # watch(Bc)

            x[n,:] = np.hstack((np.reshape(xp_view[-1,:],(self.Np*self.xdim)),np.reshape(vp,(self.Np*self.vdim)),np.reshape(E0,(self.Nx*self.vdim)),np.reshape(Bc,(self.Nx*self.vdim))))
            # watch(x[n,:])

            # Bookkeeping
            Ek, Ee, Eb = self.Energy(x[n,:])
            diff = abs((Ek - Ekold) + (Ee - Eeold) + (Eb - Ebold))/(Ekold+Eeold+Ebold)
            # print(Ek,Ee,Eb)
            print(f"Energy difference {diff}")
            if diff>1e-15:
                print(f"dt = {self.dt} and iteration {n}, going from time {tn} to {tn+self.dt*(n)}, does not conserve energy with diff = {diff}")
                    
        # if interpolation factor is not a whole integer, then the method breaks down to first order due to first order interpolation 
        interpolation_factor = (tn1-tn)/self.dt - int((tn1-tn)/self.dt)
        if interpolation_factor == 0.0:
            interpolation_factor = 1
        if return_all_steps:
            return x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:]),x
        return x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:])
    
    #yn = [xp,vp,E0,Bc] where xp = Np x 1, vp = Np x 3, E0 = Nx x 3, Bc = Nx x 3
    def Step3D(self,yn,tn,tn1,return_all_steps = False):
        xp = np.copy(np.reshape(yn[:self.xdim*self.Np],(self.Np)))
        vp = np.copy(np.reshape(yn[self.xdim*self.Np:self.xdim*self.Np+self.vdim*self.Np], (self.Np,self.vdim)))
        E0 = np.copy(np.reshape(yn[self.xdim*self.Np+self.vdim*self.Np:self.xdim*self.Np+self.vdim*(self.Np + self.Nx)], (self.Nx,self.vdim)))
        Bc = np.copy(np.reshape(yn[self.xdim*self.Np+self.vdim*(self.Np + self.Nx):self.xdim*self.Np+self.vdim*(self.Np + 2*self.Nx)], (self.Nx,self.vdim)))

        # Bookkeeping the energy in the system
        Ek, Ee, Eb = self.Energy(yn)
        
        nb_steps = math.ceil(abs((tn1-tn)/self.dt))
        x = np.empty((nb_steps+1,yn.shape[0]))
        x[0,:] = yn
        for n in np.arange(1,nb_steps+1):
            xp += vp[:,0]*self.dt
            xp_view = xp%self.L

            ix=np.floor(xp_view/self.dx).astype(int); # cell of the particle, first cell is cell 0, first node is 0 last node Nx
            frac1 = 1-(xp_view/self.dx-ix)
            ix2=(ix+1)%self.Nx # second cell of influence due to extended local support of b-spline
            
            B = np.ones((self.Nx,self.vdim))
            B[1:,:]=.5*(Bc[1:,:]+Bc[:-1,:]); B[0,:]=.5*(Bc[-1,:]+Bc[0,:]) # calculate the vertex points of the magnetic field

            Bp = np.empty((self.Np,self.vdim))
            Bp[:,:]=B[ix,:]*frac1[:,None]+B[ix2,:]*(1-frac1[:,None]) # calculate the magnetic field for each particle location as an interpolation of the nearest gridpoints
            alphap = self.alpha(Bp) # calculate all of the alpha matrices for each of the particles
            vphat = np.einsum("ijk,ik->ij",alphap,vp) # $\hat{v_p} = \alpha_p*v_p$
            J0 = np.zeros((self.Nx,self.vdim))
            # add together the contributions of all particles to the current in each gridcell 
            for i in range(self.vdim):
                np.add.at(J0[:,i], (ix), frac1*self.qp*vphat[:,i])
                np.add.at(J0[:,i], (ix2), (1-frac1)*self.qp*vphat[:,i])
            J0/=self.Vx[:,None]

            # #smoothing current
            # J0 = self.SM@J0

            # time_matrix_in  = time.perf_counter()
            M=np.zeros((self.Nx,self.Nx,self.vdim,self.vdim))
            # calculate the 3v mass matrices
            for i in range(self.vdim):
                    for j in range(self.vdim):
                        np.add.at(M[:,:,i,j], (ix, ix), .5*frac1**2*self.qp*alphap[:,i,j])
                        np.add.at(M[:,:,i,j], (ix2, ix), frac1*(1-frac1)*self.qp*alphap[:,i,j])
                        np.add.at(M[:,:,i,j], (ix2, ix2), .5*(1-frac1)**2*self.qp*alphap[:,i,j])
                        M[:,:,i,j]=M[:,:,i,j]+M[:,:,i,j].T
                        # M[:,:,i,j]=self.SM@M[:,:,i,j]@self.SM
            # time_matrix_out = time.perf_counter()
            # time_matrix += (time_matrix_out-time_matrix_in)
            A = self.qom*self.dt**2*self.theta/2*M/self.dx
            I = np.eye(self.Nx)
            Derv= np.eye(self.Nx) + np.diagflat(np.ones(self.Nx-1),-1); Derv[0,-1] = -1; Derv/=self.dx
            Derc= -np.eye(self.Nx) + np.diagflat(np.ones(self.Nx-1),1); Derc[-1,0] = 1; Derc/=self.dx

            bKrylov = np.hstack((E0[:,0] - J0[:,0]*self.dt*self.theta, E0[:,1]-(J0[:,1])*self.dt*self.theta, E0[:,2]-(J0[:,2])*self.dt*self.theta,Bc[:,1],Bc[:,2]))

            # currently doesn't use 4pi and c in the scaling terms (same as code example)
            Maxwell=np.vstack((np.hstack((I + A[:,:,0,0], A[:,:,0,1], A[:,:,0,2],  np.zeros((self.Nx,2*self.Nx)))),\
                np.hstack((A[:,:,1,0], I + A[:,:,1,1], A[:,:,1,2],np.zeros((self.Nx,self.Nx)), Derv*self.dt*self.theta)),\
                np.hstack((A[:,:,2,0],  A[:,:,2,1], I+A[:,:,2,2],  -Derv*self.dt*self.theta,  np.zeros((self.Nx,self.Nx)))),\
                np.hstack((np.zeros((self.Nx,2*self.Nx)),   -Derc*self.dt*self.theta, I, np.zeros((self.Nx,self.Nx)))),\
                np.hstack((np.zeros((self.Nx,self.Nx)),Derc*self.dt*self.theta, np.zeros((self.Nx,2*self.Nx)), I))))
            
            xKrylov=sp.sparse.linalg.spsolve(sp.sparse.csr_array(Maxwell),bKrylov)

            E12 = np.reshape((xKrylov[:self.vdim*self.Nx]),(self.Nx,self.vdim),"F")
            Bc[:,1]=(xKrylov[self.vdim*self.Nx:4*self.Nx]-Bc[:,1]*(1-self.theta))/self.theta
            Bc[:,2]=(xKrylov[4*self.Nx:5*self.Nx]-Bc[:,2]*(1-self.theta))/self.theta
            # E12 = np.zeros((self.Nx,self.vdim))
            # Bc = np.ones(B.shape)

            E0 = (E12-E0*(1-self.theta))/self.theta # $E^{n+\theta} = \theta*E^{n+1} + (1-\theta)*E^{n}$
            # E12sm = self.SM@E12
            E12sm = E12

            vp[:,:] = np.einsum("ijk,ik->ij",(2*alphap-np.eye(self.vdim)[None,:,:]),vp) + self.dt*self.qom*np.einsum("ijk,ik->ij",alphap,(E12sm[ix,:]*frac1[:,None] + E12sm[ix2,:]*(1-frac1[:,None])))
            x[n,:] = np.hstack((np.reshape(xp[:],(self.Np*self.xdim)),np.reshape(vp,(self.Np*self.vdim)),np.reshape(E0,(self.Nx*self.vdim)),np.reshape(Bc,(self.Nx*self.vdim))))
            
            # Bookkeeping
            Ekold=Ek; Ebold=Eb; Eeold=Ee
            Ek, Ee, Eb = self.Energy(x[n,:])
            # if tn > 0 or (n > 1):
            diff = abs((Ek - Ekold) + (Ee - Eeold) + (Eb - Ebold))/(Ekold+Eeold+Ebold)
            # print(f"Energy difference: {diff}, conserved energy: {diff<1e-15}")
            if diff>1e-14:
                print(f"dt = {self.dt} and iteration {n-1}, going from time {tn+self.dt*(n-1)} to {tn+self.dt*(n)}, does not conserve energy with diff = {diff}")
                    
        # if interpolation factor is not a whole integer, then the method breaks due to first order interpolation 
        interpolation_factor = (tn1-tn)/self.dt - int((tn1-tn)/self.dt)
        if interpolation_factor == 0.0:
            interpolation_factor = 1
        if return_all_steps:
            return x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:]), x
        return x[-2,:] + interpolation_factor*(x[-1,:]-x[-2,:])
    
# multiprocess fine solver function
def worker(F,pipe):
    while True:
        chunk = pipe.recv()
        res = np.empty((chunk.shape[0],chunk.shape[1]-2))
        for i in range(chunk.shape[0]):
            X0 = chunk[i,:-2]
            T0 = chunk[i,-2]
            T1 = chunk[i,-1]
            res[i] = F.Step(X0, T0,T1)
        pipe.send(res)

class PararealSolver:
    # F and G should be step functions that perform an integration for starting value x on time t until t1
    def __init__(self, iterations, F, G, E_threshold = 1e-8, it_threshold=1e-8):
        self.F = F # fine solver, should have a self.F.Step(yn,tn,tn1) function
        self.G = G # coarse solver, should have a self.F.Step(yn,tn,tn1) function
        self.iterations = iterations # max number of parareal iterations
        self.it_threshold = it_threshold # threshold to assess convergence of states
        self.E_threshold = E_threshold # threshold to assess energy conservation
        # self.Nproc = 12 # total amount of processors to be used

        # # bookkeeping for multiprocessing
        # self.pipes = []
        # self.processes = []
        # for i in range(self.Nproc):
        #     parent_pipe, child_pipe = multiprocessing.Pipe()
        #     self.pipes.append(parent_pipe)
        #     self.processes.append(multiprocessing.Process(target=worker,args=(self.F,child_pipe)))
        #     self.processes[i].start()
    def __del__(self):
        # clean up the multiprocessers
        # for i in range(self.Nproc):
        #     self.processes[i].terminate()
        #     self.processes[i].join()
        return


    # Performs the parareal alorithm on the timesteps defined in T, where X0 is the state at T[0] 
    # and using self.F and self.G to advance the solution
    def Solve(self, X0, T):
        X = np.zeros((self.iterations+1,T.shape[0],X0.shape[0]))
        X[0,0,:] = X0
        for i in range(T.shape[0]-1):
            X[0,i+1,:] = self.G.Step(X[0,i,:],T[i],T[i+1])
        k = 0
        fine_x = np.zeros((T.shape[0]-1,X0.shape[0]))
        coarse_x = np.copy(X[0,1:,:])
        new_coarse_x = np.empty(X0.shape[0])
        # keep track of which steps are already converged
        converged_until = 0

        while ((k < self.iterations) and (converged_until < T.shape[0] - 1)):
            k+=1
            X[k,:converged_until+1,:] = X[k-1,:converged_until+1,:]
            # calculate best distribution of timesteps for processors
            # N_per_proc = np.ones(self.Nproc,dtype=int)*(T.shape[0]-converged_until-1)//self.Nproc
            # for i in range((T.shape[0]-converged_until-1)%self.Nproc):
            #     N_per_proc[i] += 1
            # # give each processor its data to calculate
            # start = converged_until
            # for i in range(self.Nproc):
            #     end = start+N_per_proc[i]
            #     self.pipes[i].send(np.column_stack((X[k-1,start:end,:],T[start:end],T[start+1:end+1])))
            #     start = end
            # # receive the results from each processor
            # start = converged_until+1
            # for i in range(self.Nproc):
            #     end = start+N_per_proc[i]
            #     fine_x[start:end,:] = self.pipes[i].recv()
            #     start = end

            for i in range(converged_until,T.shape[0]-1):
                fine_x[i,:] = self.F.Step(X[k-1,i,:], T[i],T[i+1])
            # fine_x[converged_until+1:,:] = np.array(p.map(self.thread_func,np.hstack((X[k-1,converged_until:-1,:],T[converged_until:-1,None],T[converged_until+1:,None]))))
            for j in range(converged_until,T.shape[0]-1):
                # calculate new coarse solution
                new_coarse_x[:] = self.G.Step(X[k,j,:], T[j],T[j+1])
                # use the Parareal update scheme
                X[k,j+1,:] = new_coarse_x + fine_x[j,:] - coarse_x[j,:]
                coarse_x[j,:] = np.copy(new_coarse_x)
                # check for convergence
                if np.max(np.linalg.norm(X[k,:,:]-X[k-1,:,:],axis=-1)/np.linalg.norm(X[k,:,:],axis=-1))<=self.it_threshold:
                   return X[:k+1,:,:]
                # if((j == converged_until) and (np.linalg.norm(X[k,j+1,:]-X[k-1,j+1,:])<=self.it_threshold *np.linalg.norm(X[k,j+1,:]))):
                #     converged_until += 1
            print(f"For iteration {k} max relative state change: {np.max(np.linalg.norm((X[k,:,:]-X[k-1,:,:]),axis=1)/np.linalg.norm(X[k,:,:],axis=1))} and time steps until and including {converged_until} have converged.")
        return X[:k+1,:,:]
    
    def thread_func(self,args):
        X0 = args[:-2]
        T0 = args[-2]
        T1 = args[-1]
        return self.F.Step(X0, T0,T1)


class SymplecticParareal:
    # F and G should be step functions that perform an integration for starting value x on time t until t1
    def __init__(self, iterations, F, G, it_threshold=1e-8):
        self.F = F # fine solver, should have a self.F.Step(yn,tn,tn1) function
        self.G = G # coarse solver, should have a self.F.Step(yn,tn,tn1) function
        self.iterations = iterations # max number of parareal iterations
        self.it_threshold = it_threshold # threshold to assess convergence of states
        self.Nproc = 12 # total amount of processors to be used
    
    def Solve(self, X0, T):
        X = np.empty((self.iterations,T.shape[0],X0.shape[0]))
        X[0,0,:] = X0
        for i in range(T.shape[0]-1):
            X[0,i+1,:] = self.G.Step(X[0,i,:],T[i],T[i+1])
        #original total energy (only for energy conservation checks)
        Etot = np.sum(self.G.Energy(X[0,0,:]))
        k = 1
        fine_x = np.zeros((T.shape[0],X0.shape[0]))
        fine_x[0,:] = X0
        coarse_x = np.copy(X[0,:,:])
        new_coarse_x = np.empty(X0.shape[0])
        # keep track of which steps are already converged
        converged_until = 0

        # bookkeeping for multiprocessing
        pipes = []
        processes = []
        for i in range(self.Nproc):
            parent_pipe, child_pipe = multiprocessing.Pipe()
            pipes.append(parent_pipe)
            processes.append(multiprocessing.Process(target=worker,args=(self.F,child_pipe)))
            processes[i].start()

        while ((k < self.iterations) and (converged_until < T.shape[0]-1)):
            X[k,:converged_until+1,:] = X[k-1,:converged_until+1,:]
            # calculate best distribution of timesteps for processors
            N_per_proc = np.ones(self.Nproc,dtype=int)*(T.shape[0]-1-converged_until)//self.Nproc
            for i in range((T.shape[0]-1-converged_until)%self.Nproc):
                N_per_proc[i] += 1
            # give each processor its data to calculate
            start = converged_until
            for i in range(self.Nproc):
                end = start+N_per_proc[i]
                pipes[i].send(np.column_stack((X[k-1,start:end,:],T[start:end],T[start+1:end+1])))
                start = end
            # receive the results from each processor
            start = converged_until+1
            for i in range(self.Nproc):
                end = start+N_per_proc[i]
                fine_x[start:end,:] = pipes[i].recv()
                start = end
            
            print(f"For iteration {k} max energy difference: {np.max(np.abs(list(map(lambda x: np.sum(self.F.Energy(x))-Etot , X[k,:,:])))/Etot)}, max relative state change: {np.max(np.linalg.norm((X[k,:,:]-X[k-1,:,:])/np.linalg.norm(X[k,:,:]),axis=1))}")
            k += 1
        for i in range(self.Nproc):
            processes[i].terminate()
            processes[i].join()
        return X[:k,:,:]

