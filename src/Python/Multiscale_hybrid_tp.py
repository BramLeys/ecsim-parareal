import numpy as np
import math
import Solvers
import scipy.interpolate as interpolate


def FiniteDifference(F, x, eps= 1e-8):
    n = np.size(x)
    grad = np.empty((n,n))
    f0 = F(x)
    delta = np.zeros(n)
    for i in range(n):
        delta[i] = eps
        f1 = F(x+delta)
        grad[:,i] = (f1-f0)/eps
        delta[i] = 0
    return grad

def NewtonsIteration(F,x, gradF, threshold=1e-6):
    xn = x
    diff = math.inf
    while diff > threshold:
        xn1 = xn - F(xn)/gradF(xn)
        diff = np.linalg.norm((xn1-xn))/np.linalg.norm(xn1)
        xn = xn1
    return xn


def Normalize(vec):
    c = np.linalg.norm(vec)
    if c != 0:
        return vec/c
    return vec

# gives rotationmatrix to map 3D vec1 onto vec2
def RotationMatrix(vec1, vec2):
    a = vec1/np.linalg.norm(vec1)
    b = vec2/np.linalg.norm(vec2)
    c = a.dot(b)
    if c == 1 or c==-1:
        return np.eye(3)*c
    v = a.cross(b)
    A = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + A + A.dot(A)*1/(1+c)

class MultiscaleHybridSolver:
    def __init__(self, t_stop, nb_nodes, B, gradB=None, t_start=0):
        self.q = -1.602176634e-19
        self.m = 9.1093837015e-31
        self.t_end = t_stop
        self.t0 = t_start
        self.B = B
        self.N = nb_nodes
        self.F = Solvers.BorisSolver(t_stop, nb_nodes, lambda x:0, B, t_start)
        self.dt = (t_stop-t_start)/self.N
        self.kE = 0
        if gradB is None:
            gradB = lambda x: FiniteDifference(self.B,x)
        self.gradB = gradB
        self.u_fine = np.empty((6,nb_nodes+1))
        self.u_coarse = np.empty((5,nb_nodes+1))
        self.basis = np.eye(3,3)

    # RK4
    def CoarseStepper(self,U0,t,dt):
        x_c0 = U0[:3]
        v_z0 = U0[3]
        v_theta0 = U0[4]

        x_c_func = lambda t,x: v_z0*Normalize(self.B(x)) + v_z0**2/(self.q/self.m*np.linalg.norm(self.B(x)))*np.cross(Normalize(self.B(x)),np.dot(self.gradB(x),Normalize(self.B(x))))
        v_z_func = lambda t,x: -0.5*(self.kE-v_z0**2)/(np.linalg.norm(self.B(x)))*np.dot(Normalize(self.B(x)),np.dot(self.gradB(x),Normalize(self.B(x))))
        v_theta_func = lambda t,x: self.q/self.m*np.linalg.norm(self.B(x))

        x_c = Solvers.RK4Step(x_c_func,t,x_c0,dt)
        v_z = Solvers.RK4Step(v_z_func,t,v_z0,dt)
        v_theta = Solvers.RK4Step(v_theta_func,t,v_theta0,dt)
        return np.vstack((x_c,v_z,v_theta))
        

    def FineStepper(self,u0,t,dt):
        return self.F.Step(u0[0:3],u0[3:],t,t+dt)

    # U should be in the form u = (x,v)
    def Coarsify(self,u):
        x = u[0:3]
        v = u[3:]
        b0 = self.B(x)
        e3 = Normalize(b0)
        e1 = Normalize(v-v.dot(e3)*e3)
        e2 = np.cross(e1,e3)
        self.basis = np.array([e1,e2,e3]).transpose()
        x_rot = self.basis.dot(x)
        v_rot = self.basis.dot(v)
        # cylindrical transform
        v_z = v_rot[3]
        x_p = np.linalg.norm[x_rot[:2]]
        x_theta = math.acos(x_rot[0]/x_p)
        v_p = np.linalg.norm[v_rot[:2]]
        v_theta = math.acos(v_rot[0]/v_p)

        # x_omega = dx_theta/dt => finite difference -> find previous x_theta value
        dt = 0.5
        x_theta_prev = 0
        x_omega = (x_theta-x_theta_prev)/dt

        G = lambda x_c: (x_c -x) + math.cos(x_theta-v_theta)*self.B(x).cross(v)/x_omega + math.sin(x_theta-v_theta)*(v-v_z*self.B(x))/x_omega
        # Check derivative
        dG = lambda x_c: -x + (Normalize(self.gradB(x_c)).cross(v))/np.linalg.norm(self.gradB(x_c))
        x_c = x - e3.cross(v)/np.linalg.norm(b0)
        x_c = NewtonsIteration(G,x_c,dG)
        return np.vstack((x_c,v_z, v_theta))

    # U should be in the form U = (x_c, v_z, v_theta)    
    def Reconstruct(self, U):
        # Get all of the coordinates from the coarse data
        x_c = U[0:3]
        v_z = U[3]
        v_theta = U[4]
        # Calculate auxiliary variables
        v_p = math.sqrt(self.kE - v_theta**2)
        Bc = Normalize(self.B(x_c))
        x_theta = v_theta - 0.5*Bc.dot(self.gradB(x_c).dot(Bc))
        # x_omega = dx_theta/dt => finite diference -> Get ahold of previous coarse info
        previous_U = np.empty(4)
        Bc_prev = Normalize(self.B(previous_U[0:3]))
        x_theta_prev = previous_U[4] - 0.5*Bc_prev.dot(self.gradB(x_c).dot(Bc_prev))
        x_omega = (x_theta-x_theta_prev)/self.dt

        x_p = v_p/x_omega
        x_z = 0

        # Get rotationmatrix from last iteration
        R = RotationMatrix(self.basis[:,2],Bc)
        e1 = R.dot(self.basis[:,0])
        e2 = R.dot(self.basis[:,1])
        e3 = Normalize(Bc)
        self.basis = np.array([e1,e2,e3]).transpose()

        # inverse polar 
        x_rot = [x_p*math.cos(x_theta),x_p*math.sin(x_theta),x_z]
        v_rot = [v_p*math.cos(v_theta),v_p*math.sin(v_theta),v_z]
        # rotate
        x = np.linalg.solve(self.basis,x_rot)
        v = np.linalg.solve(self.basis,v_rot)
        # add guiding center location
        x = x + x_c
        return np.vstack((x,v))

    def Solve(self,x0,v0):
        self.kE = np.linalg.norm(v0)**2
        self.u_fine[:,0] = np.vstack((x0,v0))
        b0 = self.B(x0)
        b0_norm = np.linalg.norm(b0)
        e3 = b0/b0_norm
        e1 = Normalize(v0-v0.dot(e3)*e3)
        e2 = e1.cross(e3)
        self.basis = np.array((e1,e2,e3))
        # what should v_theta be initialised to?
        self.u_coarse[:,0] = np.vstack((1/b0_norm*(e3.cross(v0)),e3.dot(v0),0))
        # CT = self.CoarseStepper(self.u_coarse[:,0],0,self.t_end)


        H = math.ceil(self.t_end - 4/5)
        theta_max = H + 4/5
        if H%2 == 1:
            H += 1

        thetas = np.empty(15)
        lookup = {}
        i = 0
        for g in [0,1,2,3,4]:
            for h in [0,1,2]:
                thetas[i] = g*math.pi/4 + H/math.pi*h
                lookup[thetas[i]] = (g,h)
                i +=1
        thetas = thetas.sort()
        coarse_solution_at_theta = np.array(list(map(lambda t: self.CoarseStepper(self.u_coarse[:,0],0,t), thetas)))
        t,c,k = interpolate.splrep(thetas,coarse_solution_at_theta,k=3)
        S = interpolate.BSpline(t,c,k)

        Ds = np.empty((5,3))
        i = 1
        x = np.linspace(0,theta_max,10)
        integral = 0
        for i in range(15):
            theta = thetas[i]
            (g,h) = lookup[theta]
            if i < 14:
                dtheta = thetas[i+1]-theta
            else:
                dtheta = theta_max-theta
            U0 = S(theta)
            second_term = self.CoarseStepper(U0, theta,theta_max-theta)
            Ds[g,h] = self.CoarseStepper(self.Coarsify(self.FineStepper(self.Reconstruct(U0,self.basis,self.kE),theta,dtheta),self.basis, self.kE), theta + dtheta, theta_max-theta-dtheta) - second_term
            y = self.Chi(g,h,x)
            integral += np.trapz(y,x)*Ds[g,h]
        
        return self.Reconstruct(S(theta_max) + integral)
    
    def Chi(self,g,h,theta):
        return

    def Step(self,un, Un, tn, dt):
        return
