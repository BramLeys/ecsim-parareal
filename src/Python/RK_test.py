import matplotlib.pyplot as plt
import numpy as np
import Solvers
import matplotlib as mpl
import math

q = -1.602176634e-19
m = 9.1093837015e-31

refinement_levels = 8

# b = lambda x,t: np.array([-math.sin(x[0])*m/q,-math.cos(x[1])*m/q,(-math.sin(x[2])-math.cos(x[2]))*m/q])
# grad_b = lambda x,t: np.array([[-m/q*math.cos(x[0]), 0,0],[0,math.sin(x[1])*m/q,0],[0,0,m/q*(math.sin(x[2])-math.cos(x[2]))]])

# analytical_x = lambda t: np.array([math.sin(t),math.cos(t),math.sin(t) + math.cos(t)])
# analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)])

# def gyromodel(u,Ek,b,gradb,t):
#     vz = u[3]
#     xc = u[:3]
#     bc = b(xc,t)
#     bc_norm = np.linalg.norm(bc)
#     bc_hat = bc/bc_norm
#     res = np.empty(4)
#     gradB = gradb(xc,t)
#     res[:3] = vz*bc_hat + vz**2/(q/m*bc_norm)*bc_hat.cross(gradB*bc_hat)
#     res[3] = -1/2*(Ek-vz**2)/(bc_norm)*bc_hat.dot(gradB*bc_hat)
#     return res

lambdaa = -4.24
f = lambda t,x: lambdaa*x

analytical_x = lambda x: 100*math.exp(lambdaa*x)

errors_x = np.empty((1,refinement_levels))
dimension = 1
t0 = 0
t_end = 10
for n in range(refinement_levels):
    N = (2**n)*10
    ts = np.linspace(t0,t_end,N+1)
    dt = (t_end-t0)/N
    X = np.empty((dimension,N+1))
    X[:,0] = analytical_x(0)
    for i in range(1,N+1):
        X[:,i]= Solvers.RK2Step(f,X[:,i-1],ts[i-1],ts[i],dt)
    errors_x[:,n] = np.linalg.norm(X-np.transpose(np.array([analytical_x(t) for t in ts])))

ax = plt.figure().add_subplot()
plt.title("Simulation")
ax.plot(ts,X.transpose(), label='simulated x')
ax.plot(ts,np.array([analytical_x(t) for t in ts]), label="actual x")

ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('x')

ax = plt.figure().add_subplot()
plt.title("Convergence of errors for test equation RK4")
ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_x,axis=0), label='Errors on fine grid')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**2 for n in range(refinement_levels)], label='O(dt^2).')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**4 for n in range(refinement_levels)], label='O(dt^4).')

ax.legend()
ax.set_xlabel('i (#nodes = 10*(2^i))')
ax.set_ylabel('Norm of error')


plt.show()


