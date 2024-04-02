import matplotlib.pyplot as plt
import numpy as np
import Solvers
import matplotlib as mpl
import math

# Tests are created using method of manufactured solutions

q = -1.602176634e-19
m = 9.1093837015e-31

refinement_levels = 8

# Pure electrical field
E = lambda x,t: np.array([-math.sin(t)*m/q,-math.cos(t)*m/q,(-math.sin(t)-math.cos(t))*m/q])
B = lambda x,t: np.array([0,0,0])
analytical_x = lambda t: np.array([math.sin(t),math.cos(t),math.sin(t) + math.cos(t)])
analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)])

errors_x = np.empty((3,refinement_levels))
errors_v = np.empty((3,refinement_levels))
basic = Solvers.BorisSolver(10,1,E,B)
for n in range(refinement_levels):
    basic.SetN((2**n)*10)
    basic.Solve(analytical_x(0),analytical_v(0))
    errors_x[:,n] = np.linalg.norm(basic.x-np.transpose(np.array([analytical_x(t) for t in basic.t])))
    errors_v[:,n] = np.linalg.norm(basic.v-np.transpose(np.array([analytical_v(t-basic.dt/2) for t in basic.t])))

ax = plt.figure().add_subplot()
plt.title("Convergence of errors with increasing refinement of mesh for an electric field")
ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_x,axis=0), label='Errors on the position for pure electrical field.')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**2 for n in range(refinement_levels)], label='O(dt^2).')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**1 for n in range(refinement_levels)], label='O(dt).')
ax.legend()
ax.set_xlabel('i (#nodes = 10*(2^i))')
ax.set_ylabel('Norm of error')

# Pure magnetic field
E = lambda x,t: np.array([0,0,0])
B = lambda x,t: np.array([0,0,m/q])
analytical_x = lambda t: np.array([math.sin(t),math.cos(t),t])
analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),1])

errors_x = np.empty((3,refinement_levels))
errors_v = np.empty((3,refinement_levels))
for n in range(refinement_levels):
    basic = Solvers.BorisSolver(10,(2**n)*10,E,B)
    basic.Solve(analytical_x(0),analytical_v(0))
    errors_x[:,n] = np.linalg.norm(basic.x-np.transpose(np.array([analytical_x(t) for t in basic.t])))
    errors_v[:,n] = np.linalg.norm(basic.v-np.transpose(np.array([analytical_v(t-basic.dt/2) for t in basic.t])))

ax= plt.figure().add_subplot()
plt.title("Convergence of errors with increasing refinement of mesh for a magnetic field")
ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_x,axis=0), label='Errors on the position for pure magnetic field.')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**2 for n in range(refinement_levels)], label='O(dt^2).')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**1 for n in range(refinement_levels)], label='O(dt).')
ax.legend()
ax.set_xlabel('i (#nodes = 10*(2^i))')
ax.set_ylabel('Norm of error')


# Combination of electric and magnetic field

E = lambda x,t: np.array([-m/q*math.sin(t) + m/(2*q)*math.cos(t), -m/(2*q)*math.sin(t), -m/(2*q)*math.sin(t) - 3*m/(2*q)*math.cos(t)])
B = lambda x,t: np.array([-m/(2*q), m/(2*q),m/(2*q)])
analytical_x = lambda t: np.array([math.sin(t),math.cos(t),math.sin(t) + math.cos(t)])
analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)])

errors_x = np.empty((3,refinement_levels))
errors_v = np.empty((3,refinement_levels))
for n in range(refinement_levels):
    basic = Solvers.BorisSolver(10,(2**n)*10,E,B)
    basic.Solve(analytical_x(0),analytical_v(0))
    errors_x[:,n] = np.linalg.norm(basic.x-np.transpose(np.array([analytical_x(t) for t in basic.t])))
    errors_v[:,n] = np.linalg.norm(basic.v-np.transpose(np.array([analytical_v(t-basic.dt/2) for t in basic.t])))

ax = plt.figure().add_subplot()
plt.title("Convergence of errors with increasing refinement of mesh for both magnetic and electric fields")
ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_x,axis=0), label='Errors on the position for both electric and magnetic field.')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**2 for n in range(refinement_levels)], label='O(dt^2).')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**1 for n in range(refinement_levels)], label='O(dt).')

ax.legend()
ax.set_xlabel('i (#nodes = 10*(2^i))')
ax.set_ylabel('Norm of error')


#interpolation scheme
E = lambda x,t: np.array([-m/q*math.sin(t) + m/(2*q)*math.cos(t), -m/(2*q)*math.sin(t), -m/(2*q)*math.sin(t) - 3*m/(2*q)*math.cos(t)])
B = lambda x,t: np.array([-m/(2*q), m/(2*q),m/(2*q)])
analytical_x = lambda t: np.array([math.sin(t),math.cos(t),math.sin(t) + math.cos(t)])
analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)])

errors_x = np.empty((3,refinement_levels))
errors_v = np.empty((3,refinement_levels))
errors_interpolated_x = np.empty((3,refinement_levels))
errors_interpolated_v = np.empty((3,refinement_levels))
dimension = 3
t0 = 0
t_end = 10
for n in range(refinement_levels):
    N = (2**n)*10
    basic = Solvers.BorisSolver(t_end,2*N,E,B,t0)
    basic.Solve(analytical_x(0),analytical_v(0))
    ts = np.linspace(t0,t_end,N+1)
    dt = (t_end-t0)/N
    X = np.empty((dimension,N+1))
    X[:,0] = analytical_x(0)
    V = np.empty((dimension,N+1))
    V[:,0] = analytical_v(0)
    V[:,0] = basic.UpdateVelocity(X[:,0],V[:,0], t0,-0.5*dt/2)
    for i in range(1,N+1):
        X[:,i], V[:,i] = basic.Step(X[:,i-1],V[:,i-1],ts[i-1],ts[i])
    errors_x[:,n] = np.linalg.norm(basic.x[:,0::2]-np.transpose(np.array([analytical_x(t) for t in ts])))
    errors_interpolated_x[:,n] = np.linalg.norm(X-np.transpose(np.array([analytical_x(t) for t in ts])))
    errors_v[:,n] = np.linalg.norm(basic.v[:,0::2]-np.transpose(np.array([analytical_v(t-dt/2) for t in ts])))
    errors_interpolated_v[:,n] = np.linalg.norm(V-np.transpose(np.array([analytical_v(t-dt/2) for t in ts])))

ax = plt.figure().add_subplot()
plt.title("Convergence of errors with increasing refinement of mesh for both magnetic and electric fields")
ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_x,axis=0), label='Errors on fine grid')
ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_interpolated_x,axis=0), label='Errors on interpolated grid')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**2 for n in range(refinement_levels)], label='O(dt^2).')
ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**1 for n in range(refinement_levels)], label='O(dt).')

ax.legend()
ax.set_xlabel('i (#nodes = 10*(2^i))')
ax.set_ylabel('Norm of error')


plt.show()