import matplotlib.pyplot as plt
import numpy as np
import Solvers
import matplotlib as mpl
import math

q = -1.602176634e-19
m = 9.1093837015e-31

E = lambda x,t: np.array([-m/q*math.sin(t) + m/(2*q)*math.cos(t), -m/(2*q)*math.sin(t), -m/(2*q)*math.sin(t) - 3*m/(2*q)*math.cos(t)])
B = lambda x,t: np.array([-m/(2*q), m/(2*q),m/(2*q)])
analytical_x = lambda t: np.array([math.sin(t),math.cos(t),math.sin(t) + math.cos(t)])
analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)])

N_base = 10
dimension = 3
t0 =0
t_end = 1
basic = Solvers.BorisSolver(t_end,N_base,E,B,t0)
refinements = 5
err = np.empty((refinements,2))
for j in range(refinements):
    N = N_base * 2**j
    basic.SetN(N)
    ts = np.linspace(t0,t_end,N+1)
    dt = (t_end-t0)/N
    X = np.empty((dimension,N+1))
    X[:,0] = analytical_x(0)
    V = np.empty((dimension,N+1))
    V[:,0] = analytical_v(-dt/2)
    # V[:,0] = basic.UpdateVelocity(X[:,0],V[:,0], t0,-0.5*dt)
    for i in range(1,N+1):
        X[:,i], V[:,i] = basic.Step(X[:,i-1],V[:,i-1],ts[i-1],ts[i])
    err[j,:] = [np.linalg.norm(analytical_x(t_end) - X[:,-1])/np.linalg.norm(analytical_x(t_end)), np.linalg.norm(analytical_v(t_end-0.5*dt) - V[:,-1])/np.linalg.norm(analytical_v(t_end-0.5*dt))]
    if (j > 0):
        print(f"convergence: {err[j,:]/err[j-1,:]}")
print(err)
# ax = plt.figure().add_subplot(projection="3d")
# ax.plot(X[0,:],X[1,:],ts, label="Should be twice as accurate")
# ax.plot(basic.x[0,:],basic.x[1,:],basic.t,label="Regular solve")
# plt.title("Evolution of particle in x and y direction with time in the z-axis")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('t')
# ax.legend()

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
# ax.view_init(elev=20., azim=-35, roll=0)
# plt.show()

