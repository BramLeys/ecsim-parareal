import matplotlib.pyplot as plt
import numpy as np
import Parareal
import math
import Solvers

lambdaa = np.array((-1,-2))
def f(t,x):
    return lambdaa*x

if __name__ == '__main__':
    # q = -1.602176634e-19
    # m = 9.1093837015e-31

    # E = lambda x,t: np.array([-m/q*math.sin(t) + m/(2*q)*math.cos(t), -m/(2*q)*math.sin(t), -m/(2*q)*math.sin(t) - 3*m/(2*q)*math.cos(t)])
    # B = lambda x,t: np.array([-m/(2*q), m/(2*q),m/(2*q)])
    # analytical_x = lambda t: np.array([math.sin(t),math.cos(t),math.sin(t) + math.cos(t)])
    # analytical_v = lambda t: np.array([math.cos(t),-math.sin(t),math.cos(t) - math.sin(t)])

    # fine = Solvers.BorisSolver(10,N*2,E,B)
    # coarse = Solvers.BorisSolver(10,N,E,B)
    # para = Solvers.PararealSolver(10,N,2*N,fine.Step,coarse.Step)
    # para.Solve(analytical_x(0), analytical_v(0))
    # analytical_x = lambda t: np.array((100*math.exp(lambdaa*t),50*math.exp(2*lambdaa*t)))
    analytical_x = lambda t: 10*np.exp(lambdaa*t)

    t0 = 0
    t1 = 10
    dt_coarse = 1e-1
    dt_fine = dt_coarse/10
    N = int(t1/dt_coarse)
    T = np.linspace(t0,t1,N+1)
    refinement_levels = 8
    iterations = 10
    F = Solvers.ForwardEuler(f,dt_fine)
    G = Solvers.ForwardEuler(f,dt_coarse)

    P = Parareal.PararealSolver(iterations,F,G)
    X = P.Solve(analytical_x(t0), T)

    coarse_x = np.empty((T.shape[0],2))
    coarse_x[0,:] = analytical_x(t0)
    for i in range(T.shape[0]-1):
        coarse_x[i+1,:] = G.Step(coarse_x[i,:],T[i],T[i+1])

    fine_x = np.empty((T.shape[0],2))
    fine_x[0,:] = analytical_x(t0)
    for i in range(T.shape[0]-1):
        fine_x[i+1,:] = F.Step(fine_x[i,:],T[i],T[i+1])

    ax = plt.figure().add_subplot()
    for k in range(X.shape[0]):
        ax.semilogy(T,X[k,:,0], label="parareal simulation iteration "+str(k))
    ax.semilogy(T,coarse_x[:,0], label="coarse simulation x")
    ax.semilogy(T,fine_x[:,0], label="fine simulation x")
    ax.semilogy(T,np.array([analytical_x(t) for t in T])[:,0], label="actual x")
    ax.legend()
    plt.title("Simulation")
    ax.set_xlabel('t')
    ax.set_ylabel('X')


    errors_x = np.empty((1,refinement_levels))
    for n in range(refinement_levels):
        F.dt = dt_coarse/2**n
        P = Parareal.PararealSolver(iterations,F,G)
        X = P.Solve(analytical_x(t0), T)
        errors_x[:,n] = np.linalg.norm(X[-1,:,:]-np.array([analytical_x(t) for t in T]))


    ax = plt.figure().add_subplot()
    plt.title("Convergence of errors for test equation Parareal")
    ax.semilogy(range(1,refinement_levels+1),np.linalg.norm(errors_x,axis=0), label='Errors on Parareal simulation')
    ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**2 for n in range(refinement_levels)], label='O(dt^2).')
    ax.semilogy(range(1,refinement_levels+1),[(1/((2**n)*10))**4 for n in range(refinement_levels)], label='O(dt^4).')

    ax.legend()
    ax.set_xlabel('i (#timesteps = 10*(2^i))')
    ax.set_ylabel('Norm of error')


    plt.show()
