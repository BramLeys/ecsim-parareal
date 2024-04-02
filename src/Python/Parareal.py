import numpy as np
import multiprocessing

class PararealSolver:
    # F and G should have methods F.Step(X0,t0,t1) and G.Step(X0,t0,t1) which give a solution of x(t1) starting from x(t0) = X0
    def __init__(self, iterations, F, G, threshold=1e-8):
        self.F = F  # fine propagator
        self.G = G # coarse propagator
        self.max_iter = iterations # max amount of parareal iterations
        self.threshold = threshold # relative convergence tolerance for states
        self.Nproc = 5 # number of processes

    # 
    def Solve(self, X0, T):
        X = np.empty((self.max_iter,T.shape[0],X0.shape[0]))
        # perform initial coarse solution
        X[0,0,:] = X0
        for i in range(T.shape[0]-1):
            X[0,i+1,:] = self.G.Step(X[0,i,:],T[i],T[i+1])
        k = 1
        fine_x = np.zeros((T.shape[0],X0.shape[0]))
        fine_x[0,:] = X0
        coarse_x = np.copy(X[0,:,:])
        new_coarse_x = np.empty(X0.shape[0])
        converged_until = 0
        with multiprocessing.Pool(self.Nproc) as p:
            # check that the max_iterations are not reached and that not all timesteps have converged
            while ((k < self.max_iter) and (converged_until < T.shape[0]-1)):
                # reuse already converged values from the previous iteration
                X[k,:converged_until+1,:] = X[k-1,:converged_until+1,:]
                # perform all of the needed fine solver steps in parallel
                fine_x[converged_until+1:,:] = np.array(p.map(self.thread_func,np.hstack((X[k-1,converged_until:-1,:],T[converged_until:-1,None],T[converged_until+1:,None]))))
                for j in range(converged_until+1,T.shape[0]):
                    new_coarse_x[:] = self.G.Step(X[k,j-1,:], T[j-1],T[j])
                    X[k,j,:] = new_coarse_x + fine_x[j,:] - coarse_x[j,:]

                    # check for convergence of the states (all previous time steps must have already converged)
                    if((np.linalg.norm(X[k,j,:]-X[k-1,j,:])< self.threshold*np.linalg.norm(X[k,j,:])) and (j == converged_until+1)):
                        converged_until = j
                    coarse_x[j,:] = np.copy(new_coarse_x)
                print(f"For iteration {k} max state change: {np.max(np.linalg.norm((X[k,:,:]-X[k-1,:,:]),axis=1))} and time steps up until {converged_until} have converged.")
                k += 1
            p.close()
        return X[:k,:,:]
    
    def thread_func(self,args):
        X0 = args[:-2]
        T0 = args[-2]
        T1 = args[-1]
        return self.F.Step(X0,T0,T1)
