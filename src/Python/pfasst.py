import numpy as np
import math
from scipy import interpolate


class PFASSTSolver:
    # F and G should be step functions that perform an integration for starting value x on time t until t1
    def __init__(self, iterations, F, G, threshold=1e-8):
        self.F = F
        self.G = G
        self.iterations = iterations
        self.THRESHOLD = threshold


    def Solve(self, X0, T):
        X = np.empty((self.iterations,T.shape[0],X0.shape[0]))
        X[:,0,:] = X0
        for i in range(T.shape[0]-1):
            X[0,i+1,:] = self.G(X[0,i,:],T[i],T[i+1])
        k = 1
        fine_x = np.empty((T.shape[0],X0.shape[0]))
        fine_x[0,:] = X0
        coarse_x = X[0,:,:]
        new_coarse_x = np.empty(X0.shape[0])
        while (k < self.iterations) and not ((abs(X[k,:,:]-X[k-1,:,:])<=self.THRESHOLD*X[k,:,:]).all()):
            # PARALLELIZE THIS
            for j in range(1,T.shape[0]):
                fine_x[j,:]= self.F(X[k-1,j-1,:], T[j-1],T[j])
            for j in range(1,T.shape[0]):
                new_coarse_x[:] = self.G(X[k,j-1,:], T[j-1],T[j])
                X[k,j,:] = new_coarse_x + fine_x[j,:] - coarse_x[j,:]
                coarse_x[j,:] = new_coarse_x
            k += 1
        return X