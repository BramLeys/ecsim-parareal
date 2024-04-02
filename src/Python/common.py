import numpy as np
import math
# r needs to be the square of the actual radius (for energy conservation during parareal)
def Sphere2Cart(cyl):
    r = np.copy(cyl[:,0])
    theta = np.copy(cyl[:,1]) # in [0,pi]
    phi = np.copy(cyl[:,2]) # in [-pi,pi]
    # sanitization after the parareal iteration result (messes up the energy conservation!!!)
    indr = np.where(r<0)[0]
    r[indr] *= -1
    phi[indr] += math.pi
    theta[indr] += math.pi
    r = np.sqrt(np.copy(r))
    return np.column_stack((r*np.cos(phi)*np.sin(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(theta)))

# r needs to be the square of the actual radius (for energy conservation during parareal)
def Cart2Sphere(cart):
    theta = np.zeros(cart.shape[0])
    r = np.linalg.norm(cart,axis=1)
    theta[r>0] = np.arccos(cart[:,2][r>0]/r[r>0]) # angle from z-axis in [0,pi]
    phi = np.arctan2(cart[:,1],cart[:,0]) # angle in xy-plane in [-pi,pi]
    # assert(np.all(abs(cart - Sphere2Cart(np.column_stack((r**2, theta, phi)))) < 1e-15))
    return np.column_stack((r**2, theta, phi))

