import numpy as np
from scipy.special import gamma as gfunc


# power-law density profile with arbitrary normalization

def rho(r, gamma):
    return 1./r**gamma

def Sigma(R, gamma):
    return R**(1-gamma)*np.pi**0.5*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)
  
def M2d(R, gamma):
    return 2*np.pi**1.5/(3.-gamma)*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)*R**(3-gamma)

def M3d(r, gamma):
    return 4*np.pi/(3.-gamma)*r**(3-gamma)

