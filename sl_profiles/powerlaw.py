import numpy as np
from scipy.special import gamma as gfunc


<<<<<<< HEAD
# power-law density profile with arbitrary normalization

def rho(r, gamma):
    return 1./r**gamma

def Sigma(R, gamma):
    return R**(1-gamma)*np.pi**0.5*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)
  
def M2d(R, gamma):
    return 2*np.pi**1.5/(3.-gamma)*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)*R**(3-gamma)

def M3d(r, gamma):
    return 4*np.pi/(3.-gamma)*r**(3-gamma)
=======
def rho(r, gammap):
    return r**(-gammap)

def Sigma(R, gammap):
    return R**(1. - gammap)*np.pi**0.5*gfunc(0.5*(gammap - 1.))/gfunc(0.5*gammap)

def M2d(R, gammap):
    return 2*np.pi**1.5/(3.-gammap)*gfunc((gammap-1.)/2.)/gfunc(gammap/2.)*R**(3-gammap)

def M3d(r, gammap):
    return 4*np.pi/(3.-gammap)*r**(3-gammap)
>>>>>>> 3338f6c6fef5210c50d205008eaf6f03f3b931f9

