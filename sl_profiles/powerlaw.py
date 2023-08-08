import numpy as np
from scipy.special import gamma as gfunc


def rho(r, gammap):
    return r**(-gammap)

def Sigma(R, gammap):
    return R**(1. - gammap)*np.pi**0.5*gfunc(0.5*(gammap - 1.))/gfunc(0.5*gammap)

def M2d(R, gammap):
    return 2*np.pi**1.5/(3.-gammap)*gfunc((gammap-1.)/2.)/gfunc(gammap/2.)*R**(3-gammap)

def M3d(r, gammap):
    return 4*np.pi/(3.-gammap)*r**(3-gammap)

