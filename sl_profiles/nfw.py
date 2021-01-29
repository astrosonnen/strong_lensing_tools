# scripts to calculate lensing-related quantities for a spherical NFW profile.
# The profile is normalized so that the mass enclosed within a 3D shell of radius equal to the scale radius r_s is one.

import numpy as np


M3d1 = np.log(2.) - 0.5

def gfunc(x):
    xx = np.atleast_1d(x)
    g = xx*.0
    arr = xx[xx<1]
    g[xx<1] = np.log(arr/2) + 1/np.sqrt(1 - arr**2)*np.arccosh(1/arr)
    arr = xx[xx==1]
    g[xx==1] = 1 + np.log(0.5)
    arr = xx[xx>1]
    g[xx>1] = np.log(arr/2) + 1/np.sqrt(arr**2-1)*np.arccos(1/arr)
    return g

def Ffunc(x):
    xx = np.atleast_1d(x)
    c1 = xx<1
    c2 = xx==1
    c3 = xx>1
    xx[c1] = 1/(xx[c1]**2-1)*(1 - 1/np.sqrt(1-xx[c1]**2)*np.arccosh(1/xx[c1]))
    xx[c2] = 1/3.
    xx[c3] = 1/(xx[c3]**2-1)*(1 - 1/np.sqrt(xx[c3]**2-1)*np.arccos(1/xx[c3]))
    return xx

def hfunc(x):
    xx = np.atleast_1d(x)
    h = xx*.0
    arr = xx[xx<1]
    h[xx<1] = np.log(arr/2)**2 - np.arccosh(1/arr)**2
    arr = xx[xx>=1]
    h[xx>=1] = np.log(arr/2)**2 - np.arccos(1/arr)**2
    return h


def rho(r,rs):
    return 1./(r/rs)/(1 + r/rs)**2/(4*np.pi*rs**3)/M3d1

def M3d(r,rs):
    return (np.log(1 + r/rs) - r/(r+rs))/M3d1

def Sigma(r,rs):
    return Ffunc(r/rs)/(2*np.pi*rs**2)/M3d1

def M2d(r,rs):
    return gfunc(r/rs)/M3d1

def lenspot(r,rs):
    return hfunc(r/rs)/(2.*np.pi)/M3d1


