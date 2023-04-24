import numpy as np
from scipy.integrate import quad


# assuming flat cosmology

H0 = 100.
h = 0.7
omegaL = 0.7
omegaM = 0.3
omegar = 0.
yr = 365.*3600.*24.

default_cosmo = {'omegaM': omegaM, 'omegaL': omegaL, 'omegar': omegar, 'h': h}

Mpc = 3.08568025e24
kpc = Mpc/1000.
c = 2.99792458e10
G = 6.67300e-8
f_bar = 4.825 / 30.7 # baryon fraction

M_Sun = 1.98892e33

def comovd(z1, z2=0., cosmo=default_cosmo): # comoving distance in Mpc
    for par in default_cosmo:
        if not par in cosmo:
            cosmo[par] = default_cosmo[par]
    omegak = 1. - cosmo['omegaM'] - cosmo['omegaL']

    if z1 > z2:
        z1, z2 = z2, z1
    I = quad(lambda z: 1./(cosmo['omegaL'] + cosmo['omegaM']*(1+z)**3 + cosmo['omegar']*(1+z)**4 + omegak*(1+z)**2)**0.5, z1, z2)
    return c/(H0*cosmo['h']*10.**5)*I[0]

def dcomovdz(z, cosmo=default_cosmo): # derivative of the comoving distance wrt redshift
    for par in default_cosmo:
        if not par in cosmo:
            cosmo[par] = default_cosmo[par]
    omegak = 1. - cosmo['omegaM'] - cosmo['omegaL']

    return c/(H0*cosmo['h']*10.**5)/(cosmo['omegaL'] + cosmo['omegaM']*(1+z)**3 + cosmo['omegar']*(1+z)**4 + omegak*(1+z)**2)**0.5

def Dang(z1, z2=0., cosmo=default_cosmo):
    for par in default_cosmo:
        if not par in cosmo:
            cosmo[par] = default_cosmo[par]
    omegak = 1. - cosmo['omegaM'] - cosmo['omegaL']

    if z1>z2:
        z1, z2 = z2, z1
    cd = comovd(z1, z2, cosmo=cosmo)
    #returns the angular diameter disance in Mpc (default)
    if omegak == 0.:
        D = cd/(1+z2)
    elif omegak > 0.:
        D = c/(H0*cosmo['h']*10.**5)/(abs(omegak))**0.5*np.sinh(omegak**0.5*(H0*cosmo['h']*10.**5)/c*cd)/(1+z2)
    else:
        D = c/(H0*cosmo['h']*10.**5)/(abs(omegak))**0.5*np.sin((-omegak)**0.5*(H0*cosmo['h']*10.**5)/c*cd)/(1+z2)

    return D

def rhoc(z, cosmo=default_cosmo):
    """ critical density of the Universe in M_Sun / Mpc**3"""
    for par in default_cosmo:
        if not par in cosmo:
            cosmo[par] = default_cosmo[par]

    return 3*(H0*cosmo['h']*10**5/Mpc)**2/(8.*np.pi*G)*(cosmo['omegaM']*(1+z)**3 + cosmo['omegaL']) / M_Sun * Mpc**3

def lookback(z, cosmo=default_cosmo):
    for par in default_cosmo:
        if not par in cosmo:
            cosmo[par] = default_cosmo[par]
    omegak = 1. - cosmo['omegaM'] - cosmo['omegaL']
    I = quad(lambda z: 1./(1.+z)/(cosmo['omegaL'] + cosmo['omegaM']*(1+z)**3 + cosmo['omegar']*(1+z)**4 + omegak*(1+z)**2)**0.5, 0., z)
    return I[0]/(H0*cosmo['h']*10.**5*yr/Mpc)

def uniage(z, cosmo=default_cosmo):
    for par in default_cosmo:
        if not par in cosmo:
            cosmo[par] = default_cosmo[par]
    omegak = 1. - cosmo['omegaM'] - cosmo['omegaL']
    I = quad(lambda z: 1./(1.+z)/(cosmo['omegaL'] + cosmo['omegaM']*(1+z)**3 + cosmo['omegar']*(1+z)**4 + omegak*(1+z)**2)**0.5, z, np.inf)
    return I[0]/(H0*cosmo['h']*10.**5*yr/Mpc)

def arcsec2kpc(z, cosmo=default_cosmo):
    return np.deg2rad(1./3600.) * Dang(z, cosmo=cosmo) * 1000.

def Sigma_cr(z1, z2, cosmo=default_cosmo):
    # Critical surface mass density in M_Sun/kpc^2
    dd = Dang(z1)
    ds = Dang(z2)
    dds = Dang(z1, z2)
    
    return c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2

