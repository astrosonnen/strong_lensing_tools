# scripts to calculate lensing-related quantities for a dual Pseudo Isothermal Spherical mass distribution

import numpy as np
from sl_cosmology import G, M_Sun, Mpc


# rc = core radius; rt = truncation radius
# rt > rc must be satisfied

def Mtot(rc, rt):
    return 2.*pi**2*rc**2*rt**2/(rc + rt)

def rho(r, rc, rt):
    return (rc + rt)/(2.*np.pi**2*rc**2*rt**2) / (1. + r**2/rc**2) / (1. + r**2/rt**2)

def M3d(r, rc, rt):
    return 2./(np.pi*(rt - rc)) * (rt*np.arctan(r/rt) - rc*np.arctan(r/rc))

def Sigma(R, rc, rt):
    return 1./(2.*np.pi*(rt - rc)) * (1./(rc**2 + R**2)**0.5 - 1./(rt**2 + R**2)**0.5)

def M2d(R, rc, rt):
    return 1./(rt - rc) * ((rc**2 + R**2)**0.5 - rc - (rt**2 + R**2)**0.5 + rt)

def sigma_v_from_Mtot(Mtot, rt):
    # converts the total mass into the velocity dispersion
    # input: Mtot in Solar masses, rt in kpc
    # returns: sigma_v in km/s
    return (G*Mtot*M_Sun/np.pi/(rt * Mpc / 1000.))**0.5 / 1e5

def Mtot_from_sigma_v(sigma_v, rt):
    # converts the total mass into the velocity dispersion
    # input: sigma_v in km/s, rt in kpc
    # returns: Mtot in Solar masses
    return (sigma_v * 1e5)**2/G * np.pi * rt * Mpc / 1000. / M_Sun

