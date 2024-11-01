import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1
from sl_profiles import deVaucouleurs as deV, gnfw, nfw
from scipy.interpolate import splrep, splev


def adcontr_m3d(r, mstar, reff, mhalo, rs, rvir, eps=1.):

    # computes the 3d enclosed mass within radius r,
    # for an adiabatically contracted dark matter halo
    # with contraction efficiency eps

    nfw_norm = mhalo / nfw.M3d(rvir, rs)
    def zerofunc(ri):
        # finds the initial radius of the shell at current radius r
        return nfw_norm * nfw.M3d(ri, rs) * ((ri -r) * (1. + eps*mstar/mhalo) + r*eps*mstar/mhalo) - r*eps*mstar*deV.fast_M3d(r/reff)

    ri = brentq(zerofunc, r, rvir)

    return ri, nfw_norm * nfw.M3d(ri, rs)

def get_rho_slope_e(mstar, reff, mhalo, rs, rvir, eps=1., dr=1e-3):

    # computes density and density slope at r=reff

    ri_e, mdm_e = adcontr_m3d(reff, mstar, reff, mhalo, rs, rvir, eps=eps)

    ri_e_up, mdm_e_up = adcontr_m3d(reff + dr, mstar, reff, mhalo, rs, rvir, eps=eps)
    ri_e_dw, mdm_e_dw = adcontr_m3d(reff - dr, mstar, reff, mhalo, rs, rvir, eps=eps)

    dmdr_e = (mdm_e_up - mdm_e_dw)/(2.*dr)
    d2mdr2_e = (mdm_e_up - 2.*mdm_e + mdm_e_dw)/dr**2

    rho_e = dmdr_e / (4*np.pi*reff**2)
    logslope_e = reff/dmdr_e * d2mdr2_e - 2. # logarithmic slope

    return rho_e, logslope_e

def find_gnfw(mstar, reff, mhalo, rs, rvir, eps=1.):

    # finds the gnfw profile with the same halo mass,
    # same density and density slope at r=reff

    cvir = rvir/rs

    rho_e, logslope_e = get_rho_slope_e(mstar, reff, mhalo, rs, rvir, eps=eps)

    gamma_max = -logslope_e - 1e-3

    def rsfunc(gamma):
        # rs as a function of gamma, given the logarithmic slope at reff
        return reff * (3. + logslope_e)/(-logslope_e - gamma)

    def norm_func(gamma):
        # Density normalisation.
        # If multiplied by 1/r^gamma/(1+r/rs)^(3-gamma), it returns the density
        #return rho_e * reff**gamma * (1. + (-logslope_e - gamma)/(3. + logslope_e))**(3.-gamma)
        return rho_e * reff**gamma * (1. + reff/rsfunc(gamma))**(3.-gamma)

    def integral_func(gamma):
        return (rvir/rsfunc(gamma))**(3. - gamma)/(3.-gamma) * hyp2f1(3.-gamma, 3.-gamma, 4.-gamma, -rvir/rsfunc(gamma))   

    def zerofunc(gamma):
        return norm_func(gamma) * 4.*np.pi * rsfunc(gamma)**(3.-gamma) * integral_func(gamma) - mhalo

    if zerofunc(0.) * zerofunc(gamma_max) < 0.:
        gamma = brentq(zerofunc, 0., gamma_max)
        rs = rsfunc(gamma)
    else:
        gamma = -1.
        rs = -1.

    return rs, gamma

