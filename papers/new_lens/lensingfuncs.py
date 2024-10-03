from sl_profiles import deVaucouleurs as deV, nfw
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
import numpy as np


# defines lensing-related functions

def pfind(tein, tein_min=0.1):
    # lens-finding probability as a function of the Einstein radius
    return np.heaviside(tein - tein_min, 1.)

def alpha_dm(x, nfw_norm, rs, s_cr):
    # deflection angle (in kpc)
    return nfw_norm * nfw.M2d(abs(x), rs) / np.pi/x/s_cr

def alpha_star(x, mstar, reff, s_cr): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(x, nfw_norm, rs, mstar, reff, s_cr):
    return alpha_dm(x, nfw_norm, rs, s_cr) + alpha_star(x, mstar, reff, s_cr)

def kappa(x, nfw_norm, rs, mstar, reff, s_cr): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + nfw_norm * nfw.Sigma(abs(x), rs))/s_cr
   
def mu_r(x, nfw_norm, rs, mstar, reff, s_cr):
    # radial magnification
    return (1. + alpha(x, nfw_norm, rs, mstar, reff, s_cr)/x - 2.*kappa(x, nfw_norm, rs, mstar, reff, s_cr))**(-1)

def mu_t(x, nfw_norm, rs, mstar, reff, s_cr):
    # tangential magnification
    return (1. - alpha(x, nfw_norm, rs, mstar, reff, s_cr)/x)**(-1)

xmin = 0.01
xmax = 100.

def get_rein_kpc(mstar, reff, nfw_norm, rs, s_cr):

    def zerofunc(x):
        return x - alpha(x, nfw_norm, rs, mstar, reff, s_cr)

    if zerofunc(xmin) > 0.:
        rein_kpc = xmin
    elif zerofunc(xmax) < 0.:
        rein_kpc = xmax
    else:
        rein_kpc = brentq(zerofunc, xmin, xmax)

    return rein_kpc

def get_radcaust(mstar, reff, nfw_norm, rs, s_cr, rein_kpc):

    def zerofunc(x):
        return 1./mu_r(x, nfw_norm, rs, mstar, reff, s_cr)

    if zerofunc(-xmin) > 0.:
        xrad = xmin
    else:
        xrad = -brentq(zerofunc, -rein_kpc, -xmin)

    radcaust_kpc = -(xrad - alpha(xrad, nfw_norm, rs, mstar, reff, s_cr))
    return xrad, radcaust_kpc

def get_crosssect(mstar, reff, nfw_norm, rs, s_cr, rein_kpc, xrad_kpc, arcsec2kpc, muB_min=1., nxB=1001):
    # computes the lensing cross-section. 
    # Defined as the source-plane area mapped to pairs of images with 
    # magnification larger than muB_min.

    # ray-traces from 2nd image back to the source plane
    xB_grid = np.linspace(-rein_kpc, -xrad_kpc, nxB)
    y_grid = xB_grid - alpha(xB_grid, nfw_norm, rs, mstar, reff, s_cr)
    beta_grid = y_grid / arcsec2kpc
    beta_grid[0] = 0.

    # maintains an invertible mapping
    good_beta = np.ones(nxB, dtype=bool)
    beta_highest = 0.
    for m in range(1, nxB):
        if beta_grid[m] > beta_highest:
            beta_highest = beta_grid[m]
        else:
            good_beta[m] = False

    xB_grid = xB_grid[good_beta]
    beta_grid = beta_grid[good_beta]

    # computes the magnification of 2nd image
    muB_grid = mu_r(xB_grid, nfw_norm, rs, mstar, reff, s_cr) * mu_t(xB_grid, nfw_norm, rs, mstar, reff, s_cr)

    # integrates source plane area mapped to images with magnification larger than muB_min
    integrand = 2.*np.pi * beta_grid
    integrand[abs(muB_grid) < muB_min] = 0.

    if good_beta.sum() > 3:
        integrand_spline = splrep(beta_grid, integrand, k=1)
        return splint(beta_grid[0], beta_grid[-1], integrand_spline)
    else:
        return 0.

