from sl_profiles import deVaucouleurs as deV, gnfw
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad
import numpy as np


fibre_arcsec = 1.5
seeing_arcsec = 1.5
theta_max = 5.*fibre_arcsec
psf_sigma = seeing_arcsec/2.35

# computes the psf smearing integral
ntheta = 101
theta_grid = np.linspace(0., theta_max, ntheta)
nr = 16
r_grid = np.linspace(0., fibre_arcsec, nr)
smear_grid = np.zeros(ntheta)

for i in range(ntheta):
    def sbfunc(r, phi):
        return 1./(2.*np.pi)/psf_sigma**2 * np.exp(-0.5*((r*np.cos(phi) - theta_grid[i])**2 + r**2*np.sin(phi)**2)/psf_sigma**2)

    r_integrand = np.zeros(nr)
    for l in range(nr):
        r_integrand[l] = r_grid[l] * quad(lambda phi: sbfunc(r_grid[l], phi), 0., 2.*np.pi)[0]
    r_spline = splrep(r_grid, r_integrand)
    smear_grid[i] = splint(0., fibre_arcsec, r_spline)
 
smear_spline = splrep(theta_grid, smear_grid)

# defines lensing-related functions

def alpha_dm(x, gnfw_norm, rs, gamma, s_cr):
    # deflection angle (in kpc)
    return gnfw_norm * gnfw.fast_M2d(abs(x), rs, gamma) / np.pi/x/s_cr

def alpha_star(x, mstar, reff, s_cr): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha_kpc(x, gnfw_norm, rs, gamma, mstar, reff, s_cr):
    return alpha_dm(x, gnfw_norm, rs, gamma, s_cr) + alpha_star(x, mstar, reff, s_cr)

def kappa(x, gnfw_norm, rs, gamma, mstar, reff, s_cr): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + gnfw_norm * gnfw.fast_Sigma(abs(x), rs, gamma))/s_cr
   
def mu_r(x, gnfw_norm, rs, gamma, mstar, reff, s_cr):
    # radial magnification
    return (1. + alpha_kpc(x, gnfw_norm, rs, gamma, mstar, reff, s_cr)/x - 2.*kappa(x, gnfw_norm, rs, gamma, mstar, reff, s_cr))**(-1)

def mu_t(x, gnfw_norm, rs, gamma, mstar, reff, s_cr):
    # tangential magnification
    return (1. - alpha_kpc(x, gnfw_norm, rs, gamma, mstar, reff, s_cr)/x)**(-1)

def get_rein_kpc(mstar, reff, gnfw_norm, rs, gamma, s_cr, xmax=100.):

    def zerofunc(x):
        return x - alpha_kpc(x, gnfw_norm, rs, gamma, mstar, reff, s_cr)

    xmin = 1.01*rs*gnfw.Rgrid_min
    xmax = 0.99*rs*gnfw.Rgrid_max

    if zerofunc(xmin) > 0.:
        rein_kpc = xmin
    elif zerofunc(xmax) < 0.:
        rein_kpc = xmax
    else:
        rein_kpc = brentq(zerofunc, xmin, xmax)

    return rein_kpc

def get_radcaust(mstar, reff, gnfw_norm, rs, gamma, s_cr, rein_kpc):

    def zerofunc(x):
        return 1./mu_r(x, gnfw_norm, rs, gamma, mstar, reff, s_cr)

    xmin = 1.01*rs*gnfw.Rgrid_min
    xmax = 0.99*rs*gnfw.Rgrid_max

    if zerofunc(-xmin) > 0.:
        xrad = xmin
    else:
        xrad = -brentq(zerofunc, -rein_kpc, -xmin)

    radcaust_kpc = -(xrad - alpha_kpc(xrad, gnfw_norm, rs, gamma, mstar, reff, s_cr))

    if radcaust_kpc > 0.:

        def xA_zerofunc(xA):
            return xA - alpha_kpc(xA, gnfw_norm, rs, gamma, mstar, reff, s_cr) - radcaust_kpc
    
        if xA_zerofunc(rein_kpc) * xA_zerofunc(xmax) > 0.:
            rein_here = get_rein_kpc(mstar, reff, gnfw_norm, rs, gamma, s_cr)
    
        xA_max = brentq(xA_zerofunc, rein_kpc, xmax)

    else:
        xrad = 0.
        radcaust_kpc = 0.
        xA_max = 0.

    return xrad, radcaust_kpc, xA_max

def get_crosssect(mstar, reff, gnfw_norm, rs, gamma, s_cr, rein_kpc, xrad_kpc, xA_max, arcsec2kpc, muB_min=1., mufibre_min=3., nxB=1001):
    # computes the lensing cross-section. 
    # Defined as the source-plane area mapped to pairs of images with 
    # magnification larger than muB_min and total flux within the fibre
    # magnified by at least mufibre_min

    xB_grid = np.linspace(-rein_kpc+1e-3, -xrad_kpc, nxB)
    yB_grid = xB_grid - alpha_kpc(xB_grid, gnfw_norm, rs, gamma, mstar, reff, s_cr) # source-plane grid in image-plane kpc

    xA_grid = np.linspace(rein_kpc, xA_max)
    yA_grid = xA_grid - alpha_kpc(xA_grid, gnfw_norm, rs, gamma, mstar, reff, s_cr)

    xA_spline = splrep(yA_grid, xA_grid)

    keep_yB = np.ones(nxB, dtype=bool)
    # check if yB array is monotonically increasing
    for m in range(1, nxB):
        if yB_grid[m] <= yB_grid[:m].max():
            keep_yB[m] = False

    if keep_yB.sum() > 3:
        yB_grid = yB_grid[keep_yB]
        beta_grid = yB_grid / arcsec2kpc
    
        xB_grid = xB_grid[keep_yB]
    
        xA_xBgrid = splev(yB_grid, xA_spline)
    
        thetaB_grid = xB_grid / arcsec2kpc
        thetaA_grid = xA_xBgrid / arcsec2kpc
    
        muA_grid = abs(mu_r(xA_xBgrid, gnfw_norm, rs, gamma, mstar, reff, s_cr) * mu_t(xA_xBgrid, gnfw_norm, rs, gamma, mstar, reff, s_cr))
        muB_grid = abs(mu_r(xB_grid, gnfw_norm, rs, gamma, mstar, reff, s_cr) * mu_t(xB_grid, gnfw_norm, rs, gamma, mstar, reff, s_cr))
    
        muA_seeing_grid = muA_grid * splev(thetaA_grid, smear_spline)
        muB_seeing_grid = muB_grid * splev(thetaB_grid, smear_spline)
        
        mutot_seeing_grid = muA_seeing_grid + muB_seeing_grid
        if np.isnan(mutot_seeing_grid).sum() > 0:
            print(muA_grid)
            print(muB_grid)
            print(mutot_seeing_grid)
            df
        
        good = (mutot_seeing_grid > 3.) & (muB_grid > muB_min)
    
        bad = np.logical_not(good)
        
        integrand_arr = 2.*np.pi*beta_grid
        integrand_arr[bad] = 0.
        
        integrand_spline = splrep(beta_grid, integrand_arr, k=1)
    
        return splint(beta_grid[0], beta_grid[-1], integrand_spline)

    else:
        return 0.

