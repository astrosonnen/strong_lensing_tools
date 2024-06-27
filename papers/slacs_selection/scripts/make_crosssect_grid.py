import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad
from parent_sample_pars import *
from fitpars import gamma_min, gamma_max
import h5py


# lensing cross-section of an axisymmetric power-law lens
# with unit Einstein radius and source flux equal to the
# detection limit of the survey.
# Takes into account the finite fibre size: only image configurations
# with at least one image within 1.5 arcsec AND
# total flux (1st + 2nd image) larger than 3 are considered.

muB_min = 1.

arcsec2rad = np.deg2rad(1./3600.)

# defines lensing-related functions
def alpha(x, tein, gamma):
    return tein * x/abs(x) * (abs(x)/tein)**(2.-gamma)

def kappa(x, tein, gamma): 
    # dimensionless surface mass density
    return (3.-gamma)/2. * (abs(x)/tein)**(1.-gamma)

def mu_r(x, tein, gamma):
    # radial magnification
    return (1. + alpha(x, tein, gamma)/x - 2.*kappa(x, tein, gamma))**(-1)

def mu_t(x, tein, gamma):
    # tangential magnification
    return (1. - alpha(x, tein, gamma)/x)**(-1)

def pl_ycaust(tein, gamma):

    xmin = 0.01

    def radial_invmag(x):
        return 1. + alpha(x, tein, gamma)/x - 2.*kappa(x, tein, gamma)

    # finds the radial caustic
    if radial_invmag(xmin)*radial_invmag(tein) > 0.:
        xradcrit = xmin
    else:
        xradcrit = brentq(radial_invmag, xmin, tein)

    ycaust = -(xradcrit - alpha(xradcrit, tein, gamma))

    return ycaust, xradcrit

xmin = 0.01
xmax = 5.*fibre_arcsec

ngamma = 81
gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

tein_min = 0.
tein_max = 5.
ntein = 51
tein_grid = np.linspace(tein_min, tein_max, ntein)

dx = 0.0001

grid_file = h5py.File('fibre_crosssect_grid.hdf5', 'w')

grid_file.create_dataset('tein_grid', data=tein_grid)
grid_file.create_dataset('gamma_grid', data=gamma_grid)
ycaust_grid = np.zeros((ntein, ngamma))
mufibre2_cs_grid = np.zeros((ntein, ngamma))
mufibre3_cs_grid = np.zeros((ntein, ngamma))

nbeta = 1001

psf_sigma = seeing_arcsec/2.35

nr = 16
r_grid = np.linspace(0., fibre_arcsec, nr)

for i in range(1, ntein):

    print(i)

    for j in range(ngamma):
        ycaust, xradcrit = pl_ycaust(tein_grid[i], gamma_grid[j])

        ycaust_grid[i, j] = ycaust

        #xB_grid = np.linspace(-min(fibre_arcsec, tein_grid[i]), -xradcrit, nbeta)
        xB_grid = np.linspace(-tein_grid[i], -xradcrit, nbeta)
        muB_grid = abs(mu_r(xB_grid, tein_grid[i], gamma_grid[j]) * mu_t(xB_grid, tein_grid[i], gamma_grid[j]))
        beta_grid = xB_grid - alpha(xB_grid, tein_grid[i], gamma_grid[j])

        xA_grid = 0.*beta_grid + np.inf
        muA_grid = 0.*beta_grid
        muA_seeing_grid = 0.*beta_grid
        muB_seeing_grid = 0.*beta_grid

        for k in range(1, nbeta):
            # solves the lens equation 
            def xA_zerofunc(xA):
                return xA - alpha(xA, tein_grid[i], gamma_grid[j]) - beta_grid[k]
            #if (tein_grid[i] < xmax) and (xA_zerofunc(xmax) >= 0.):
            if xA_zerofunc(xmax) >= 0.:
                xA_here = brentq(xA_zerofunc, tein_grid[i], xmax)
                muA_grid[k] = abs(mu_r(xA_here, tein_grid[i], gamma_grid[j]) * mu_t(xA_here, tein_grid[i], gamma_grid[j]))
                xA_grid[k] = xA_here

                def muA_func(r, phi):
                    return 1./(2.*np.pi)/psf_sigma**2 * np.exp(-0.5*((r*np.cos(phi) - xA_here)**2 + r**2*np.sin(phi)**2)/psf_sigma**2) * abs(mu_r(xA_here, tein_grid[i], gamma_grid[j]))

                muA_integrand = np.zeros(nr)
                for l in range(nr):
                    muA_integrand[l] = r_grid[l] * quad(lambda phi: muA_func(r_grid[l], phi), 0., 2.*np.pi)[0] * abs(mu_t(xA_here, tein_grid[i], gamma_grid[j]))

                muA_spline = splrep(r_grid, muA_integrand)
                muA_seeing_grid[k] = splint(0., r_grid[-1], muA_spline)

                def muB_func(r, phi):
                    return 1./(2.*np.pi)/psf_sigma**2 * np.exp(-0.5*((r*np.cos(phi) - xB_grid[k])**2 + r**2*np.sin(phi)**2)/psf_sigma**2) * abs(mu_r(xB_grid[k], tein_grid[i], gamma_grid[j]))

                muB_integrand = np.zeros(nr)
                for l in range(nr):
                    muB_integrand[l] = r_grid[l] * quad(lambda phi: muB_func(r_grid[l], phi), 0., 2.*np.pi)[0] * abs(mu_t(xB_grid[k], tein_grid[i], gamma_grid[j]))

                muB_spline = splrep(r_grid, muB_integrand)
                muB_seeing_grid[k] = splint(0., r_grid[-1], muB_spline)

        mutot_seeing_grid = muA_seeing_grid + muB_seeing_grid

        good = (mutot_seeing_grid > 2.) & (muB_grid > muB_min)
        bad = np.logical_not(good)

        integrand_arr = 2.*np.pi*beta_grid
        integrand_arr[bad] = 0.

        integrand_spline = splrep(beta_grid, integrand_arr, k=1)
        mufibre2_cs_grid[i, j] = splint(beta_grid[0], beta_grid[-1], integrand_spline)

        good = (mutot_seeing_grid > 3.) & (muB_grid > muB_min)
        bad = np.logical_not(good)

        integrand_arr = 2.*np.pi*beta_grid
        integrand_arr[bad] = 0.

        integrand_spline = splrep(beta_grid, integrand_arr, k=1)
        mufibre3_cs_grid[i, j] = splint(beta_grid[0], beta_grid[-1], integrand_spline)

grid_file.create_dataset('mufibre2_cs_grid', data=mufibre2_cs_grid)
grid_file.create_dataset('mufibre3_cs_grid', data=mufibre3_cs_grid)
grid_file.create_dataset('ycaust_grid', data=ycaust_grid)

grid_file.attrs['muB_min'] = muB_min

grid_file.close()

