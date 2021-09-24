import numpy as np
from sl_profiles import nfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun, yr
import sl_cosmology
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
import h5py
import sys


mockname = '1e5mock_0'

griddir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

maxmagB_det = mockfile.attrs['maxmagB_det']
mag_err = mockfile.attrs['mag_err']
ms_min = mockfile.attrs['ms_min']
ms_max = mockfile.attrs['ms_max']

tein = 1.

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

def pl_ycaust(gamma):

    xmin = 0.01

    def radial_invmag(x):
        return 1. + alpha(x, 1., gamma)/x - 2.*kappa(x, 1., gamma)

    # finds the radial caustic
    if radial_invmag(xmin)*radial_invmag(tein) > 0.:
        xradcrit = xmin
    else:
        xradcrit = brentq(radial_invmag, xmin, tein)

    ycaust = -(xradcrit - alpha(xradcrit, 1., gamma))

    return ycaust, xradcrit

xmin = 0.01
xmax = 100.

nxB = 101

gamma_min = 1.2
gamma_max = 2.8

ngamma = 81

gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

nms = 101
ms_grid = np.linspace(ms_min, ms_max, nms)

dx = 0.0001

pl_ycaust_grid = 0. * gamma_grid
pl_xradcrit_grid = 0. * gamma_grid
for i in range(ngamma):
    ycaust, xradcrit = pl_ycaust(gamma_grid[i])
    pl_ycaust_grid[i] = ycaust
    pl_xradcrit_grid[i] = xradcrit

pl_ycaust_spline = splrep(gamma_grid, pl_ycaust_grid)
pl_xradcrit_spline = splrep(gamma_grid, pl_xradcrit_grid)

grid_file = h5py.File(griddir+'/%s_crosssect_grid.hdf5'%mockname, 'w')

grid_file.create_dataset('gamma_grid', data=gamma_grid)
grid_file.create_dataset('ms_grid', data=ms_grid)
grid_file.create_dataset('pl_ycaust_grid', data=pl_ycaust_grid)
grid_file.create_dataset('pl_xradcrit_grid', data=pl_xradcrit_grid)

beta_caust_grid = np.zeros(ngamma)
crosssect_grid = np.zeros((ngamma, nms))

for i in range(ngamma):

    xradcrit = pl_xradcrit_grid[i]

    xB_arr_here = np.linspace(-1., -xradcrit, nxB)
    muB_arr_here = mu_r(xB_arr_here, 1., gamma_grid[i]) * mu_t(xB_arr_here, 1., gamma_grid[i])
    beta_arr_here = xB_arr_here - alpha(xB_arr_here, 1., gamma_grid[i])

    beta_arr_here[0] = 0.
    beta_caust = beta_arr_here[-1]

    beta_caust_grid[i] = beta_caust

    muB_spline = splrep(beta_arr_here, muB_arr_here)

    for l in range(nms):
        magB_arr_here = ms_grid[l] - 2.5*np.log10(abs(muB_arr_here))
        integrand_spline = splrep(beta_arr_here, 2.*np.pi*beta_arr_here * 0.5 * (1. - erf((magB_arr_here - maxmagB_det)/2.**0.5/mag_err)))
        crosssect_grid[i, l] = splint(0., beta_arr_here[-1], integrand_spline)

grid_file.create_dataset('crosssect_grid', data=crosssect_grid)
grid_file.create_dataset('beta_caust_grid', data=beta_caust_grid)

grid_file.close()

