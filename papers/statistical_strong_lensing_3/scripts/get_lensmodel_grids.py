import numpy as np
from sl_cosmology import Mpc, c, G, M_Sun, yr
import sl_cosmology
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
import h5py
import sys


mockname = '1e5mock_0'

griddir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

maxmagB_det = mockfile.attrs['maxmagB_det']
ms_min = mockfile.attrs['ms_min']
ms_max = mockfile.attrs['ms_max']
mag_err = mockfile.attrs['mag_err']

lenses = mockfile['Lenses']

xA_samp = lenses['xA'][()]
xB_samp = lenses['xB'][()]

magA_obs = lenses['magA_obs'][()]
magB_obs = lenses['magB_obs'][()]

nlens = len(xA_samp)

zs = mockfile.attrs['zs'] # source redshift
ds = sl_cosmology.Dang(zs)
alpha_sch = mockfile.attrs['alpha_sch']
Mstar_sch = mockfile.attrs['Mstar_sch']

mstar_sch = Mstar_sch + 2.5*np.log10(ds**2/1e-10*(1.+zs))

def phifunc(m):
    return (10.**(-0.4*(m - mstar_sch)))**(alpha_sch+1.) * np.exp(-10.**(-0.4*(m - mstar_sch)))
 
# calculates the average number of sources within 1 cubic comoving Mpc
Nbkg_cmpc3 = quad(phifunc, ms_min, ms_max)[0]

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

def mu_tot(x, tein, gamma):
    # tangential magnification
    return mu_r(x, tein, gamma) * mu_t(x, tein, gamma)

def pl_ycaust(gamma):

    tein = 1.

    xmin = 0.01 * tein

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
xmax = 100.

nxB = 101

gamma_min = 1.2
gamma_max = 2.8

ngamma = 801

gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

nms = 251
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

dx = 0.0001
nxB = 101

grids_file = h5py.File(griddir+'/%s_lensmodel_grids.hdf5'%mockname, 'w')

grids_file.create_dataset('gamma_grid', data=gamma_grid)
grids_file.create_dataset('ms_grid', data=ms_grid)

for i in range(nlens):

    tein_grid = ((xA_samp[i] - xB_samp[i]) / (xA_samp[i]**(2.-gamma_grid) + (-xB_samp[i])**(2.-gamma_grid)))**(1./(gamma_grid - 1.))

    detJ_grid = np.zeros(ngamma)
    beta_grid = xA_samp[i] - alpha(xA_samp[i], tein_grid, gamma_grid)

    muA_grid = mu_tot(xA_samp[i], tein_grid, gamma_grid)
    muB_grid = mu_tot(xB_samp[i], tein_grid, gamma_grid)

    mag_integral_grid = np.zeros(ngamma)
    crosssect_grid = np.zeros((ngamma, nms))
    grid_range = np.ones(ngamma, dtype=bool)

    xA_up = xA_samp[i] + dx
    xA_dw = xA_samp[i] - dx
    
    xB_up = xB_samp[i] + dx
    xB_dw = xB_samp[i] - dx

    tein_xA_up_grid = ((xA_up - xB_samp[i]) / (xA_up**(2.-gamma_grid) + (-xB_samp[i])**(2.-gamma_grid)))**(1./(gamma_grid - 1.))
    tein_xA_dw_grid = ((xA_dw - xB_samp[i]) / (xA_dw**(2.-gamma_grid) + (-xB_samp[i])**(2.-gamma_grid)))**(1./(gamma_grid - 1.))

    tein_xB_up_grid = ((xA_samp[i] - xB_up) / (xA_samp[i]**(2.-gamma_grid) + (-xB_up)**(2.-gamma_grid)))**(1./(gamma_grid - 1.))
    tein_xB_dw_grid = ((xA_samp[i] - xB_dw) / (xA_samp[i]**(2.-gamma_grid) + (-xB_dw)**(2.-gamma_grid)))**(1./(gamma_grid - 1.))

    beta_xA_up_grid = xA_up - alpha(xA_up, tein_xA_up_grid, gamma_grid)
    beta_xA_dw_grid = xA_dw - alpha(xA_dw, tein_xA_dw_grid, gamma_grid)

    beta_xB_up_grid = xA_samp[i] - alpha(xA_samp[i], tein_xB_up_grid, gamma_grid)
    beta_xB_dw_grid = xA_samp[i] - alpha(xA_samp[i], tein_xB_dw_grid, gamma_grid)

    dltein_dxA = (np.log10(tein_xA_up_grid) - np.log10(tein_xA_dw_grid))/(2.*dx)
    dltein_dxB = (np.log10(tein_xB_up_grid) - np.log10(tein_xB_dw_grid))/(2.*dx)

    dbeta_dxA = (beta_xA_up_grid - beta_xA_dw_grid)/(2.*dx)
    dbeta_dxB = (beta_xB_up_grid - beta_xB_dw_grid)/(2.*dx)

    detJ_grid = abs(dltein_dxA*dbeta_dxB - dltein_dxB*dbeta_dxA)

    bad = mu_r(xB_samp[i], tein_grid, gamma_grid) < 0.
    grid_range[bad] = False

    for j in range(ngamma):
        xradcrit_here = pl_xradcrit_grid[j] * tein_grid[j]

        if grid_range[j]:

            xB_arr_here = np.linspace(-tein_grid[j], -xradcrit_here, nxB)
            muB_arr_here = mu_tot(xB_arr_here, tein_grid[j], gamma_grid[j])
            beta_arr_here = xB_arr_here - alpha(xB_arr_here, tein_grid[j], gamma_grid[j])

            beta_arr_here[0] = 0.
            beta_caust = beta_arr_here[-1]

            muB_spline = splrep(beta_arr_here, muB_arr_here)

            for l in range(nms):
                magB_arr_here = ms_grid[l] - 2.5*np.log10(abs(muB_arr_here))
                integrand_spline = splrep(beta_arr_here, 2.*np.pi*beta_arr_here * 0.5 * (1. - erf((magB_arr_here - maxmagB_det)/2.**0.5/mag_err)))
                crosssect_grid[j, l] = splint(0., beta_arr_here[-1], integrand_spline)

            magA_here = ms_grid - 2.5*np.log10(abs(muA_grid[j]))
            magB_here = ms_grid - 2.5*np.log10(abs(muB_grid[j]))

            mag_integrand_spline = splrep(ms_grid, phifunc(ms_grid)/Nbkg_cmpc3 * np.exp(-0.5*(magA_here - magA_obs[i])**2/mag_err**2) * np.exp(-0.5*(magB_here - magB_obs[i])**2/mag_err**2) * (1. - erf((magB_here - maxmagB_det)/2.**0.5/mag_err)))
            mag_integral_grid[j] = splint(ms_grid[0], ms_grid[-1], mag_integrand_spline)

    print(i, grid_range.sum())

    group = grids_file.create_group('lens_%04d'%i)

    group.create_dataset('beta_grid', data=beta_grid)
    group.create_dataset('muA_grid', data=muA_grid)
    group.create_dataset('muB_grid', data=muB_grid)
    group.create_dataset('crosssect_grid', data=crosssect_grid)
    group.create_dataset('mag_integral_grid', data=mag_integral_grid)
    group.create_dataset('tein_grid', data=tein_grid)
    group.create_dataset('detJ_grid', data=detJ_grid)
    group.create_dataset('grid_range', data=grid_range)

grids_file.close()


