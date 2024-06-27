import numpy as np
import h5py
import sl_cosmology
from parent_sample_pars import *
from scipy.interpolate import splrep, splev, splint


chain = h5py.File('mz_inference.hdf5', 'r')

burnin = 200
# takes the median of the marginal posterior of each parameter
mbar = np.median(chain['mbar'][:, burnin:])
alpha = np.median(chain['alpha'][:, burnin:])
mt0 = np.median(chain['mt0'][:, burnin:])
mt1 = np.median(chain['mt1'][:, burnin:])
mt2 = np.median(chain['mt2'][:, burnin:])
mt3 = np.median(chain['mt3'][:, burnin:])
mt4 = np.median(chain['mt4'][:, burnin:])
mt5 = np.median(chain['mt5'][:, burnin:])
sigmat = np.median(chain['sigmat'][:, burnin:])

# computes the derivative of the comoving volume with redshift
nzd = 36
zd_grid = np.linspace(zmin, zmax, nzd)
arcsec2kpc_grid = np.zeros(nzd)
dvdz_grid = np.zeros(nzd)
for n in range(nzd):
    dvdz_grid[n] = sl_cosmology.comovd(zd_grid[n])**2 * sl_cosmology.dcomovdz(zd_grid[n])
    arcsec2kpc_grid[n] = sl_cosmology.arcsec2kpc(zd_grid[n])

dvdz_spline = splrep(zd_grid, dvdz_grid)
arcsec2kpc_spline = splrep(zd_grid, arcsec2kpc_grid)

# prepares a grid in stellar mass
nms = 101
ms_grid = np.linspace(lmchab_min, lmchab_max, nms)

# defines the stellar mass-redshift function
def mtfunc(z):
    return mt0 + mt1*z + mt2*z**2 + mt3*z**3 + mt4*z**4 + mt5*z**5

def ftfunc(z, ms):
    return 1./np.pi * np.arctan((ms - mtfunc(z))/sigmat) + 0.5

def msdist(z, ms):
    return splev(z, dvdz_spline) * ftfunc(z, ms) * (10.**(ms - mbar))**(alpha + 1) * np.exp(-10.**(ms - mbar))

# at each redshift, integrates the distribution over the stellar mass
# (computes the number of galaxies at each redshift interval)
msint_grid = np.zeros(nzd)
for n in range(nzd):
    integrand_grid = msdist(zd_grid[n], ms_grid)
    integrand_spline = splrep(ms_grid, integrand_grid)
    msint_grid[n] = splint(ms_grid[0], ms_grid[-1], integrand_spline)

msint_spline = splrep(zd_grid, msint_grid) # this is the marginal z distribution

invnorm = splint(zd_grid[0], zd_grid[-1], msint_spline) # normalisation

# computes the cumulative probability in redshift
pzcum_grid = np.zeros(nzd)
for n in range(nzd):
    pzcum_grid[n] = splint(zmin, zd_grid[n], msint_spline)/invnorm

# prepares a spline with the inverse cumulative probability
invpzcum_spline = splrep(pzcum_grid, zd_grid)

def draw_mz(npop):
    # draw values of z
    zd_popsamp = splev(np.random.rand(npop), invpzcum_spline)
    arcsec2kpc_popsamp = splev(zd_popsamp, arcsec2kpc_spline)
    t_popsamp = np.random.rand(npop)
    
    ms_popsamp = np.zeros(npop)
    
    # draws values of ms
    for i in range(npop):
        msfunc_grid = msdist(zd_popsamp[i], ms_grid)
        msfunc_spline = splrep(ms_grid, msfunc_grid, k=1)
    
        cump_grid = 0. * msfunc_grid
        for j in range(nms):
            cump_grid[j] = splint(ms_grid[0], ms_grid[j], msfunc_spline)
    
        cump_grid /= cump_grid[-1]
    
        invcump_spline = splrep(cump_grid, ms_grid, k=1)
      
        ms_popsamp[i] = splev(t_popsamp[i], invcump_spline)

    return zd_popsamp, ms_popsamp, arcsec2kpc_popsamp

