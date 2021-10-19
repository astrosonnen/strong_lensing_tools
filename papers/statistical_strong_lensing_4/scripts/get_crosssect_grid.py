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


mockname = '1e4mock_0'

griddir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

zd = mockfile.attrs['zd']
c200 = mockfile.attrs['c200']

zs_grid = mockfile['Grids/zs_grid'][()]
pz_grid = mockfile['Grids/pz_grid'][()]
mstar_sch_grid = mockfile['Grids/mstar_sch_grid'][()]
s_cr_grid = mockfile['Grids/s_cr_grid'][()]
ms_grid = mockfile['Grids/ms_grid'][()]
dvol_dz_grid = mockfile['Grids/dvol_dz_grid'][()]

pz_spline = splrep(zs_grid, pz_grid)
dvol_dz_spline = splrep(zs_grid, dvol_dz_grid)
s_cr_spline = splrep(zs_grid, s_cr_grid)
mstar_sch_spline = splrep(zs_grid, mstar_sch_grid)

lreff_mu = mockfile.attrs['lreff_mu']
lreff_beta = mockfile.attrs['lreff_beta']
lreff_sig = mockfile.attrs['lreff_sig']
lmstar_piv = mockfile.attrs['lmstar_piv']

maxmagB_det = mockfile.attrs['maxmagB_det']
mag_err = mockfile.attrs['mag_err']
ms_min = mockfile.attrs['ms_min']
ms_max = mockfile.attrs['ms_max']

dd = sl_cosmology.Dang(zd)

rhoc = sl_cosmology.rhoc(zd)

# defines lensing-related functions
def alpha_dm(s_cr, x, nfw_norm, rs):
    # deflection angle (in kpc)
    return nfw_norm * nfw.M2d(abs(x), rs) / np.pi/x/s_cr

def alpha_star(s_cr, x, mstar, reff): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(s_cr, x, nfw_norm, rs, mstar, reff):
    return alpha_dm(s_cr, x, nfw_norm, rs) + alpha_star(s_cr, x, mstar, reff)

def kappa(s_cr, x, nfw_norm, rs, mstar, reff): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + nfw_norm * nfw.Sigma(abs(x), rs))/s_cr
   
def mu_r(s_cr, x, nfw_norm, rs, mstar, reff):
    # radial magnification
    return (1. + alpha(s_cr, x, nfw_norm, rs, mstar, reff)/x - 2.*kappa(s_cr, x, nfw_norm, rs, mstar, reff))**(-1)

def mu_t(s_cr, x, nfw_norm, rs, mstar, reff):
    # tangential magnification
    return (1. - alpha(s_cr, x, nfw_norm, rs, mstar, reff)/x)**(-1)

lm200_min = 11.
lm200_max = 15.

nlm200 = 41
lm200_grid = np.linspace(lm200_min, lm200_max, nlm200)
r200_grid = (10.**lm200_grid*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
rs_grid = r200_grid/c200
nfw_norm_grid = 10.**lm200_grid/nfw.M3d(r200_grid, rs_grid)

lmstar_min = 10.5
lmstar_max = 12.5

nlmstar = 41
lmstar_grid = np.linspace(lmstar_min, lmstar_max, nlmstar)

nlreff = 47
lreff_grid = np.linspace(-0.1, 2., nlreff)

nms = 61
ms_grid = np.linspace(ms_min, ms_max, nms)

nbeta = 101
beta_grid = np.logspace(-2., 2., nbeta)

nzs = len(zs_grid)

dx = 0.0001
dx_search = 0.01
nxB = 101

grid_file = h5py.File(griddir+'/%s_crosssect_grid.hdf5'%mockname, 'w')

grid_file.attrs['c200'] = c200
grid_file.create_dataset('lm200_grid', data=lm200_grid)
grid_file.create_dataset('nfw_norm_grid', data=nfw_norm_grid)
grid_file.create_dataset('lmstar_grid', data=lmstar_grid)
grid_file.create_dataset('lreff_grid', data=lreff_grid)
grid_file.create_dataset('ms_grid', data=ms_grid)
grid_file.create_dataset('zs_grid', data=zs_grid)
grid_file.create_dataset('beta_grid', data=beta_grid)

rein_grid = np.zeros((nzs, nlm200, nlmstar, nlreff))
crosssect_grid = np.zeros((nzs, nlm200, nlmstar, nlreff, nms))
muB_grid = np.zeros((nzs, nlm200, nlmstar, nlreff, nbeta))
xradcrit_grid = np.zeros((nzs, nlm200, nlmstar, nlreff))
beta_caust_grid = np.zeros((nzs, nlm200, nlmstar, nlreff))

for s in range(nzs):

    for i in range(nlm200):
        print(s, i)
        nfw_norm = nfw_norm_grid[i]
        rs = rs_grid[i]
    
        for j in range(nlmstar):
            mstar = 10.**lmstar_grid[j]
    
            for k in range(nlreff):
                reff = 10.**lreff_grid[k]
                xmin = deV.rgrid_min*reff
                xmax = deV.rgrid_max*reff
    
                muB_grid_here = np.zeros(nbeta)
     
                def zerofunc(x):
                    return alpha(s_cr_grid[s], x, nfw_norm, rs, mstar, reff) - x
                    
                if zerofunc(xmin) < 0.:
                    rein = 0.
                    xradcrit = 0.
                    beta_caust = 0.
                elif zerofunc(xmax) > 0.:
                    rein = np.inf
                    xradcrit = 0.
                    beta_caust = 0.
                else:
                    rein = brentq(zerofunc, xmin, xmax)
            
                    def radial_invmag(x):
                        return 1. + alpha(s_cr_grid[s], x, nfw_norm, rs, mstar, reff)/x - 2.*kappa(s_cr_grid[s], x, nfw_norm, rs, mstar, reff)
                
                    # finds the radial caustic
                    if radial_invmag(xmin)*radial_invmag(rein) > 0.:
                        xradcrit = xmin
                    else:
                        xradcrit = brentq(radial_invmag, xmin, rein)
                
                    xB_arr_here = np.linspace(-rein, -xradcrit, nxB)
                    muB_arr_here = mu_r(s_cr_grid[s], xB_arr_here, nfw_norm, rs, mstar, reff) * mu_t(s_cr_grid[s], xB_arr_here, nfw_norm, rs, mstar, reff)
                    beta_arr_here = xB_arr_here - alpha(s_cr_grid[s], xB_arr_here, nfw_norm, rs, mstar, reff)
    
                    # if there is more than one radial caustic (how??)
                    beta_highest = beta_arr_here[0]
                    trimmed = np.ones(nxB, dtype=bool)
                    for m in range(1, nxB):
                        if beta_arr_here[m] > beta_highest:
                            beta_highest = beta_arr_here[m]
                        else:
                            trimmed[m] = False
    
                    xB_arr_here = xB_arr_here[trimmed]
                    muB_arr_here = muB_arr_here[trimmed]
                    beta_arr_here = beta_arr_here[trimmed]
    
                    beta_arr_here[0] = 0.
                    beta_caust = beta_arr_here[-1]
    
                    muB_spline = splrep(beta_arr_here, muB_arr_here)
    
                    lensed = beta_grid < beta_caust
                    if lensed.sum() > 0:
                        muB_grid_here[lensed] = splev(beta_grid[lensed], muB_spline)
    
                    for l in range(nms):
                        magB_arr_here = ms_grid[l] - 2.5*np.log10(abs(muB_arr_here))
                        integrand_spline = splrep(beta_arr_here, 2.*np.pi*beta_arr_here * 0.5 * (1. - erf((magB_arr_here - maxmagB_det)/2.**0.5/mag_err)))
                        crosssect_grid[s, i, j, k, l] = splint(0., beta_arr_here[-1], integrand_spline)
    
                rein_grid[s, i, j, k] = rein
                xradcrit_grid[s, i, j, k] = xradcrit
                beta_caust_grid[s, i, j, k] = beta_caust
                muB_grid[s, i, j, k, :] = muB_grid_here

grid_file.create_dataset('rein_grid', data=rein_grid)
grid_file.create_dataset('crosssect_grid', data=crosssect_grid)
grid_file.create_dataset('muB_grid', data=muB_grid)
grid_file.create_dataset('beta_caust_grid', data=beta_caust_grid)
grid_file.create_dataset('xradcrit_grid', data=xradcrit_grid)

grid_file.close()

