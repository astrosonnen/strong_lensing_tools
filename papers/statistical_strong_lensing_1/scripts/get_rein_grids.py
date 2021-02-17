import numpy as np
from sl_profiles import gnfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun
import sl_cosmology
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
import h5py
import sys


mockname = 'contrmock_0'

gridsdir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

ngal = mockfile.attrs['ngal']

zd = mockfile.attrs['zd']
zs = mockfile.attrs['zs']

rs_fixed = 100.

gammadm_min = 0.8 
gammadm_max = 1.8

lreff_samp = mockfile['lreff'][()]
rein_true_samp = mockfile['rein'][()]
xA_samp = mockfile['impos'][:, 0]
xB_samp = mockfile['impos'][:, 1]

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

dd = sl_cosmology.Dang(zd)
ds = sl_cosmology.Dang(zs)
dds = sl_cosmology.Dang(zs, zd)

s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2

rhoc = sl_cosmology.rhoc(zd)

# defines lensing-related functions
def alpha_dm(x, gnfw_norm, rs, gammadm):
    # deflection angle (in kpc)
    return gnfw_norm * gnfw.fast_M2d(abs(x), rs, gammadm) / np.pi/x/s_cr

def alpha_star(x, mstar, reff): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(x, gnfw_norm, rs, gammadm, mstar, reff):
    return alpha_dm(x, gnfw_norm, rs, gammadm) + alpha_star(x, mstar, reff)

def kappa(x, gnfw_norm, rs, gammadm, mstar, reff): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + gnfw_norm * gnfw.fast_Sigma(abs(x), rs, gammadm))/s_cr

def mu_r(x, gnfw_norm, rs, gammadm, mstar, reff):
    # radial magnification
    return (1. + alpha(x, gnfw_norm, rs, gammadm, mstar, reff)/x - 2.*kappa(x, gnfw_norm, rs, gammadm, mstar, reff))**(-1)

lmdm5_min = 10.
lmdm5_max = 12.
nlmdm5 = 201
lmdm5_grid = np.linspace(lmdm5_min, lmdm5_max, nlmdm5)

dlm200 = 0.01
dgammadm = 0.01
dx = 0.0001
dx_search = 0.01

Rfrac_min = gnfw.R_grid[0]
Rfrac_max = gnfw.R_grid[-1]

gammadm_grid = np.arange(gammadm_min, gammadm_max, dgammadm)
ngammadm = len(gammadm_grid)

gnfw_norm_grid = np.zeros((ngammadm, nlmdm5))
for i in range(ngammadm):
    gnfw_norm_grid[i, :] = 10.**lmdm5_grid / gnfw.fast_M2d(5., rs_fixed*np.ones(nlmdm5), gammadm_grid[i]*np.ones(nlmdm5))

grids_file = h5py.File(gridsdir+'/%s_rein_grids.hdf5'%mockname, 'w')

grids_file.create_dataset('lmdm5_grid', data=lmdm5_grid)
grids_file.create_dataset('gammadm_grid', data=gammadm_grid)
grids_file.create_dataset('gnfw_norm_grid', data=gnfw_norm_grid)
grids_file.attrs['rs_fixed'] = rs_fixed

for i in range(ngal):

    print(i)

    reff = 10.**lreff_samp[i]

    rein = rein_true_samp[i]

    mstar_frac = deV.M2d(rein, reff)

    lmstar_grid = np.zeros((ngammadm, nlmdm5))
    dlmstar_drein_grid = np.zeros((ngammadm, nlmdm5))
    rmur_grid = np.zeros((ngammadm, nlmdm5))
    grid_range = np.ones((ngammadm, nlmdm5), dtype=bool)

    alpha_star_rein = alpha_star(rein, 1., reff)
    alpha_star_rein_up = alpha_star(rein+dx, 1., reff)
    alpha_star_rein_dw = alpha_star(rein-dx, 1., reff)

    rein_up = rein + dx
    rein_dw = rein - dx
    
    # loops over gammadm
    for j in range(ngammadm):
        # at each point in the halo mass grid, calculates stellar mass needed to obtain the observed Einstein radius
        alpha_dm_rein = alpha_dm(rein, gnfw_norm_grid[j, :], rs_fixed, gammadm_grid[j])
        alpha_dm_rein_up = alpha_dm(rein_up, gnfw_norm_grid[j, :], rs_fixed, gammadm_grid[j])
        alpha_dm_rein_dw = alpha_dm(rein_dw, gnfw_norm_grid[j, :], rs_fixed, gammadm_grid[j])

        mstar_grid_here = (rein - alpha_dm_rein)/alpha_star_rein

        lmstar_grid[j, :] = np.log10(mstar_grid_here)
    
        grid_range[j, :] = mstar_grid_here > 1e9

        mstar_rein_up_grid = (rein_up - alpha_dm_rein_up)/alpha_star_rein_up
        mstar_rein_dw_grid = (rein_dw - alpha_dm_rein_dw)/alpha_star_rein_dw
   
        dlmstar_drein = (np.log10(mstar_rein_up_grid) - np.log10(mstar_rein_dw_grid))/(2.*dx)

        dlmstar_drein_grid[j, :] = abs(dlmstar_drein)

        mu_rA = mu_r(xA_samp[i], gnfw_norm_grid[j], rs_fixed, gammadm_grid[j], mstar_grid_here, reff)
        mu_rB = mu_r(xB_samp[i], gnfw_norm_grid[j], rs_fixed, gammadm_grid[j], mstar_grid_here, reff)
        rmur_grid[j, :] = mu_rA/mu_rB

    group = grids_file.create_group('lens_%04d'%i)

    group.create_dataset('lmstar_grid', data=lmstar_grid)
    group.create_dataset('rmur_grid', data=rmur_grid)
    group.create_dataset('dlmstar_drein_grid', data=dlmstar_drein_grid)
    group.create_dataset('grid_range', data=grid_range)

grids_file.close()


