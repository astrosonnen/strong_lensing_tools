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

gridsdir = './' # destination path of output file (will take up several GB)

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

ngal = mockfile.attrs['ngal']

zd = mockfile.attrs['zd']
zs = mockfile.attrs['zs']

rs_fixed = 100. # scale radius in kpc

# boundaries of inner dark matter slope
gammadm_min = 0.8
gammadm_max = 1.8

minmag = mockfile.attrs['minmag']

lreff_samp = mockfile['lreff'][()]
xA_samp = mockfile['impos'][:, 0]
xB_samp = mockfile['impos'][:, 1]

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

# angular diameter distances
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

def mu_t(x, gnfw_norm, rs, gammadm, mstar, reff):
    # tangential magnification
    return (1. - alpha(x, gnfw_norm, rs, gammadm, mstar, reff)/x)**(-1)

lmdm5_min = 10.
lmdm5_max = 12.
nlmdm5 = 201
lmdm5_grid = np.linspace(lmdm5_min, lmdm5_max, nlmdm5)

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

grids_file = h5py.File(gridsdir+'/%s_lensmodel_grids.hdf5'%mockname, 'w')

grids_file.attrs['rs'] = rs_fixed
grids_file.create_dataset('lmdm5_grid', data=lmdm5_grid)
grids_file.create_dataset('gammadm_grid', data=gammadm_grid)
grids_file.create_dataset('gnfw_norm_grid', data=gnfw_norm_grid)

for i in range(ngal):

    reff = 10.**lreff_samp[i]

    lmstar_grid = np.zeros((ngammadm, nlmdm5))
    detJ_grid = np.zeros((ngammadm, nlmdm5))
    rmur_grid = np.zeros((ngammadm, nlmdm5))
    beta_grid = np.zeros((ngammadm, nlmdm5))
    beta_max_grid = np.zeros((ngammadm, nlmdm5))
    grid_range = np.ones((ngammadm, nlmdm5), dtype=bool)
    xmin_grid = np.zeros((ngammadm, nlmdm5))

    alpha_star_xA = alpha_star(xA_samp[i], 1., reff)
    alpha_star_xA_up = alpha_star(xA_samp[i]+dx, 1., reff)
    alpha_star_xA_dw = alpha_star(xA_samp[i]-dx, 1., reff)

    alpha_star_xB = alpha_star(xB_samp[i], 1., reff)
    alpha_star_xB_up = alpha_star(xB_samp[i]+dx, 1., reff)
    alpha_star_xB_dw = alpha_star(xB_samp[i]-dx, 1., reff)

    xA_up = xA_samp[i] + dx
    xA_dw = xA_samp[i] - dx
    
    xB_up = xB_samp[i] + dx
    xB_dw = xB_samp[i] - dx

    # loops over gammadm
    for j in range(ngammadm):
        # at each point in the halo mass grid, calculates stellar mass needed to obtain the observed image positions
        alpha_dm_xA = alpha_dm(xA_samp[i], gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j])
        alpha_dm_xA_up = alpha_dm(xA_samp[i]+dx, gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j])
        alpha_dm_xA_dw = alpha_dm(xA_samp[i]-dx, gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j])

        alpha_dm_xB = alpha_dm(xB_samp[i], gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j])
        alpha_dm_xB_up = alpha_dm(xB_samp[i]+dx, gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j])
        alpha_dm_xB_dw = alpha_dm(xB_samp[i]-dx, gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j])
   
        mstar_grid_here = (xA_samp[i] - xB_samp[i] - alpha_dm_xA + alpha_dm_xB)/(alpha_star_xA - alpha_star_xB)
        lmstar_grid[j, :] = np.log10(mstar_grid_here)
    
        grid_range[j, :] = mstar_grid_here > 1e9
   
        mur_xA_grid = mu_r(xA_samp[i], gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j], mstar_grid_here, reff) 
        mur_xB_grid = mu_r(xB_samp[i], gnfw_norm_grid[j, :], rs_fixed*np.ones(nlmdm5), gammadm_grid[j], mstar_grid_here, reff)
        rmur_grid[j, :] = mur_xA_grid / mur_xB_grid

        beta_grid[j, :] = xA_samp[i] - alpha_dm_xA - mstar_grid_here*alpha_star_xA
    
        mstar_xA_up_grid = (xA_up - xB_samp[i] - alpha_dm_xA_up + alpha_dm_xB)/(alpha_star_xA_up - alpha_star_xB)
        mstar_xA_dw_grid = (xA_dw - xB_samp[i] - alpha_dm_xA_dw + alpha_dm_xB)/(alpha_star_xA_dw - alpha_star_xB)

        mstar_xB_up_grid = (xA_samp[i] - xB_up - alpha_dm_xA + alpha_dm_xB_up)/(alpha_star_xA - alpha_star_xB_up)
        mstar_xB_dw_grid = (xA_samp[i] - xB_dw - alpha_dm_xA + alpha_dm_xB_dw)/(alpha_star_xA - alpha_star_xB_dw)

        beta_xA_up = xA_up - alpha_dm_xA_up - mstar_xA_up_grid * alpha_star_xA_up
        beta_xA_dw = xA_dw - alpha_dm_xA_dw - mstar_xA_dw_grid * alpha_star_xA_dw

        beta_xB_up = xA_samp[i] - alpha_dm_xA - mstar_xB_up_grid * alpha_star_xA
        beta_xB_dw = xA_samp[i] - alpha_dm_xA - mstar_xB_dw_grid * alpha_star_xA

        dlmstar_dxA = (np.log10(mstar_xA_up_grid) - np.log10(mstar_xA_dw_grid))/(2.*dx)
        dlmstar_dxB = (np.log10(mstar_xB_up_grid) - np.log10(mstar_xB_dw_grid))/(2.*dx)

        dbeta_dxA = (beta_xA_up - beta_xA_dw)/(2.*dx)
        dbeta_dxB = (beta_xB_up - beta_xB_dw)/(2.*dx)

        detJ_grid[j, :] = abs(dlmstar_dxA*dbeta_dxB - dlmstar_dxB*dbeta_dxA)

        for k in range(nlmdm5):
            if grid_range[j, k]:
                # finds maximum allowed value of beta

                xmin = max(deV.rgrid_min*reff, Rfrac_min*rs_fixed)
                xmin_grid[j, k] = xmin

                mu_r_xB = mu_r(xB_samp[i], gnfw_norm_grid[j, k], rs_fixed, gammadm_grid[j], mstar_grid_here[k], reff) 
                mu_t_xB = mu_t(xB_samp[i], gnfw_norm_grid[j, k], rs_fixed, gammadm_grid[j], mstar_grid_here[k], reff) 
                if mu_r_xB < 0.:
                    # If the model predicts a third brighter image, discards the point (this is the third image)
                    grid_range[j, k] = False
                elif abs(mu_r_xB * mu_t_xB) < minmag :
                    # If the predicted magnification of image B is too small, discards the point
                    grid_range[j, k] = False
                else: 
                    # searching for smallest values of beta for which the magnification of 2nd image is equal to minmag
                    xcimg_max_search = np.arange(xB_samp[i], -xmin, dx_search)
                    xcimg_indarr = np.arange(len(xcimg_max_search))
                    mu_r_search = mu_r(xcimg_max_search, gnfw_norm_grid[j, k], rs_fixed, gammadm_grid[j], mstar_grid_here[k], reff)
                    mu_t_search = mu_t(xcimg_max_search, gnfw_norm_grid[j, k], rs_fixed, gammadm_grid[j], mstar_grid_here[k], reff)
        
                    cimg = mu_r_search > 0. # only looks outside of the radial critical curve
                    mu_ltmin = cimg & (abs(mu_r_search * mu_t_search) < minmag)
        
                    if mu_ltmin.sum() > 0: # at some point, the magnification of the counter-image drops below minmag
                        # takes the point right before mu drops below minmag, as long as it's closer to the center than xB                    
                        xcimg_max = max(xB_samp[i], xcimg_max_search[mu_ltmin][0] - dx_search) 
                    else: # otherwise takes the radial critical curve as the limit
                        xcimg_max = xcimg_max_search[cimg][-1]
                    beta_max = xcimg_max - alpha(xcimg_max, gnfw_norm_grid[j, k], rs_fixed, gammadm_grid[j], mstar_grid_here[k], reff)

                    if beta_grid[j, k] > beta_max:
                        # excluding point because predicted magnification of image B is too small:
                        grid_range[j, k] = False
                    beta_max_grid[j, k] = beta_max

        print(i, j, grid_range[j, :].sum())

    group = grids_file.create_group('lens_%04d'%i)

    group.create_dataset('lmstar_grid', data=lmstar_grid)
    group.create_dataset('beta_grid', data=beta_grid)
    group.create_dataset('beta_max_grid', data=beta_max_grid)
    group.create_dataset('detJ_grid', data=detJ_grid)
    group.create_dataset('rmur_grid', data=rmur_grid)
    group.create_dataset('grid_range', data=grid_range)

grids_file.close()


