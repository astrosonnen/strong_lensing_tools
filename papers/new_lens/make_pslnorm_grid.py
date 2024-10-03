import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import leastsq
import sl_cosmology
from sl_profiles import deVaucouleurs as deV, nfw
import h5py
import ndinterp
from lenspars import s_cr, arcsec2kpc
from lensingfuncs import *
from halo_pars import invcumhmf_spline, shmr, sigmalogms, deltaVir, rho_c, cvir_func
from masssize_pars import *
import sys


# this script computes the normalisation of the strong lens population prior
# on a grid of values of the IMF mismatch parameters

laimf_min = -0.1
laimf_max = 0.3
naimf = 41
laimf_grid = np.linspace(laimf_min, laimf_max, naimf)

# The normalisation of P_SL is essentially the average lensing cross-section
# of the population of galaxies. To compute it, we generate a large
# sample of galaxies, compute their lensing cross-section, and take the mean

ngal = 10000 # number of galaxies

# draws halo masses from the halo mass function
# uses the inverse cumulative distribution computed in halo_pars
lmvir_samp = splev(np.random.rand(ngal), invcumhmf_spline)

rvir_samp = (10.**lmvir_samp*3./deltaVir/(4.*np.pi)/rho_c)**(1./3.)
cvir_samp = cvir_func(lmvir_samp)
rs_samp = rvir_samp/cvir_samp
nfw_norm_samp = 10.**lmvir_samp / nfw.M3d(rvir_samp, rs_samp)

# draws sps stellar masses: logNormal around the SHMR
lmsps_samp = np.random.normal(shmr(lmvir_samp), sigmalogms, ngal)

# draws half-light radii
lreff_samp = np.random.normal(masssize_mu + masssize_beta * (lmsps_samp - masssize_mpiv), masssize_sigma, ngal)

output_file = h5py.File('pslnorm_grid.hdf5', 'w')

output_file.create_dataset('laimf_grid', data=laimf_grid)

norm_grid = np.zeros(naimf)

print('Computing P_SL on a grid of log(aimf)')
for n in range(naimf):

    print('%d/%d'%(n+1, naimf))

    cs_samp = np.zeros(ngal)
    tein_samp = np.zeros(ngal)
    
    for i in range(ngal):
        reff_kpc = 10.**lreff_samp[i]
        mstar = 10.**(lmsps_samp[i] + laimf_grid[n])

        # computes Einstein radius
        rein_here = get_rein_kpc(mstar, reff_kpc, nfw_norm_samp[i], rs_samp[i], s_cr)
        tein = rein_here / arcsec2kpc
    
        # computes radial caustic and critical curve (if any)
        xrad_kpc, radcaust_kpc = get_radcaust(mstar, reff_kpc, nfw_norm_samp[i], rs_samp[i], s_cr, rein_here)
    
        # computes the cross-section
        cs_samp[i] = get_crosssect(mstar, reff_kpc, nfw_norm_samp[i], rs_samp[i], s_cr, rein_here, xrad_kpc, arcsec2kpc)
        cs_samp[i] *= pfind(tein)

    norm_grid[n] = cs_samp.mean()

output_file.create_dataset('norm_grid', data=norm_grid)

