import numpy as np
from astropy.io import fits as pyfits
import h5py
import ndinterp
import emcee
from scipy.interpolate import splrep, splev, splint
from fitpars import *
from parent_sample_pars import *
import sl_cosmology
from lensingtools import lens_catalogs
from scipy.stats import truncnorm
from scipy.special import erf
import sys


f = open('../SLACS_table.cat', 'r')
slacs_names = np.loadtxt(f, usecols=(0, ), dtype=str)
f.close()

nslacs = len(slacs_names)

f = open('../SLACS_table.cat', 'r')
slacs_zd, slacs_zs, slacs_reff_arcsec, slacs_reff_kpc, slacs_tein, slacs_rein, slacs_ms_obs, slacs_ms_err, slacs_sigma_obs, slacs_sigma_err = np.loadtxt(f, usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12), unpack=True)
f.close()

# reads the lensing and dynamics grids
lensgrid_file = h5py.File('slacs_lensing_grids.hdf5', 'r')
gamma_lens_grid = lensgrid_file['gamma_grid'][()]
ngamma = len(gamma_lens_grid)

jeans_file = h5py.File('slacs_jeans_grids.hdf5', 'r')
gamma_jeans_grid = jeans_file['gamma_grid'][()]

output_file = h5py.File('slacs_flatprior_grids.hdf5', 'w')
output_file.create_dataset('gamma_grid', data=gamma_lens_grid)

for n in range(nslacs):

    group = output_file.create_group(slacs_names[n])

    m5_grid = lensgrid_file[slacs_names[n]]['m5_grid'][()]
    m5_spline = splrep(gamma_lens_grid, m5_grid)

    dm5drein_grid = lensgrid_file[slacs_names[n]]['dm5drein_grid'][()]

    logp_grid = 0. * gamma_lens_grid

    s2_grid = jeans_file[slacs_names[n]]['s2_grid'][()]
    s2_spline = splrep(gamma_jeans_grid, s2_grid)

    s2_model = 10.**m5_grid * splev(gamma_lens_grid, s2_spline)
    sigma_model = s2_model**0.5

    sigma_obs = slacs_sigma_obs[n]
    sigma_err = slacs_sigma_err[n]

    logp_grid = -0.5*(sigma_model - sigma_obs)**2/sigma_err**2 - np.log(sigma_err) - np.log(dm5drein_grid)

    group.create_dataset('m5_grid', data=m5_grid)
    group.create_dataset('logp_grid', data=logp_grid)
    group.create_dataset('sigma_grid', data=sigma_model)

