import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw
from sl_cosmology import Mpc, Dang, G, c, M_Sun, arcsec2kpc as arcsec2kpc_func
import sl_cosmology
import h5py
from parent_sample_pars import *
from fitpars import gamma_min, gamma_max
from astropy.io import fits as pyfits


# grid of values of sigma_ap^2, as a function of:
# - redshift
# - half-light radius (in physical units)
# - power-law slope
# for a power-law density profile with unit M5 (projected mass within 5kpc).

nz = 21
z_grid = np.linspace(zmin, zmax, nz)

nreff = 21
lreff_min = -0.3
lreff_max = 2.3

lreff_grid = np.linspace(lreff_min, lreff_max, nreff)

ngamma = 17

gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

s2_grid = np.zeros((nz, nreff, ngamma))

nr = 1001

kpc = Mpc/1000.

for i in range(nz):
    arcsec2kpc = arcsec2kpc_func(z_grid[i])
    fibre_kpc = 1.5 * arcsec2kpc
    seeing_kpc = 1.5 * arcsec2kpc

    for j in range(nreff):
        print(i, j)
        reff_kpc = 10.**lreff_grid[j]
        r3d_grid = np.logspace(np.log10(reff_kpc) - 3., np.log10(reff_kpc) + 3., nr)
        for k in range(ngamma):
            norm = 1./powerlaw.M2d(5., gamma_grid[k])
            m3d_grid = norm * powerlaw.M3d(r3d_grid, gamma_grid[k])

            s2_grid[i, j, k] = sigma_model.sigma2((r3d_grid, m3d_grid), fibre_kpc, reff_kpc, tracer_profiles.deVaucouleurs, seeing=seeing_kpc)

s2_grid *= G*M_Sun/kpc/1e10

sigma_grid_file = h5py.File('sigma2_grid.hdf5', 'w')
sigma_grid_file.create_dataset('s2_grid', data=s2_grid)

sigma_grid_file.create_dataset('z_grid', data=z_grid)
sigma_grid_file.create_dataset('lreff_grid', data=lreff_grid)
sigma_grid_file.create_dataset('gamma_grid', data=gamma_grid)

