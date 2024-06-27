import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw
from sl_cosmology import Mpc, Dang, G, c, M_Sun, arcsec2kpc as arcsec2kpc_func
import h5py
from parent_sample_pars import *
from fitpars import *


# grid of values of sigma_ap^2 for each lens, as a function of:
# - power-law slope
# for a power-law density profile with unit M5 (projected mass within 5kpc).

f = open('../SLACS_table.cat', 'r')
names = np.loadtxt(f, usecols=(0, ), dtype=str)
f.close()

nslacs = len(names)

f = open('../SLACS_table.cat', 'r')
slacs_zd, slacs_zs, slacs_reff, slacs_rein = np.loadtxt(f, usecols=(3, 4, 6, 8), unpack=True)
f.close()

nr = 1001

kpc = Mpc/1000.

ngamma = 17
gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

sigma_grid_file = h5py.File('slacs_jeans_grids.hdf5', 'w')
sigma_grid_file.create_dataset('gamma_grid', data=gamma_grid)

for n in range(nslacs):

    print('%d %s'%(n, names[n]))

    group = sigma_grid_file.create_group(names[n])

    zd = slacs_zd[n]
    reff_kpc = slacs_reff[n]
    arcsec2kpc = arcsec2kpc_func(zd)

    fibre_kpc = fibre_arcsec * arcsec2kpc
    seeing_kpc = fibre_arcsec * arcsec2kpc
    
    r3d_grid = np.logspace(np.log10(reff_kpc) - 3., np.log10(reff_kpc) + 3., nr)

    s2_grid = np.zeros(ngamma)

    for k in range(ngamma):
        norm = 1./powerlaw.M2d(5., gamma_grid[k])
        m3d_grid = norm * powerlaw.M3d(r3d_grid, gamma_grid[k])
    
        s2_grid[k] = sigma_model.sigma2((r3d_grid, m3d_grid), fibre_kpc, reff_kpc, tracer_profiles.deVaucouleurs, seeing=seeing_kpc)

    s2_grid *= G*M_Sun/kpc/1e10
    group.create_dataset('s2_grid', data=s2_grid)

