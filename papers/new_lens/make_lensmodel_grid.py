import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import leastsq
import sl_cosmology
from sl_profiles import deVaucouleurs as deV, nfw
import h5py
import ndinterp
from lenspars import reff_kpc, s_cr, arcsec2kpc
from lensingfuncs import *
from halo_pars import rho_c, deltaVir, cvir_func
import sys


# this script computes the Einstein radius and lensing cross-section
# of the lens, on a grid of stellar and halo mass

lmvir_min = 11.5
lmvir_max = 13.5
nmvir = 101
lmvir_grid = np.linspace(lmvir_min, lmvir_max, nmvir)

rvir_grid = (10.**lmvir_grid*3./deltaVir/(4.*np.pi)/rho_c)**(1./3.)
cvir_grid = cvir_func(lmvir_grid)
rs_grid = rvir_grid/cvir_grid

lmstar_min = 10.
lmstar_max = 11.
nmstar = 101
lmstar_grid = np.linspace(lmstar_min, lmstar_max, nmstar)

output_file = h5py.File('lensmodel_grid.hdf5', 'w')

output_file.create_dataset('lmvir_grid', data=lmvir_grid)
output_file.create_dataset('lmstar_grid', data=lmstar_grid)

cs_grid = np.zeros((nmvir, nmstar))
tein_grid = np.zeros((nmvir, nmstar))

for i in range(nmvir):
    print('%d/%d'%(i, nmvir-1))
    nfw_norm = 10.**lmvir_grid[i] / nfw.M3d(rvir_grid[i], rs_grid[i])
    for k in range(nmstar):
        mstar = 10.**lmstar_grid[k]

        # computes Einstein radius
        rein_here = get_rein_kpc(mstar, reff_kpc, nfw_norm, rs_grid[i], s_cr)
        tein = rein_here / arcsec2kpc
        tein_grid[i, k] = tein

        # computes radial caustic and critical curve (if any)
        xrad_kpc, radcaust_kpc = get_radcaust(mstar, reff_kpc, nfw_norm, rs_grid[i], s_cr, rein_here)

        cs_grid[i, k] = get_crosssect(mstar, reff_kpc, nfw_norm, rs_grid[i], s_cr, rein_here, xrad_kpc, arcsec2kpc)

        # multiplies the cross-section by the lens finding probability
        cs_grid[i, k] *= pfind(tein)

output_file.create_dataset('tein_grid', data=tein_grid)
output_file.create_dataset('cs_grid', data=cs_grid)

