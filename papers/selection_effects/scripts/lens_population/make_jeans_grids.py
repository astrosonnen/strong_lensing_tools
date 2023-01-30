import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import nfw
from sl_cosmology import Mpc, Dang, G, c, M_Sun
import sl_cosmology
from sl_profiles import deVaucouleurs as deV, gnfw
import h5py
from poppars import *


# calculates grids of sigma^2 for the stellar component
# and the dark matter component.
# The stellar component has only 1 non-linear dimension: the size.
# Therefore, sigma^2 is computed on a 1D with unit mass.
# The dark matter halo has three degrees of freedom.
# Therefore it uses the same grid over which the contracted profile has
# been computed.

redshift = 0.2
dd = Dang(redshift)
rhoc = sl_cosmology.rhoc(redshift)
arcsec2kpc = np.deg2rad(1./3600.) * dd * 1000.
kpc = Mpc / 1000.

aperture_kpc = 1.5 * arcsec2kpc
seeing_kpc = aperture_kpc

nreff = 16
lreff_grid = np.linspace(0.5, 2., nreff)

nr3d = 1001
r3d_scalefree = np.logspace(-3., 3., nr3d) # radial grid, from 1/100 to 100 times Reff
deV_rho_scalefree = deV.rho(r3d_scalefree, 1.)
deV_m3d_unitmass = deV.fast_M3d(r3d_scalefree)

s2_deV_grid = np.zeros(nreff)

for i in range(nreff):
    reff = 10.**lreff_grid[i]
    s2_deV_grid[i] = sigma_model.sigma2((r3d_scalefree*reff, deV_m3d_unitmass), aperture_kpc, reff, tracer_profiles.deVaucouleurs, seeing=seeing_kpc)

s2_deV_grid *= G*M_Sun/kpc/1e10

deV_file = h5py.File('s2_deV_grid.hdf5', 'w')
deV_file.create_dataset('lreff_grid', data=lreff_grid)
deV_file.create_dataset('s2_grid', data=s2_deV_grid)

# loads the gnfw parameters grid

gnfwpar_file = h5py.File('gnfwpar_grid.hdf5', 'r')

lmstar_grid = gnfwpar_file['lmstar_grid'][()]
nmstar = len(lmstar_grid)

dlreff_grid = gnfwpar_file['dlreff_grid'][()]
nreff = len(dlreff_grid)

dlm200_grid = gnfwpar_file['dlm200_grid'][()]
nm200 = len(dlm200_grid)

rs_grid = gnfwpar_file['rs_grid'][()]
gammadm_grid = gnfwpar_file['gammadm_grid'][()]

s2_grid = np.zeros((nmstar, nreff, nm200))

for i in range(nmstar):
    mstar = 10.**lmstar_grid[i]
    lreff_model = mu_R + beta_R * (lmstar_grid[i] - mu_sps - lmobs_piv)
    lm200_model = mu_h + beta_h * (lmstar_grid[i] - lmstar_piv)

    for j in range(nreff):
        print(i, j)
        lreff_here = lreff_model + dlreff_grid[j]
        reff = 10.**lreff_here

        for k in range(nm200):
            m200 = 10.**(lm200_model + dlm200_grid[k])
            r200 = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.

            rs_here = rs_grid[i, j, k]
            gammadm_here = gammadm_grid[i, j, k]

            gnfw_norm = m200 / gnfw.M3d(r200, rs_here, gammadm_here)

            m3d_grid = mstar * deV_m3d_unitmass + gnfw_norm * gnfw.fast_M3d(reff * r3d_scalefree, rs_here, gammadm_here)

            s2_grid[i, j, k] = sigma_model.sigma2((r3d_scalefree*reff, m3d_grid), aperture_kpc, reff, tracer_profiles.deVaucouleurs, seeing=seeing_kpc)

s2_grid *= G*M_Sun/kpc/1e10
sigma_grid = s2_grid**0.5
 
sigma_grid_file = h5py.File('deVgnfw_sigma_grid.hdf5', 'w')
sigma_grid_file.create_dataset('sigma_grid', data=sigma_grid)

sigma_grid_file.create_dataset('lmstar_grid', data=lmstar_grid)
sigma_grid_file.create_dataset('dlreff_grid', data=dlreff_grid)
sigma_grid_file.create_dataset('dlm200_grid', data=dlm200_grid)
