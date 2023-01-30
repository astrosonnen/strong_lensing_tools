import numpy as np
from sl_profiles import nfw, gnfw, sersic, eaglecontr, deVaucouleurs as deV
import sl_cosmology
from scipy.interpolate import splrep, splev, splint, interp1d
from scipy.optimize import minimize, leastsq
from poppars import *
import h5py


# each lens consists of a de Vaucouleurs stellar component and a gNFW halo.
# The parameters of the gNFW profile are defined by first using the Cautun 2020
# contraction model, then fitting the resulting profile with a gNFW model.
# It is time-consuming, therefore it's more convenient to do this on a grid
# of values of stellar mass, halo mass and half-light radius.

lmstar_min = 10.5
lmstar_max = 12.5

dlreff_min = -0.7
dlreff_max = 0.7

dlm200_min = -1.
dlm200_max = 1.

nmstar = 21
nreff = 15
nm200 = 21
nz = 7

lmstar_grid = np.linspace(lmstar_min, lmstar_max, nmstar)
dlreff_grid = np.linspace(dlreff_min, dlreff_max, nreff)
dlm200_grid = np.linspace(dlm200_min, dlm200_max, nm200)
z_grid = np.linspace(zmin, zmax, nz)

gammadm_grid = np.zeros((nmstar, nreff, nm200, nz))
rs_grid = np.zeros((nmstar, nreff, nm200, nz))

nr3d = 1001
r3d_scalefree = np.logspace(-3., 3., nr3d) # radial grid, from 1/100 to 100 times Reff
deV_rho_scalefree = deV.rho(r3d_scalefree, 1.)
deV_m3d_unitmass = deV.fast_M3d(r3d_scalefree)

nR2d = 101
R2d_scalefree = np.logspace(-3., 2., nR2d)

R2d_fit_grid = np.logspace(0., np.log10(30.), nR2d) # radial grid over which to fit the gNFW model

def get_halo_splines(m200, mstar, reff, rhoc):
    r200 = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs = r200/c200_0
    r3d_grid = r3d_scalefree * reff
    R2d_grid = R2d_scalefree * reff

    xmin = 1.01*R2d_grid[0]
    xmax = 0.99*R2d_grid[-1]

    nfw_norm = m200/nfw.M3d(r200, rs)
    nfw_rho = nfw_norm * nfw.rho(r3d_grid, rs)
    nfw_m3d = nfw_norm * nfw.M3d(r3d_grid, rs)

    deV_rho = mstar * deV_rho_scalefree / reff**3
    deV_m3d = mstar * deV_m3d_unitmass

    density_ratio = eaglecontr.contract_density(nfw_rho, deV_rho, nfw_m3d, deV_m3d)/ nfw_rho
    density_ratio[density_ratio < 1.] = 1.
    density_increase = interp1d(r3d_grid, density_ratio, bounds_error=False, \
                                                    fill_value=(density_ratio[0], density_ratio[-1]) )

    contr_Sigma = eaglecontr.projected_density_NFW_contracted(R2d_grid, m200, c200_0, rs, density_increase)
    contr_Sigma_spline = splrep(R2d_grid, contr_Sigma)
    contr_SigmaR_spline = splrep(R2d_grid, contr_Sigma * R2d_grid)

    return contr_Sigma_spline, contr_SigmaR_spline, R2d_grid

def gnfw_fit(halo_Sigma_spline, m200, rhoc):

    r200_here = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.

    halo_Sigma_fit = splev(R2d_fit_grid, halo_Sigma_spline)

    def fitfunc(p):
        gammadm, rs = p
        gnfw_norm = m200 / gnfw.M3d(r200_here, rs, gammadm)
        return gnfw_norm * gnfw.fast_Sigma(R2d_fit_grid, rs, gammadm)

    def errfunc(p):
        return (fitfunc(p) - halo_Sigma_fit)/halo_Sigma_fit

    pfit = leastsq(errfunc, (1.4, r200_here/c200_0))[0]
    gammadm_fit, rs_fit = pfit

    return gammadm_fit, rs_fit

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

            for l in range(nz):
                rhoc = sl_cosmology.rhoc(z_grid[l])
                halo_Sigma_spline, halo_SigmaR_spline, R2d_grid = get_halo_splines(m200, mstar, reff, rhoc)

                gammadm_fit, rs_fit = gnfw_fit(halo_Sigma_spline, m200, rhoc)

                gammadm_grid[i, j, k, l] = gammadm_fit
                rs_grid[i, j, k, l] = rs_fit

grid_file = h5py.File('gnfwpar_grid.hdf5', 'w')

grid_file.create_dataset('lmstar_grid', data=lmstar_grid)
grid_file.create_dataset('dlreff_grid', data=dlreff_grid)
grid_file.create_dataset('dlm200_grid', data=dlm200_grid)
grid_file.create_dataset('z_grid', data=z_grid)

grid_file.create_dataset('gammadm_grid', data=gammadm_grid)
grid_file.create_dataset('rs_grid', data=rs_grid)

