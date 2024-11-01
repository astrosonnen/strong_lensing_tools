import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad
import sl_cosmology
from sl_profiles import nfw, gnfw
from read_slacs import *
from halo_pars import c200_func, mu_mh_obs, beta_mh_obs, sigma_mh_obs
from adcontr_funcs import *
from gnfw_lensingfuncs import *
import h5py


# Einstein radius, stellar mass and Jacobian of SLACS deV+nfw lenses, as a function of halo mass

dx = 0.01

nmh = 31
nms = 27

eps_min = 0.
eps_max = 1.
neps = 21
eps_grid = np.linspace(eps_min, eps_max, neps)

grid_file = h5py.File('slacs_teincs_%dx%dx%d_grids.hdf5'%(nmh, nms, neps), 'w')

grid_file.create_dataset('eps_grid', data=eps_grid)

for n in range(nslacs):

    print(slacs_names[n])
    group = grid_file.create_group(slacs_names[n])

    zd = slacs_zd[n]
    zs = slacs_zs[n]

    arcsec2kpc = slacs_rein[n]/slacs_tein[n]

    s_cr = sl_cosmology.Sigma_cr(zd, zs)
    tein = slacs_rein
    rein = tein * arcsec2kpc

    reff = slacs_reff_kpc[n]

    rein_up = rein + dx
    rein_dw = rein - dx

    rhoc = sl_cosmology.rhoc(zd)

    lmstar_min = slacs_ms_obs[n] - 0.5
    lmstar_max = slacs_ms_obs[n] + 0.8
    lmstar_grid = np.linspace(lmstar_min, lmstar_max, nms)

    mu_mh_lens = mu_mh_obs + beta_mh_obs * (slacs_ms_obs[n] - 11.3)
    mh_min = mu_mh_lens - 1.5
    mh_max = mu_mh_lens + 1.5
    mh_grid = np.linspace(mh_min, mh_max, nmh)
    c200_0_grid = c200_func(mh_grid)

    group.create_dataset('mh_grid', data=mh_grid)
    group.create_dataset('lmstar_grid', data=lmstar_grid)

    r200_grid = (10.**mh_grid*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs_0_grid = r200_grid/c200_0_grid

    cs_grid = np.zeros((neps, nmh, nms))
    rs_grid = np.zeros((neps, nmh, nms))
    gamma_grid = np.zeros((neps, nmh, nms))
    gnfw_norm_grid = np.zeros((neps, nmh, nms))
    tein_grid = np.zeros((neps, nmh, nms))
    rein_grid = np.zeros((neps, nmh, nms))

    for i in range(neps):
        print(n, i)
        for j in range(nmh):

            for k in range(nms):

                rs, gamma = find_gnfw(10.**lmstar_grid[k], reff, 10.**mh_grid[j], rs_0_grid[j], r200_grid[j], eps=eps_grid[i])
                if rs > 0. and gamma > 0.:
                    rs_grid[i, j, k] = rs
                    gamma_grid[i, j, k] = rs
    
                    gnfw_norm = 10.**mh_grid[j] / gnfw.M3d(r200_grid[j], rs, gamma)
                    gnfw_norm_grid[i, j, k] = gnfw_norm
    
                    rein = get_rein_kpc(10.**lmstar_grid[k], reff, gnfw_norm, rs, gamma, s_cr)
                    radcrit_kpc, radcaust, xA_max = get_radcaust(10.**lmstar_grid[k], reff, gnfw_norm, rs, gamma, s_cr, rein)
    
                    if radcaust > 0.:
                        rein_grid[i, j, k] = rein
                        tein_grid[i, j, k] = rein / arcsec2kpc
                        cs_grid[i, j, k] = get_crosssect(10.**lmstar_grid[k], reff, gnfw_norm, rs, gamma, s_cr, rein, radcrit_kpc, xA_max, arcsec2kpc)

    group.create_dataset('tein_grid', data=tein_grid)
    group.create_dataset('rs_grid', data=rs_grid)
    group.create_dataset('gamma_grid', data=gamma_grid)
    group.create_dataset('gnfw_norm_grid', data=gnfw_norm_grid)
    group.create_dataset('cs_grid', data=cs_grid)

grid_file.close()

