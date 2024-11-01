import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad
import sl_cosmology
from sl_profiles import nfw, gnfw
from read_slacs import *
from halo_pars import *
from adcontr_funcs import find_gnfw
from pop_funcs import get_msonly_pop
from gnfw_lensingfuncs import *
import h5py


npop = 10000
seedno = 0

pop = get_msonly_pop(npop, seedno)
mu_mh_0 = mu_mh_0_func(pop['ms'][()])
mh_scat = pop['mh_scat'][()]
mh_0_pop = mu_mh_0 + mh_scat * sigma_mh_obs

dmh_min = -0.5
dmh_max = 0.5
nmh = 11
dmh_grid = np.linspace(dmh_min, dmh_max, nmh)

lasps_min = 0.
lasps_max = 0.3
nasps = 11
lasps_grid = np.linspace(lasps_min, lasps_max, nasps)

dx = 0.01

grid_file = h5py.File('npop%1.0e_teincs_grids.hdf5'%npop, 'w')
grid_file.attrs['npop'] = npop
grid_file.attrs['seedno'] = seedno

for par in pop:
    grid_file.create_dataset(par, data=pop[par])

eps_min = 0.
eps_max = 1.
neps = 11
eps_grid = np.linspace(eps_min, eps_max, neps)

grid_file.create_dataset('dmh_grid', data=dmh_grid)
grid_file.create_dataset('lasps_grid', data=lasps_grid)
grid_file.create_dataset('eps_grid', data=eps_grid)

for n in range(npop):

    group = grid_file.create_group('%05d'%n)

    zd = pop['zd'][n]
    zs = pop['zs'][n]
    arcsec2kpc = pop['arcsec2kpc'][n]
    rhoc = pop['rhoc'][n]
    s_cr = pop['s_cr'][n]
    reff = 10.**pop['re'][n]

    mh_grid = mh_0_pop[n] + dmh_grid
    r200_grid = (10.**mh_grid*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs_0_grid = r200_grid/c200_func(mh_grid)

    lmstar_grid = pop['ms'][n] + lasps_grid

    cs_grid = np.zeros((neps, nmh, nasps))
    rs_grid = np.zeros((neps, nmh, nasps))
    gamma_grid = np.zeros((neps, nmh, nasps))
    tein_grid = np.zeros((neps, nmh, nasps))

    group.create_dataset('mh_grid', data=mh_grid)

    if pop['bkg'][n]:
        for i in range(neps):
            print(n, i)
            for j in range(nmh):
    
                for k in range(nasps):
    
                    rs, gamma = find_gnfw(10.**lmstar_grid[k], reff, 10.**mh_grid[j], rs_0_grid[j], r200_grid[j], eps=eps_grid[i])
                    if rs > 0. and gamma > 0.:
                        rs_grid[i, j, k] = rs
                        gamma_grid[i, j, k] = gamma
        
                        gnfw_norm = 10.**mh_grid[j] / gnfw.M3d(r200_grid[j], rs, gamma)

                        rein = get_rein_kpc(10.**lmstar_grid[k], reff, gnfw_norm, rs, gamma, s_cr)
                        radcrit_kpc, radcaust, xA_max = get_radcaust(10.**lmstar_grid[k], reff, gnfw_norm, rs, gamma, s_cr, rein)
        
                        if radcaust > 0.:
                            tein_grid[i, j, k] = rein / arcsec2kpc
                            cs_grid[i, j, k] = get_crosssect(10.**lmstar_grid[k], reff, gnfw_norm, rs, gamma, s_cr, rein, radcrit_kpc, xA_max, arcsec2kpc)

    group.create_dataset('tein_grid', data=tein_grid)
    group.create_dataset('rs_grid', data=rs_grid)
    group.create_dataset('gamma_grid', data=gamma_grid)
    group.create_dataset('cs_grid', data=cs_grid)

grid_file.close()

