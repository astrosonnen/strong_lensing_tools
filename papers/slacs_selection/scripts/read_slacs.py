import numpy as np
import h5py
from scipy.interpolate import splrep, splev, splint
from fitpars import *
from parent_sample_pars import *
import sl_cosmology
from sl_cosmology import Dang, Mpc, M_Sun, c, G, kpc


f = open('../SLACS_table.cat', 'r')
slacs_names = np.loadtxt(f, usecols=(0, ), dtype=str)
f.close()

nslacs = len(slacs_names)

f = open('../SLACS_table.cat', 'r')
slacs_zd, slacs_zs, slacs_reff_arcsec, slacs_reff_kpc, slacs_tein, slacs_rein, slacs_ms_obs, slacs_ms_err, slacs_sigma_obs, slacs_sigma_err = np.loadtxt(f, usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12), unpack=True)
f.close()

slacs_r = np.log10(slacs_reff_kpc)
slacs_drat = np.zeros(nslacs)
slacs_tein_est = np.zeros(nslacs)

# reads the individual lens inference on the power-law parameters, assuming flat priors
slacs_gamma_med = np.zeros(nslacs)
slacs_gamma_uperr = np.zeros(nslacs)
slacs_gamma_dwerr = np.zeros(nslacs)

slacs_m5_med = np.zeros(nslacs)
slacs_m5_uperr = np.zeros(nslacs)
slacs_m5_dwerr = np.zeros(nslacs)

grids_file = h5py.File('slacs_flatprior_grids.hdf5', 'r')
gamma_grid = grids_file['gamma_grid'][()]

for n in range(nslacs):

    drat = Dang(slacs_zd[n], slacs_zs[n])/Dang(slacs_zs[n])
    slacs_drat[n] = drat

    slacs_tein_est[n] = np.rad2deg(4.*np.pi * (slacs_sigma_obs[n]/3e5)**2 * drat) * 3600.

    group = grids_file[slacs_names[n]]
    logp_grid = group['logp_grid'][()]
    logp_grid -= logp_grid.max()
    p_grid = np.exp(logp_grid)
    p_grid /= p_grid.sum()
    p_spline = splrep(gamma_grid, p_grid)
    p_cumsum = p_grid.cumsum()

    range_here = (p_cumsum >= 0.16) & (p_cumsum <= 0.84)

    med_ind = abs(p_cumsum - 0.5).argmin()

    slacs_m5_med[n] = group['m5_grid'][med_ind]
    slacs_gamma_med[n] = gamma_grid[med_ind]

    m5_dw = group['m5_grid'][range_here][0]
    m5_up = group['m5_grid'][range_here][-1]
    m5_med = group['m5_grid'][med_ind]

    if m5_up > m5_dw:
        slacs_m5_uperr[n] = m5_up - m5_med
        slacs_m5_dwerr[n] = m5_med - m5_dw
    else:
        slacs_m5_uperr[n] = m5_dw - m5_med
        slacs_m5_dwerr[n] = m5_med - m5_up

    slacs_gamma_uperr[n] = gamma_grid[range_here][-1] - gamma_grid[med_ind]
    slacs_gamma_dwerr[n] = gamma_grid[med_ind] - gamma_grid[range_here][0]

slacs_median_sigma_relerr = np.median(slacs_sigma_err/slacs_sigma_obs)

grids_file.close()

