import numpy as np
import h5py
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

slacs_median_sigma_relerr = np.median(slacs_sigma_err/slacs_sigma_obs)

