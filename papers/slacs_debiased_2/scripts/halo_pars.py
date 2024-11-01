import numpy as np


# halo mass distribution from Sonnenfeld et al. (2018)

h = 0.7

mu_mh_obs = 13.04
mu_mh_err = 0.04

beta_mh_obs = 1.48
beta_mh_err = 0.15

sigma_mh_obs = 0.31
sigma_mh_err = 0.04

def mu_mh_0_func(ms):
    return mu_mh_obs + beta_mh_obs * (ms - 11.3)

def c200_func(lm200, h=h):
    # mass-concentration relation at z=1 from Dutton & Maccio (2014)
    # Delta=200 halo mass definition
    return 10.**(0.905 - 0.101*(lm200 - 12. + np.log10(h)))

