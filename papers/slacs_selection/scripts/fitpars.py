import numpy as np

mpiv = 11.
z_piv = 0.2

mpiv_slacs = 11.3
rpiv_slacs = 0.8

gamma_min = 1.2
gamma_max = 2.8

ms_min = 10.
ms_max = 12.5

# upper bound to source redshift population
zs_max = 2.

# Fundamental plane prior
slacs_fpscat = 0.039
fiducial_fpscat = 0.047
err_fpscat = 0.008

# quadratic mass-size relation from Hyde and Bernardi (2009)
def hb09quad_mu_r_func(ms):
    return 7.55 - 1.84*ms + 0.110*ms**2

# dispersion in log-Reff at fixed stellar mass
s19_sigma_r = 0.112

# mass-veldisp relation of the parent sample
def mu_v_parent(ms):
    return 2.2577 + 0.3034 * (ms - 11.) - 0.0761 * (ms - 11.)**2

mu_v_prior = mu_v_parent(11.3)
err_mu_v = 0.03

beta_v_prior = (mu_v_parent(11.31) - mu_v_parent(11.29))/0.02
err_beta_v = 0.03

sigma_v_prior = 0.0773
err_sigma_v = 0.01

