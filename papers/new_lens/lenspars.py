import numpy as np
from sl_cosmology import Sigma_cr, arcsec2kpc as arcsec2kpc_func


zd = 1.055 # lens redshift
zs = 2.792 # source redshift

# Using the default cosmology: H0=70, omegaM=0.3.
s_cr = Sigma_cr(zd, zs)
arcsec2kpc = arcsec2kpc_func(zd)

reff_kpc = 2.3 # half-light radius in kpc
rein_kpc = 1.8 # Einstein radius in kpc
rein_err = 0.3 # 1-sigma uncertainty on the Einstein radius

# Einstein radius in arcsec.
# Computing from rein_kpc, but the other way around would be better
tein_obs = rein_kpc / arcsec2kpc
tein_err = rein_err / arcsec2kpc

lmsps_obs = np.log10(3.1e10) # log observed stellar mass
lmsps_err = 0.5*(np.log10(3.7e10) - np.log10(2.7e10)) # uncertainty on log(Mstar)

