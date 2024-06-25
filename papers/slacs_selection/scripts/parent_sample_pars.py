import numpy as np


# redshift limits
zmin = 0.05
zmax = 0.40

# maximum Halpha emission line equivalent width
ha_max = 0.

# stellar mass cuts (outliers are likely catastrophic failures)
lmchab_min = 9.
lmchab_max = 12.5

# minimum (systematic) uncertainty in stellar mass measurements
lmchab_err_min = 0.1

# average offset between DR18 and SLACS stellar masses
lmstar_shift = -0.225

# half-light radius cuts
reff_arcsec_min = 0.1
reff_arcsec_max = 29.

# average offset between DR18 and SLACS half-light radius
lreff_shift = -np.log10(1.087)

# velocity dispersion cuts
veldisp_min = 0.
veldisp_max = 500.
veldisp_syst_relerr = 0. 

# spectroscopic data parameters
fibre_arcsec = 1.5
seeing_arcsec = 1.5

# u-r colour cut
umr_min = 2.

