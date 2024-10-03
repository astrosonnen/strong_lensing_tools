import numpy as np


# stellar mass-size relation from van der Wel et al. (2014)

# taking the average over the z=0.75 and z=1.25 bins.

masssize_mu = 0.5*(0.42 + 0.22)
masssize_beta = 0.5*(0.71 + 0.76) # this is called alpha in vvW14.
masssize_sigma = 0.12 # scatter in logRe

masssize_mpiv = np.log10(5e10) # pivot of mass-size relation

