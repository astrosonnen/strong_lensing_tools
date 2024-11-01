import numpy as np


# mass-size relation from Sonnenfeld et al. (2019)
def s19_mu_r_func(ms):
    return 0.774 + 0.977 * (ms - 11.4)

s19_sigma_r = 0.112
sigma_r = s19_sigma_r

# mass-size relation from SLACS-X (Auger et al. 2010)
def slacsx_mu_r_func(ms):
    return 0.52 + 0.89 * (ms - 11.)

# quadratic mass-size relation from Hyde and Bernardi (2009)
def hb09quad_mu_r_func(ms):
    return 7.55 - 1.84*ms + 0.110*ms**2

# quadratic mass-size relation from fit to Meert et al. (2014) data

def mu_r_meert14(ms):
    ms_upenn = ms + 0.084
    return 0.684 + 0.6059 * (ms_upenn - 11.) + 0.0980 * (ms_upenn - 11.)**2

meert14_sigma_r = 0.170

