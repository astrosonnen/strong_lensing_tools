import numpy as np
from scipy.interpolate import splrep, splev, splint
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import mass_function
from lenspars import zd


h = 0.7
my_cosmo = {'flat': True, 'H0': h*100., 'Om0': 0.3, 'Ob0': 0.043, 'sigma8': 0.8, 'ns': 0.97}
cosmo = cosmology.setCosmology('my_cosmo', **my_cosmo)

rho_c = cosmo.rho_c(zd)*h**2 # critical density of the universe at the lens redshift, in M_Sun/kpc^3.

deltaVir = mass_so.deltaVir(zd)

# SHMR from Shuntov et al. (2022)
# 0.8 < z < 1.1 bin

logm1 = 12.730
logms0 = 11.013
beta = 0.454
delta = 1.109
gamma = 1.925
alpha = 1.065
bsat = 5.416
betasat = 0.612
bcut = 8.845
betacut = 1.098
sigmalogms = 0.250 # mostly degenerate with logms0

# In Table F.1, sigmalogm is reported to be 0.211-0.203+0.012
# but in the figure there is no trace of that long tail. Weird.

# procedure for generating a sample of halos and galaxies:
# 1- draw halos from the halo mass function of Despali et al. (2016)
# 2- For each halo mass, draw logMstar from a Gaussian distribution
# centred at log(f_SHMR(Mh)) with scatter sigmalogM.
# Halo mass definition:
# "To compute the halo mass function, we used the COLOSSUS code (Diemer 2018)
# and we used the virial overdensity (Bryan & Norman 1998) halo mass 
# definition"

lmvir_min = 11.5
lmvir_max = 14.5
nmvir = 301

lmvir_grid = np.linspace(lmvir_min, lmvir_max, nmvir)

lmsps_gridmin = 10.
lmsps_gridmax = 11.8
nmsps = 181

lmsps_grid = np.linspace(lmsps_gridmin, lmsps_gridmax, nmsps)

def invshmr(lmsps):
    return logm1 + beta * (lmsps - logms0) + 10.**(delta * (lmsps - logms0)) / (1. + 10.**(-gamma * (lmsps - logms0))) - 0.5

invshmr_grid = invshmr(lmsps_grid)
shmr_spline = splrep(invshmr_grid, lmsps_grid)

def shmr(lmvir):
    return splev(lmvir, shmr_spline)

def dndmh(lmvir):
    mfunc = mass_function.massFunction(10.**lmvir, 2., mdef='vir', model='despali16', q_out='dndlnM')
    return mfunc

dndmh_grid = dndmh(lmvir_grid)
dndmh_spline = splrep(lmvir_grid, dndmh_grid)

cumhmf_grid = np.zeros(nmvir)
for n in range(nmvir):
    cumhmf_grid[n] = splint(lmvir_grid[0], lmvir_grid[n], dndmh_spline)

cumhmf_grid /= cumhmf_grid[-1]
invcumhmf_spline = splrep(cumhmf_grid, lmvir_grid)

def c200_func(lm200, h=h):
    # mass-concentration relation at z=1 from Dutton & Maccio (2014)
    # Delta=200 halo mass definition
    return 10.**(0.728 - 0.073*(lm200 - 12. + np.log10(h)))

def cvir_func(lmvir, h=h):
    # mass-concentration relation at z=1 from Dutton & Maccio (2014)
    # Delta=156.9
    return 10.**(0.775 - 0.073*(lmvir - 12. + np.log10(h)))

