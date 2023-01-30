import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import minimize
from poppars import *
import pylab
import h5py
import ndinterp


np.random.seed(0)

sigma_h = 0.2
sigma_sps = 0.08

def lmobsfunc(lmobs):
    return 10.**((lmobs - lmstar_muz) * (alpha_muz + 1)) * np.exp(-10.**(lmobs - lmstar_muz))

nmobs = 101
lmobs_grid = np.linspace(lmobs_min, lmobs_max, nmobs)

lmobsfunc_spline = splrep(lmobs_grid, lmobsfunc(lmobs_grid))
norm = splint(lmobs_min, lmobs_max, lmobsfunc_spline)

cumfunc_grid = 0.*lmobs_grid
for i in range(nmobs):
    cumfunc_grid[i] = splint(lmobs_min, lmobs_grid[i], lmobsfunc_spline) / norm

nsamp = 1000

invcum_spline = splrep(cumfunc_grid, lmobs_grid)

lmobs_samp = splev(np.random.rand(nsamp), invcum_spline)

lreff_samp = mu_R + beta_R * (lmobs_samp - lmobs_piv) + np.random.normal(0., sigma_R, nsamp)

lasps_samp = np.random.normal(mu_sps, sigma_sps, nsamp)

lmstar_samp = lmobs_samp + lasps_samp

dlm200_samp = np.random.normal(0., sigma_h, nsamp)
lm200_samp = mu_h + beta_h * (lmstar_samp - lmstar_piv) + dlm200_samp

mu_Rmstar_samp = mu_R + beta_R * (lmstar_samp - mu_sps - lmobs_piv)
dlreff_samp = lreff_samp - mu_Rmstar_samp

point = np.array([lmstar_samp, dlreff_samp, dlm200_samp]).T

sigma_grid_file = h5py.File('deVgnfw_sigma_grid.hdf5', 'r')
sigma_grid = sigma_grid_file['sigma_grid'][()]

lmstar_grid = sigma_grid_file['lmstar_grid'][()]
nmstar = len(lmstar_grid)

dlreff_grid = sigma_grid_file['dlreff_grid'][()]
nreff = len(dlreff_grid)

dlm200_grid = sigma_grid_file['dlm200_grid'][()]
nm200 = len(dlm200_grid)

axes = {0: splrep(lmstar_grid, np.arange(nmstar)), 1: splrep(dlreff_grid, np.arange(nreff)), 2: splrep(dlm200_grid, np.arange(nm200))}

sigma_interp = ndinterp.ndInterp(axes, sigma_grid, order=3)

sigma_samp = sigma_interp.eval(point)

good = sigma_samp > 0. # TO FIX. REFF GRID TOO SMALL

lsigma_samp = np.log10(sigma_samp[good])
lmobs_samp = lmobs_samp[good]
lreff_samp = lreff_samp[good]

def nlogpfunc(p):
    lsigma_model = p[0] + p[1]*(lmobs_samp - lmobs_piv) + p[2]*(lreff_samp - lreff_piv)
    logp_samp = -0.5*(lsigma_samp - lsigma_model)**2/p[3]**2 - np.log(p[3])
    return -logp_samp.sum()

p0 = [2.3, 0.2, -0.2, 0.1]
bounds = [(2., 3.), (-1., 1.), (-1., 1.), (0.00001, 1.)]

print(nlogpfunc(p0))

res = minimize(nlogpfunc, p0, method='L-BFGS-B', bounds=bounds)
print(res.x)

