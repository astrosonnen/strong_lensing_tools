import numpy as np
from astropy.io import fits as pyfits
import h5py
import ndinterp
import emcee
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import leastsq
from fitpars import *
from parent_sample_pars import *
from psl_adcontr_funcs import get_msonly_pop
from read_slacs import *
from masssize import *
from halo_pars import *
import sl_cosmology
from sl_cosmology import Dang, Mpc, M_Sun, c, G, kpc
from scipy.stats import truncnorm
from scipy.special import erf
import sys
import warnings


nstep = 1000
burnin = 0
nwalkers = 100
npop = 1000
nis = 1000

nzd = 36
zd_grid = np.linspace(zmin, zmax, nzd)
arcsec2kpc_grid = np.zeros(nzd)
dvdz_grid = np.zeros(nzd)
for n in range(nzd):
    dvdz_grid[n] = sl_cosmology.comovd(zd_grid[n])**2 * sl_cosmology.dcomovdz(zd_grid[n])
    arcsec2kpc_grid[n] = sl_cosmology.arcsec2kpc(zd_grid[n])

dvdz_spline = splrep(zd_grid, dvdz_grid)
arcsec2kpc_spline = splrep(zd_grid, arcsec2kpc_grid)

nms = 101
ms_grid = np.linspace(lmchab_min, lmchab_max, nms)

def mtfunc(z):
    return mt0 + mt1*z + mt2*z**2 + mt3*z**3 + mt4*z**4 + mt5*z**5

def ftfunc(z, ms):
    return 1./np.pi * np.arctan((ms - mtfunc(z))/sigmat) + 0.5

def msdist(z, ms):
    return splev(z, dvdz_spline) * ftfunc(z, ms) * (10.**(ms - mbar))**(alpha + 1) * np.exp(-10.**(ms - mbar))

sigma_r = s19_sigma_r

# reads the Einstein radius and cross-section grids
lensgrid_file = h5py.File('adcontr_teincs_31x27x21_grids.hdf5', 'r')
eps_lens_grid = lensgrid_file['eps_grid'][()]
neps = len(eps_lens_grid)

cs_interps = []
tein_interps = []
ms_impsamp = np.zeros((nslacs, nis))

slacs_tein_err = slacs_tein * 0.05

mh_lens_grids = []
for n in range(nslacs):
    group = lensgrid_file[slacs_names[n]]

    lmstar_lens_grid = group['lmstar_grid'][()]
    nmstar = len(lmstar_lens_grid)

    mh_lens_grid = group['mh_grid'][()]
    mh_lens_grids.append(mh_lens_grid)
    nmh = len(mh_lens_grid)

    tein_lens_grid = group['tein_grid'][()]
    cs_lens_grid = group['cs_grid'][()]

    axes = {0: splrep(eps_lens_grid, np.arange(neps)), 1: splrep(mh_lens_grid, np.arange(nmh)), 2: splrep(lmstar_lens_grid, np.arange(nmstar))}

    cs_interp = ndinterp.ndInterp(axes, cs_lens_grid, order=1)
    cs_interps.append(cs_interp)

    tein_interp = ndinterp.ndInterp(axes, tein_lens_grid, order=1)
    tein_interps.append(tein_interp)

    ms_min_here = lmstar_lens_grid[0]
    ms_max_here = lmstar_lens_grid[-1] - 0.3

    scaled_ms_min = (ms_min_here - slacs_ms_obs[n])/slacs_ms_err[n]
    scaled_ms_max = (ms_max_here - slacs_ms_obs[n])/slacs_ms_err[n]

    ms_impsamp[n, :] = truncnorm.rvs(scaled_ms_min, scaled_ms_max, loc=slacs_ms_obs[n], scale=slacs_ms_err[n], size=nis)

mu_r_impsamp = hb09quad_mu_r_func(ms_impsamp)

mr_prior = np.zeros((nslacs, nis))
for n in range(nslacs):
    mr_prior[n, :] = msdist(slacs_zd[n], ms_impsamp[n]) * 1./sigma_r * np.exp(-0.5*(slacs_r[n] - mu_r_impsamp[n])**2/sigma_r**2)

lensgrid_file.close()

# transforms sigma in logsigma
slacs_lsigma_obs = np.log10(slacs_sigma_obs)
slacs_lsigma_err = 0.5*(np.log10(slacs_sigma_obs + slacs_sigma_err) - np.log10(slacs_sigma_obs - slacs_sigma_err))

# defines the model parameters

lasps_par = {'name': 'lasps', 'lower': 0., 'upper': 0.3, 'guess': 0.1, 'spread': 0.01}
eps_par = {'name': 'eps', 'lower': 0., 'upper': 1., 'guess': 0.5, 'spread': 0.01}

mu_mh_par = {'name': 'mu_mh', 'lower': 12., 'upper': 14., 'guess': mu_mh_obs, 'spread': 0.01}
beta_mh_par = {'name': 'beta_mh', 'lower': 0., 'upper': 2., 'guess': beta_mh_obs, 'spread': 0.01}
sigma_mh_par = {'name': 'sigma_mh', 'lower': 0.1, 'upper': 0.5, 'guess': sigma_mh_obs, 'spread': 0.01}

mu_sigma_par = {'name': 'mu_sigma', 'lower': 1., 'upper': 3., 'guess': 2.37, 'spread': 0.01}
beta_sigma_par = {'name': 'beta_sigma', 'lower': 0., 'upper': 1., 'guess': 0.26, 'spread': 0.01}
xi_sigma_par = {'name': 'xi_sigma', 'lower': -2., 'upper': 0., 'guess': -0.34, 'spread': 0.01}
nu_sigma_par = {'name': 'nu_sigma', 'lower': -1., 'upper': 1., 'guess': 0., 'spread': 0.01}
sigma_sigma_par = {'name': 'sigma_sigma', 'lower': 0., 'upper': 1., 'guess': 0.05, 'spread': 0.01}

pars = [lasps_par, eps_par, mu_mh_par, beta_mh_par, sigma_mh_par, mu_sigma_par, beta_sigma_par, xi_sigma_par, nu_sigma_par, sigma_sigma_par]

npars = len(pars)

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

warnings.filterwarnings("ignore", "overflow encountered in exp")

def pfind_func(tein_est, t, a):
    return 1./(1. + np.exp(-a*(tein_est - t)))

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    lasps, eps, mu_mh, beta_mh, sigma_mh, mu_sigma, beta_sigma, xi_sigma, nu_sigma, sigma_sigma = p

    sumlogp = -0.5*(mu_mh - mu_mh_obs)**2/mu_mh_err**2 - 0.5*(beta_mh - beta_mh_obs)**2/beta_mh_err**2 - 0.5*(sigma_mh - sigma_mh_obs)**2/sigma_mh_err**2
    for n in range(nslacs):

        mu_mh_impsamp = mu_mh + beta_mh * (ms_impsamp[n, :] - 11.3)
        scaled_mh_min = (mh_lens_grids[n][0] - mu_mh_impsamp)/sigma_mh
        scaled_mh_max = (mh_lens_grids[n][-1] - mu_mh_impsamp)/sigma_mh

        mh_impsamp = truncnorm.rvs(scaled_mh_min, scaled_mh_max, loc=mu_mh_impsamp, scale=sigma_mh, size=nis, random_state=n+1)

        lmstar_impsamp = ms_impsamp[n, :] + lasps
        lens_point = np.array((eps * np.ones(nis), mh_impsamp, lmstar_impsamp)).T

        tein_impsamp = tein_interps[n].eval(lens_point)

        tein_like = 1./slacs_tein_err[n] * np.exp(-0.5*(tein_impsamp - slacs_tein[n])**2/slacs_tein_err[n]**2)

        mu_sigma_impsamp = mu_sigma + beta_sigma * (ms_impsamp[n, :] - mpiv_slacs) + xi_sigma * (slacs_r[n] - mu_r_impsamp[n]) + nu_sigma * (mh_impsamp - mu_mh_impsamp)
        lsigma_term = 1./(sigma_sigma**2 + slacs_lsigma_err[n]**2)**0.5 * np.exp(-0.5*(mu_sigma_impsamp - slacs_lsigma_obs[n])**2/(sigma_sigma**2 + slacs_lsigma_err[n]**2))

        bad = (tein_impsamp < 0.)

        integrand_impsamp = mr_prior[n] * tein_like * lsigma_term
        integrand_impsamp[bad] = 0.
    
        sumlogp += np.log(integrand_impsamp.mean())

    if sumlogp != sumlogp:
        return -1e300

    return sumlogp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=50)

start = []
if len(sys.argv) > 1:
    print('using last step of %s to initialize walkers'%sys.argv[1])
    startfile = h5py.File('%s'%sys.argv[1], 'r')

    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for n in range(npars):
            tmp[n] = startfile[pars[n]['name']][i, -1]
        start.append(tmp)
    startfile.close()
else:
    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for j in range(npars):
            a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['spread'], (bounds[j][1] - pars[j]['guess'])/pars[j]['spread']
            p0 = truncnorm.rvs(a, b, size=1)*pars[j]['spread'] + pars[j]['guess']
            tmp[j] = p0[0]

        start.append(tmp)

print("Sampling")

sampler.run_mcmc(start, nstep)

blobchain = sampler.blobs

ml = sampler.lnprobability.argmax()

output_file = h5py.File('noseleff_inference.hdf5', 'w')
output_file.create_dataset('logp', data = sampler.lnprobability[:, burnin:])
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, burnin:, n])
    print('%s %3.2f'%(pars[n]['name'], np.median(sampler.chain[:, -1, n].flatten())))
