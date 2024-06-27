import numpy as np
from astropy.io import fits as pyfits
import h5py
import ndinterp
import emcee
from scipy.interpolate import splrep, splev, splint
from fitpars import *
from parent_sample_pars import *
import sl_cosmology
from scipy.stats import truncnorm
from scipy.special import erf
import sys


nstep = 300
nwalkers = 100
nis = 1000

f = open('../SLACS_table.cat', 'r')
names = np.loadtxt(f, usecols=(0, ), dtype=str)
f.close()

nslacs = len(names)

f = open('../SLACS_table.cat', 'r')
slacs_reff_kpc, slacs_ms_obs, slacs_ms_err, slacs_sigma_obs, slacs_sigma_err = np.loadtxt(f, usecols=(6, 9, 10, 11, 12), unpack=True)
f.close()

slacs_r = np.log10(slacs_reff_kpc)

ms_impsamp = np.zeros((nslacs, nis))

# reads the lensing and dynamics grids
lensgrid_file = h5py.File('slacs_lensing_grids.hdf5', 'r')
gamma_lens_grid = lensgrid_file['gamma_grid'][()]
ngamma = len(gamma_lens_grid)

jeans_file = h5py.File('slacs_jeans_grids.hdf5', 'r')
gamma_jeans_grid = jeans_file['gamma_grid'][()]

gamma_impsamp = np.random.normal(0., 1., (nslacs, nis))

m5_splines = []
dm5drein_splines = []
s2_splines = []
for n in range(nslacs):
    m5_grid = lensgrid_file[names[n]]['m5_grid'][()]
    m5_splines.append(splrep(gamma_lens_grid, m5_grid))
    dm5drein_grid = lensgrid_file[names[n]]['dm5drein_grid'][()]
    dm5drein_splines.append(splrep(gamma_lens_grid, dm5drein_grid))
    s2_grid = jeans_file[names[n]]['s2_grid'][()]
    s2_spline = splrep(gamma_jeans_grid, s2_grid)
    s2_splines.append(s2_spline)

    ms_impsamp[n, :] = np.random.normal(slacs_ms_obs[n], slacs_ms_err[n], nis)

lensgrid_file.close()
jeans_file.close()

mu_r_impsamp = hb09quad_mu_r_func(ms_impsamp)

# defines the model parameters

mu_ms_par = {'name': 'mu_ms', 'lower': 10., 'upper': 12., 'guess': slacs_ms_obs.mean(), 'spread': 0.03}
sigma_ms_par = {'name': 'sigma_ms', 'lower': 0., 'upper': 1., 'guess': slacs_ms_obs.std(), 'spread': 0.03}

mu_m5_par = {'name': 'mu_m5', 'lower': 10., 'upper': 12., 'guess': 11.3, 'spread': 0.1}
sigma_m5_par = {'name': 'sigma_m5', 'lower': 0., 'upper': 1., 'guess': 0.1, 'spread': 0.03}
beta_m5_par = {'name': 'beta_m5', 'lower': -5., 'upper': 5., 'guess': 0., 'spread': 0.03}
xi_m5_par = {'name': 'xi_m5', 'lower': -5., 'upper': 5., 'guess': 0., 'spread': 0.03}

mu_gamma_par = {'name': 'mu_gamma', 'lower': 1.5, 'upper': 2.5, 'guess': 2., 'spread': 0.03}
sigma_gamma_par = {'name': 'sigma_gamma', 'lower': 0.01, 'upper': 1., 'guess': 0.1, 'spread': 0.03}
beta_gamma_par = {'name': 'beta_gamma', 'lower': -5., 'upper': 5., 'guess': 0., 'spread': 0.03}
xi_gamma_par = {'name': 'xi_gamma', 'lower': -5., 'upper': 5., 'guess': 0., 'spread': 0.03}

pars = [mu_ms_par, sigma_ms_par, mu_m5_par, sigma_m5_par, beta_m5_par, xi_m5_par, mu_gamma_par, sigma_gamma_par, beta_gamma_par, xi_gamma_par]

npars = len(pars)

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    mu_ms, sigma_ms, mu_m5, sigma_m5, beta_m5, xi_m5, mu_gamma, sigma_gamma, beta_gamma, xi_gamma = p

    sumlogp = 0.

    for n in range(nslacs):

        mu_gamma_here = mu_gamma + beta_gamma*(ms_impsamp[n, :] - mpiv_slacs) + xi_gamma*(slacs_r[n] - mu_r_impsamp[n, :])
        scaled_gamma_min_here = (gamma_min - mu_gamma_here)/sigma_gamma
        scaled_gamma_max_here = (gamma_max - mu_gamma_here)/sigma_gamma

        gamma_impsamp_here = truncnorm.rvs(scaled_gamma_min_here, scaled_gamma_max_here, loc=mu_gamma_here, scale=sigma_gamma, size=nis, random_state=n+1)
        m5_impsamp_here = splev(gamma_impsamp_here, m5_splines[n])
        dm5drein_impsamp_here = splev(gamma_impsamp_here, dm5drein_splines[n])

        mu_m5_here = mu_m5 + beta_m5*(ms_impsamp[n, :] - mpiv_slacs) + xi_m5*(slacs_r[n] - mu_r_impsamp[n, :])

        ms_term = 1./sigma_ms * np.exp(-0.5*(ms_impsamp[n, :] - mu_ms)**2/sigma_ms**2)
        m5_term = 1./sigma_m5 * np.exp(-0.5*(m5_impsamp_here - mu_m5_here)**2/sigma_m5**2)
        
        sigma_model = (10.**m5_impsamp_here * splev(gamma_impsamp_here, s2_splines[n]))**0.5
        sigma_like = 1./slacs_sigma_err[n] * np.exp(-0.5*(sigma_model - slacs_sigma_obs[n])**2/slacs_sigma_err[n]**2)

        integrand = sigma_like * ms_term * m5_term * dm5drein_impsamp_here
    
        sumlogp += np.log(integrand.sum())

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
            tmp[j] = p0

        start.append(tmp)

print("Sampling")

sampler.run_mcmc(start, nstep)

blobchain = sampler.blobs

ml = sampler.lnprobability.argmax()

output_file = h5py.File('slonly_inference.hdf5', 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])
    print('%s %3.2f'%(pars[n]['name'], sampler.chain[:, :, n].flatten()[ml]))

