import numpy as np
import sl_cosmology
from parent_sample_pars import *
from scipy.integrate import quad, dblquad
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
from astropy.io import fits as pyfits
import h5py
import sys
#import ndinterp


nstep = 300
nwalkers = 100

nsub = 10000 # number of galaxies used for the fit (10k is more than enough for our purposes)
nis = 100 # number of samples for integral

seedno = 0
np.random.seed(seedno)

# loads the parent sample
table = pyfits.open('../parent_sample.fits')[1].data

indices = np.arange(len(table))
subsamp = np.random.choice(indices, size=nsub, replace=False)

z_samp = table['z'][subsamp]
ms_samp = table['lmchab'][subsamp]
r_samp = np.log10(table['reff_kpc'][subsamp])
ms_err_samp = table['lmchab_err'][subsamp]
ms_impsamp = np.zeros((nis, nsub))
for n in range(nsub):
    ms_impsamp[:, n] = np.random.normal(ms_samp[n], ms_err_samp[n], nis)

nz = 36
z_grid = np.linspace(zmin, zmax, nz)
dvdz_grid = np.zeros(nz)
for n in range(nz):
    dvdz_grid[n] = sl_cosmology.comovd(z_grid[n])**2 * sl_cosmology.dcomovdz(z_grid[n])
dvdz_spline = splrep(z_grid, dvdz_grid)

dvdz_samp = splev(z_samp, dvdz_spline)

nms = 101
ms_grid = np.linspace(lmchab_min, lmchab_max, nms)

# defines the model parameters

mbar_par = {'name': 'mbar', 'lower': 9., 'upper': 12., 'guess': 11., 'spread': 0.1}
alpha_par = {'name': 'alpha', 'lower': -2., 'upper': 2., 'guess': -1., 'spread': 0.1}

mt0_par = {'name': 'mt0', 'lower': 8., 'upper': 12., 'guess': 9., 'spread': 0.1}
mt1_par = {'name': 'mt1', 'lower': -10., 'upper': 10., 'guess': 10., 'spread': 1.}
mt2_par = {'name': 'mt2', 'lower': -100., 'upper': 100., 'guess': 60., 'spread': 10.}
mt3_par = {'name': 'mt3', 'lower': -1000., 'upper': 1000., 'guess': -400., 'spread': 100.}
mt4_par = {'name': 'mt4', 'lower': -1000., 'upper': 1000., 'guess': 500., 'spread': 100.}
mt5_par = {'name': 'mt5', 'lower': -1000., 'upper': 1000., 'guess': -100., 'spread': 100.}
sigmat_par = {'name': 'sigmat', 'lower': 0., 'upper': 2., 'guess': 0.1, 'spread': 0.03}

pars = [mbar_par, alpha_par, mt0_par, mt1_par, mt2_par, mt3_par, mt4_par, mt5_par, sigmat_par]

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

    mbar, alpha, mt0, mt1, mt2, mt3, mt4, mt5, sigmat = p

    # computes the normalization of the distribution
    def mtfunc(z):
        return mt0 + mt1*z + mt2*z**2 + mt3*z**3 + mt4*z**4 + mt5*z**5
    
    def ftfunc(z, ms):
        return 1./np.pi * np.arctan((ms - mtfunc(z))/sigmat) + 0.5
    
    def msdist(z, ms):
        return splev(z, dvdz_spline) * ftfunc(z, ms) * (10.**(ms - mbar))**(alpha + 1) * np.exp(-10.**(ms - mbar))
    
    #invnorm = dblquad(msdist, lmchab_min, lmchab_max, zmin, zmax)[0] too slow!
    msint_grid = np.zeros(nz)
    for n in range(nz):
        integrand_grid = msdist(z_grid[n], ms_grid)
        integrand_spline = splrep(ms_grid, integrand_grid)
        msint_grid[n] = splint(ms_grid[0], ms_grid[-1], integrand_spline)
    
    msint_spline = splrep(z_grid, msint_grid)
    invnorm = splint(z_grid[0], z_grid[-1], msint_spline)

    p_integrand = 1./invnorm * ftfunc(z_samp, ms_impsamp) * (10.**(ms_impsamp - mbar))**(alpha + 1.) * np.exp(-10.**(ms_impsamp - mbar)) * dvdz_samp

    p_integrals = p_integrand.sum(axis=0)

    sumlogp = np.log(p_integrals).sum()

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

output_file = h5py.File('mz_inference.hdf5', 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])
    print('%s %3.2f'%(pars[n]['name'], sampler.chain[:, :, n].flatten()[ml]))

