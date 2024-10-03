import numpy as np
import h5py
import emcee
import sys
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev, splint
import ndinterp
from lenspars import *
from halo_pars import dndmh, shmr, sigmalogms
from masssize_pars import *
from sl_profiles import deVaucouleurs as deV, nfw


nwalkers = 50
nstep = 2500
burnin = 500

lreff_obs = np.log10(reff_kpc)

# loads the Einstein radius and lensing cross-section grids
grid_file = h5py.File('lensmodel_grid.hdf5', 'r')
lmvir_grid = grid_file['lmvir_grid'][()]
lmstar_grid = grid_file['lmstar_grid'][()]

axes = {0: splrep(lmvir_grid, np.arange(len(lmvir_grid))), 1: splrep(lmstar_grid, np.arange(len(lmstar_grid)))}

tein_interp = ndinterp.ndInterp(axes, grid_file['tein_grid'][()])

# defines the model parameters, initial guess, bounds and spread.
laimf_par = {'name': 'laimf', 'lower': -0.1, 'upper': 0.3, 'guess': 0.1, 'spread': 0.01}
lmsps_par = {'name': 'lmsps', 'lower': 10., 'upper': 12., 'guess': lmsps_obs, 'spread': 0.01}
lmvir_par = {'name': 'lmvir', 'lower': lmvir_grid[0], 'upper': lmvir_grid[-1], 'guess': 12.5, 'spread': 0.01}

pars = [laimf_par, lmsps_par, lmvir_par]

npars = len(pars)

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logpfunc(p):

    laimf, lmsps, lmvir = p

    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300, 0.

    hmf_lprior = np.log(dndmh(lmvir)) # halo mass function prior

    shmr_mu = shmr(lmvir) # average log(Msps) from the SHMR

    # SHMR prior: Gaussian in log(Msps)
    shmr_lprior = -0.5*(shmr_mu - lmsps)**2/sigmalogms**2 - np.log(sigmalogms)

    # mass-size relation prior on log(Msps)
    mrprior_mu = masssize_mpiv + (lreff_obs - masssize_mu)/masssize_beta
    mrprior_sigma = masssize_sigma/masssize_beta

    masssize_lprior = -0.5*(mrprior_mu - lmsps)**2/mrprior_sigma**2 - np.log(mrprior_sigma)

    # observed stellar mass likelihood
    lmsps_like = -0.5*(lmsps - lmsps_obs)**2/lmsps_err**2

    # predicts the Einstein radius
    lmstar = lmsps + laimf # log stellar mass
    tein = tein_interp.eval(np.array((lmvir, lmstar)).reshape((1, 2)))

    tein_llike = -0.5*(tein - tein_obs)**2/tein_err**2

    return hmf_lprior + shmr_lprior + masssize_lprior + lmsps_like + tein_llike, tein

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=nwalkers)

start = []
if len(sys.argv) > 1:
    print('using last spread of %s to initialize walkers'%sys.argv[1])
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

output = h5py.File('noseleff_inference.hdf5', 'w')
output.create_dataset('logp', data=sampler.lnprobability)

tein_chain = sampler.blobs.reshape((nstep, nwalkers)).T

for n in range(npars):
    output.create_dataset(pars[n]['name'], data=sampler.chain[:, burnin:, n])
    med_here = np.median(sampler.chain[:, -1, n])
    print('%s %3.2f'%(pars[n]['name'], med_here))

output.create_dataset('tein', data=tein_chain[:, burnin:])

