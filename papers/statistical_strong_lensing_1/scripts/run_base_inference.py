import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
import h5py
import sys


mockname = 'contrmock_0'

griddir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

nstep = 1000
nwalkers = 50

ngal = mockfile.attrs['ngal']

lmsps_piv = mockfile.attrs['lmsps_piv']
lmsps_err = mockfile.attrs['lmsps_err']
lmsps_mu = mockfile.attrs['lmsps_mu']
lmsps_sig = mockfile.attrs['lmsps_sig']
rmur_err = mockfile.attrs['rmur_err']

lreff_mu = mockfile.attrs['lreff_mu']
lreff_sig = mockfile.attrs['lreff_sig']
lreff_beta = mockfile.attrs['lreff_beta']

gammadm_min = 0.8
gammadm_max = 1.8

lmsps_obs = mockfile['lmsps_obs'][()]
lmsps_samp = mockfile['lmsps_true'][()]
lmstar_samp = mockfile['lmstar'][()]
lreff_samp = mockfile['lreff'][()]
rein_samp = mockfile['rein'][()]
rmur_obs = mockfile['rmur_obs'][()]

lmstar_grids = []
detJ_grids = []
grid_ranges = []
rmur_grids = []
beta_grids = []
beta_max_grids = []
grid_zeros = []

grids_file = h5py.File(griddir+'/%s_lensmodel_grids.hdf5'%mockname, 'r')

lmdm5_grid = grids_file['lmdm5_grid'][()]
gammadm_grid = grids_file['gammadm_grid'][()]
ngammadm = len(gammadm_grid)
nlmdm5 = len(lmdm5_grid)

gammadm_extgrid = np.dot(gammadm_grid.reshape((ngammadm, 1)), np.ones((1, nlmdm5)))
lmdm5_extgrid = np.dot(np.ones((ngammadm, 1)), lmdm5_grid.reshape((1, nlmdm5)))

for i in range(ngal):

    group = grids_file['lens_%04d'%i]
    
    lmstar_grids.append(group['lmstar_grid'][()])
    beta_grids.append(group['beta_grid'][()])
    beta_max_grids.append(group['beta_max_grid'][()])
    detJ_grids.append(group['detJ_grid'][()])
    rmur_grids.append(group['rmur_grid'][()])
    grid_ranges.append(group['grid_range'][()])
    grid_zeros.append(np.logical_not(group['grid_range'][()]))

# now runs the inference

lmdm5_mu_par = {'name': 'lmdm5_mu', 'lower': 10., 'upper': 12., 'guess': 11., 'spread': 0.1}
lmdm5_sig_par = {'name': 'lmdm5_sig', 'lower': 0.02, 'upper': 0.5, 'guess': 0.1, 'spread': 0.03}

gammadm_mu_par = {'name': 'gammadm_mu', 'lower': gammadm_min, 'upper': gammadm_max, 'guess': 1.55, 'spread': 0.03}
gammadm_sig_par = {'name': 'gammadm_sig', 'lower': 0.02, 'upper': 0.5, 'guess': 0.1, 'spread': 0.1}

laimf_par = {'name': 'laimf', 'lower': 0., 'upper': 0.25, 'guess': mockfile.attrs['laimf_mu'], 'spread': 0.01}

mockfile.close()

pars = [lmdm5_mu_par, lmdm5_sig_par, gammadm_mu_par, gammadm_sig_par, laimf_par]
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

    lmdm5_mu, lmdm5_sig, gammadm_mu, gammadm_sig, laimf = p

    sumlogp = 0.

    mdm5_term = 1./lmdm5_sig * np.exp(-0.5*(lmdm5_grid - lmdm5_mu)**2/lmdm5_sig**2)

    gammadm_term = 1./gammadm_sig * np.exp(-0.5*(gammadm_extgrid - gammadm_mu)**2/gammadm_sig**2)
    gammadm_norm = 0.5*(erf((gammadm_max - gammadm_mu)/2.**0.5/gammadm_sig) - erf((gammadm_min - gammadm_mu)/2.**0.5/gammadm_sig))
    gammadm_term /= gammadm_norm

    # loops over the strong lenses
    for i in range(ngal):

        lmsps_grid = lmstar_grids[i] - laimf
        lreff_muhere = lreff_mu + lreff_beta * (lmsps_grid - lmsps_piv)

        msps_term = 1./lmsps_sig * np.exp(-0.5*(lmsps_grid - lmsps_mu)**2/lmsps_sig**2)
        reff_term = 1./lreff_sig * np.exp(-0.5*(lreff_muhere - lreff_samp[i])**2/lreff_sig**2)

        msps_like = 1./lmsps_err * np.exp(-0.5*(lmsps_grid - lmsps_obs[i])**2/lmsps_err**2)

        beta_prior = 2.*beta_grids[i]/beta_max_grids[i]**2
        rmur_like = 1./rmur_err * np.exp(-0.5*(rmur_grids[i] - rmur_obs[i])**2/rmur_err**2)

        full_integrand = msps_term * msps_like * reff_term * mdm5_term * rmur_like * beta_prior * abs(detJ_grids[i]) * gammadm_term

        full_integrand[grid_zeros[i]] = 0.

        gammadm_integrand = np.zeros(ngammadm)

        for j in range(ngammadm):
            spline = splrep(lmdm5_grid, full_integrand[j, :])
            gammadm_integrand[j] = splint(lmdm5_grid[0], lmdm5_grid[-1], spline)

        spline = splrep(gammadm_grid, gammadm_integrand)
        integral = splint(gammadm_grid[0], gammadm_grid[-1], spline)
        sumlogp += np.log(integral)

        if sumlogp != sumlogp:
            return -1e300

    return sumlogp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=50)

start = []
if len(sys.argv) > 2:
    print 'using last step of %s to initialize walkers'%sys.argv[2]
    startfile = h5py.File('%s'%sys.argv[2], 'r')

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

output_file = h5py.File('chains/%s_base_inference.hdf5'%mockname, 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])

