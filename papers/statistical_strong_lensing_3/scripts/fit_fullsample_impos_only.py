import numpy as np
from math import factorial
import sl_cosmology
from scipy.integrate import quad
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
import h5py
import sys
import ndinterp


mockname = '1e5mock_0'

griddir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

nstep = 2000
nwalkers = 50
nis = 1000000

ngal = mockfile.attrs['ngal']
nlens = mockfile['Galaxies/lens'][()].sum()
nnot = ngal - nlens

nbkg = mockfile.attrs['nbkg'] / 3600.

magA_obs = mockfile['Lenses/magA_obs'][()]
magB_obs = mockfile['Lenses/magB_obs'][()]
islens = mockfile['Galaxies/lens'][()]
nsource_lens = mockfile['Galaxies/nsource'][islens]

# reads the cross-section grid
cs_grid_file = h5py.File(griddir+'/%s_crosssect_grid.hdf5'%mockname, 'r')

cs_gamma_grid = cs_grid_file['gamma_grid'][()]
cs_ms_grid = cs_grid_file['ms_grid'][()]

axes = {}
axes[0] = splrep(cs_gamma_grid, np.arange(len(cs_gamma_grid)))
axes[1] = splrep(cs_ms_grid, np.arange(len(cs_ms_grid)))

cs_interp = ndinterp.ndInterp(axes, cs_grid_file['crosssect_grid'][()], order=1)

# source distribution parameters

zs = mockfile.attrs['zs'] # source redshift
ds = sl_cosmology.Dang(zs)
alpha_sch = mockfile.attrs['alpha_sch']
Mstar_sch = mockfile.attrs['Mstar_sch']

mstar_sch = Mstar_sch + 2.5*np.log10(ds**2/1e-10*(1.+zs))

ms_min = mockfile.attrs['ms_min']
ms_max = mockfile.attrs['ms_max']

maxmagB_det = mockfile.attrs['maxmagB_det'] # minimum observed magnitude of 2nd image
mag_err = mockfile.attrs['mag_err'] # uncertainty on the observed magnitudes

def phifunc(m):
    return (10.**(-0.4*(m - mstar_sch)))**(alpha_sch+1.) * np.exp(-10.**(-0.4*(m - mstar_sch)))
 
# calculates the normalisation constant
phi_invnorm = quad(phifunc, ms_min, ms_max)[0]

# obtains a spline of the inverse function of the cumulative probability
ngrid = 101
mag_grid = np.linspace(ms_min, ms_max, ngrid)
cumphi_grid = 0.*mag_grid
for i in range(ngrid):
    cumphi_grid[i] = quad(phifunc, ms_min, mag_grid[i])[0]/phi_invnorm
invcumphi_spline = splrep(cumphi_grid, mag_grid)

# draws random sample for MC integral
ltein_is = np.random.normal(0., 1., nis)
gamma_is = np.random.normal(0., 1., nis)
ms_is = splev(np.random.rand(nis), invcumphi_spline)

# reads the lens model grids

ltein_grids = []
detJ_grids = []
grid_ranges = []
grid_zeros = []
beta_grids = []
mag_integrals = []

grids_file = h5py.File(griddir+'/%s_lensmodel_grids.hdf5'%mockname, 'r')

gamma_grid = grids_file['gamma_grid'][()]
ms_grid = grids_file['ms_grid'][()]
ngamma = len(gamma_grid)
nms = len(ms_grid)

for i in range(nlens):
    group = grids_file['lens_%04d'%i]
    
    ltein_grids.append(np.log10(group['tein_grid'][()]))
    beta_grids.append(group['beta_grid'][()])
    detJ_grids.append(group['detJ_grid'][()])
    grid_ranges.append(group['grid_range'][()])
    grid_zeros.append(np.logical_not(group['grid_range'][()]))

    # calculates the integral over the source magnitude 
    # (it does not depend on the hyper-parameters)
    # (well, the normalization does, but that's taken care of)
    mag_integral = np.zeros(ngamma)
    print(i)
    for j in range(ngamma):
        crosssect_grid_here = group['crosssect_grid'][j, :]
        crosssect_spline = splrep(ms_grid, crosssect_grid_here)
        lam_grid_here = nbkg * crosssect_grid_here
        poiss_grid_here = lam_grid_here**nsource_lens[i] * np.exp(-lam_grid_here) / factorial(nsource_lens[i])
        poiss_spline = splrep(ms_grid, poiss_grid_here)
        muB_here = group['muB_grid'][j]
        def mag_integrand(m):
            magB_here = m - 2.5*np.log10(abs(muB_here))
            return phifunc(m)/phi_invnorm * (1. - erf((magB_here - maxmagB_det)/2.**0.5/mag_err)) / splev(m, crosssect_spline) * splev(m, poiss_spline)
        if group['grid_range'][j]:
            mag_integral[j] = quad(mag_integrand, ms_min, ms_max)[0]

    mag_integrals.append(mag_integral)

# now runs the inference

gamma_mu_par = {'name': 'gamma_mu', 'lower': 1.8, 'upper': 2.2, 'guess': 2., 'spread': 0.01}
gamma_sig_par = {'name': 'gamma_sig', 'lower': 0.02, 'upper': 0.5, 'guess': 0.2, 'spread': 0.01}

ltein_mu_par = {'name': 'ltein_mu', 'lower': -0.5, 'upper': 0.5, 'guess': mockfile.attrs['ltein_mu'], 'spread': 0.01}
ltein_sig_par = {'name': 'ltein_sig', 'lower': 0., 'upper': 0.3, 'guess': mockfile.attrs['ltein_sig'], 'spread': 0.01}

mockfile.close()

pars = [gamma_mu_par, gamma_sig_par, ltein_mu_par, ltein_sig_par]
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

    gamma_mu, gamma_sig, ltein_mu, ltein_sig = p

    ltein_is_here = ltein_mu + ltein_sig * ltein_is
    tein_is_here = 10.**ltein_is_here

    gamma_is_here = gamma_mu + gamma_sig * gamma_is

    pop_good = (gamma_is_here > cs_gamma_grid[0]) & (gamma_is_here < cs_gamma_grid[-1])
    npop_good = pop_good.sum()

    cs_point = np.array((gamma_is_here, ms_is)).T
    cs_is = tein_is_here**2 * cs_interp.eval(cs_point)
    
    lambda_is = nbkg * cs_is[pop_good]
 
    sumlogp = nnot * np.log(np.exp(-lambda_is).sum()/float(npop_good))

    # loops over the strong lenses
    for i in range(nlens):

        gamma_term = 1./gamma_sig * np.exp(-0.5*(gamma_grid - gamma_mu)**2/gamma_sig**2)
        gamma_norm = 0.5*(erf((gamma_grid[-1] - gamma_mu)/2.**0.5/gamma_sig) - erf((gamma_grid[0] - gamma_mu)/2.**0.5/gamma_sig))
        gamma_term /= gamma_norm

        ltein_term = 1./ltein_sig * np.exp(-0.5*(ltein_grids[i] - ltein_mu)**2/ltein_sig**2)

        beta_prior = beta_grids[i]

        full_integrand = gamma_term * ltein_term * beta_prior * abs(detJ_grids[i]) * mag_integrals[i]

        full_integrand[grid_zeros[i]] = 0.

        spline = splrep(gamma_grid, full_integrand)
        integral = splint(gamma_grid[0], gamma_grid[-1], spline)

        sumlogp += np.log(integral)

        if sumlogp != sumlogp:
            return -1e300

    return sumlogp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=50)

start = []
if len(sys.argv) > 2:
    print('using last step of %s to initialize walkers'%sys.argv[2])
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

output_file = h5py.File('../chains/%s_fullsample_impos_only_inference.hdf5'%mockname, 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])

