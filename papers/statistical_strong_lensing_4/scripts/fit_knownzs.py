import numpy as np
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


mockname = '1e4mock_0'

griddir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

nstep = 1000
nwalkers = 50
nis = 10000
is_thin = 1
is_length = 100
is_indices = is_thin * np.arange(is_length//is_thin)

seedno = 0
np.random.seed(seedno)

nlens = mockfile['Galaxies/lens'][()].sum()

llum_piv = 11.

lreff_mu_true = mockfile.attrs['lreff_mu']
lreff_sig_true = mockfile.attrs['lreff_sig']
lreff_beta_true = mockfile.attrs['lreff_beta']

llum_samp = mockfile['Lenses/llum'][()]
lreff_samp = mockfile['Lenses/lreff'][()]

magA_obs = mockfile['Lenses/magA_obs'][()]
magB_obs = mockfile['Lenses/magB_obs'][()]

# reads the cross-section grid
cs_grid_file = h5py.File(griddir+'/%s_crosssect_grid.hdf5'%mockname, 'r')

cs_lm200_grid = cs_grid_file['lm200_grid'][()]
cs_lmstar_grid = cs_grid_file['lmstar_grid'][()]
cs_lreff_grid = cs_grid_file['lreff_grid'][()]
cs_zs_grid = cs_grid_file['zs_grid'][()]
cs_ms_grid = cs_grid_file['ms_grid'][()]
cs_beta_grid = cs_grid_file['beta_grid'][()]
cs_lbeta_grid = np.log10(cs_beta_grid)

axes = {}
axes[0] = splrep(cs_zs_grid, np.arange(len(cs_zs_grid)))
axes[1] = splrep(cs_lm200_grid, np.arange(len(cs_lm200_grid)))
axes[2] = splrep(cs_lmstar_grid, np.arange(len(cs_lmstar_grid)))
axes[3] = splrep(cs_lreff_grid, np.arange(len(cs_lreff_grid)))
axes[4] = splrep(cs_ms_grid, np.arange(len(cs_ms_grid)))

cs_interp = ndinterp.ndInterp(axes, cs_grid_file['crosssect_grid'][()], order=3)

# source distribution parameters

ms_min = mockfile.attrs['ms_min']
ms_max = mockfile.attrs['ms_max']

maxmagB_det = mockfile.attrs['maxmagB_det'] # minimum observed magnitude of 2nd image
mag_err = mockfile.attrs['mag_err'] # uncertainty on the observed magnitudes

# defines the galaxy luminosity distribution
llum_llstar = mockfile.attrs['llum_llstar']
llum_alpha = mockfile.attrs['llum_alpha']
llum_min = mockfile.attrs['llum_min']
llum_max = mockfile.attrs['llum_max']

def gal_phifunc(llum):
    return (10.**(llum - llum_llstar))**(llum_alpha +1.) * np.exp(-10.**(llum - llum_llstar))
 
gal_norm = 1./quad(gal_phifunc, llum_min, llum_max)[0]

# obtains a spline of the inverse function of the cumulative probability
ngrid = 101
llum_grid = np.linspace(llum_min, llum_max, ngrid)
cumgalphi_grid = 0.*llum_grid
for i in range(ngrid):
    cumgalphi_grid[i] = quad(gal_phifunc, llum_min, llum_grid[i])[0]*gal_norm
invcumgalphi_spline = splrep(cumgalphi_grid, llum_grid)

# draws random sample for MC integral
llum_is = splev(np.random.rand(nis), invcumgalphi_spline)
lustar_is = np.random.normal(0., 1., nis)
lreff_is = np.random.normal(0., 1., nis)
lm200_is = np.random.normal(0., 1., nis)

# draws nlens sources from P(zs,ms)
zs_grid = mockfile['Grids/zs_grid'][()]
cumpz_grid = mockfile['Grids/cumpz_grid'][()]
invcumpz_spline = splrep(cumpz_grid, zs_grid)
zs_is = splev(np.random.rand(nis), invcumpz_spline)

mstar_sch_spline = splrep(zs_grid, mockfile['Grids/mstar_sch_grid'][()])
ms_grid = mockfile['Grids/ms_grid'][()]

deltam_min = -3.
deltam_grid = mockfile['Grids/deltam_grid'][()]
cumfullphi_grid = mockfile['Grids/cumfullphi_grid'][()]
cumfullphi_spline = splrep(deltam_grid, cumfullphi_grid)

tms_samp = np.random.rand(nis)
# obtains ms for every source
ms_is = 0.*tms_samp
print('Drawing source magnitudes...')
for i in range(nis):
    mstar_here = splev(zs_is[i], mstar_sch_spline)
    
    deltams_grid = ms_grid - mstar_here
    deltams_cut = deltams_grid > deltam_min
    deltams_grid = deltams_grid[deltams_cut]
    
    cumphi_grid = splev(deltams_grid, cumfullphi_spline)
    cumphi_grid -= cumphi_grid[0]
    cumphi_grid /= (cumphi_grid[-1] - cumphi_grid[0])
    
    invcumphi_spline = splrep(cumphi_grid, ms_grid[deltams_cut])

    ms_here = splev(tms_samp[i], invcumphi_spline)
    ms_is[i] = ms_here

# reads the lens model samples

lm200_samps = []
lustar_samps = []
magB_samps = []
beta_samps = []

samps_file = h5py.File(griddir+'/%s_lensmodel_knownzs_samps.hdf5'%mockname, 'r')

for i in range(nlens):
    group = samps_file['lens_%04d'%i]
    
    lm200_samps.append(group['lm200_samp'][:, is_indices].flatten())
    lustar_samps.append(group['lustar_samp'][:, is_indices].flatten())
    beta_samps.append(group['beta_samp'][:, is_indices].flatten())
    magB_samps.append(group['magB_samp'][:, is_indices].flatten())

# now runs the inference

lreff_sig_par = {'name': 'lreff_sig', 'lower': 0., 'upper': 0.5, 'guess': lreff_sig_true, 'spread': 0.01}
lreff_beta_par = {'name': 'lreff_beta', 'lower': 0., 'upper': 2., 'guess': lreff_beta_true, 'spread': 0.01}

lm200_mu_par = {'name': 'lm200_mu', 'lower': 12.5, 'upper': 13.5, 'guess': 13., 'spread': 0.01}
lm200_sig_par = {'name': 'lm200_sig', 'lower': 0., 'upper': 0.5, 'guess': 0.2, 'spread': 0.01}
lm200_beta_par = {'name': 'lm200_beta', 'lower': 0., 'upper': 3., 'guess': 1.5, 'spread': 0.01}

lustar_mu_par = {'name': 'lustar_mu', 'lower': 0.3, 'upper': 0.7, 'guess': mockfile.attrs['lustar_mu'], 'spread': 0.01}
lustar_sig_par = {'name': 'lustar_sig', 'lower': 0., 'upper': 0.3, 'guess': mockfile.attrs['lustar_sig'], 'spread': 0.01}

mockfile.close()

lreff_mu = lreff_mu_true

pars = [lreff_sig_par, lreff_beta_par, lm200_mu_par, lm200_sig_par, lm200_beta_par, lustar_mu_par, lustar_sig_par]
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

    lreff_sig, lreff_beta, lm200_mu, lm200_sig, lm200_beta, lustar_mu, lustar_sig = p
    sumlogp = 0.

    # normalizes the population probability distribution.
    # rescales the important sample in lm200, lustar, reff

    lustar_is_here = lustar_mu + lustar_sig * lustar_is
    lmstar_is_here = llum_is + lustar_is_here

    lreff_is_muhere = lreff_mu + lreff_beta * (lmstar_is_here - (lustar_mu + llum_piv))
    lreff_is_here = lreff_is_muhere + lreff_sig * lreff_is
    lm200_is_muhere = lm200_mu + lm200_beta * (lmstar_is_here - (lustar_mu + llum_piv))
    lm200_is_here = lm200_is_muhere + lm200_sig * lm200_is

    lmstar_too_large = lmstar_is_here > cs_lmstar_grid[-1]
    lmstar_too_small = lmstar_is_here < cs_lmstar_grid[0]

    lm200_too_large = lm200_is_here > cs_lm200_grid[-1]
    lm200_too_small = lm200_is_here < cs_lm200_grid[0]

    lreff_too_large = lreff_is_here > cs_lreff_grid[-1]
    lreff_too_small = lreff_is_here < cs_lreff_grid[0]

    cs_point = np.array((zs_is, lm200_is_here, lmstar_is_here, lreff_is_here, ms_is)).T
    cs_is = cs_interp.eval(cs_point)

    out_of_bounds = lmstar_too_large | lmstar_too_small | lm200_too_large | lm200_too_small | lreff_too_large | lreff_too_small
    noob = out_of_bounds.sum()
    ngood = nis - noob

    if noob > 0.5*nis:
        return -1e300

    pop_norm = ngood/cs_is.sum()

    # loops over the strong lenses
    for i in range(nlens):

        lmstar_samp = lustar_samps[i] + llum_samp[i]

        lreff_muhere = lreff_mu + lreff_beta * (lmstar_samp - (lustar_mu + llum_piv))

        lum_term = gal_phifunc(llum_samp[i])

        lreff_term = 1./lreff_sig * np.exp(-0.5*(lreff_muhere - lreff_samp[i])**2/lreff_sig**2)

        lm200_muhere = lm200_mu + lm200_beta * (lmstar_samp - (lustar_mu + llum_piv))
        lm200_term = 1./lm200_sig * np.exp(-0.5*(lm200_samps[i] - lm200_muhere)**2/lm200_sig**2)
        lustar_term = 1./lustar_sig * np.exp(-0.5*(lustar_samps[i] - lustar_mu)**2/lustar_sig**2)

        beta_prior = beta_samps[i] * (1. - erf((magB_samps[i] - maxmagB_det)/2.**0.5/mag_err))
       
        full_integrand = pop_norm * lum_term * lreff_term * lm200_term * lustar_term * beta_prior

        integral = full_integrand.sum()
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

blobchain = sampler.blobs

output_file = h5py.File('../chains/%s_knownzs_inference.hdf5'%mockname, 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])

