import numpy as np
from astropy.io import fits as pyfits
import h5py
import ndinterp
import emcee
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import leastsq
from fitpars import *
from parent_sample_pars import *
from read_slacs import *
import mz_distribution
import sl_cosmology
from sl_cosmology import Dang, Mpc, M_Sun, c, G, kpc
from scipy.stats import truncnorm
from scipy.special import erf
import sys


nstep = 1000
nwalkers = 100
nis = 1000
npop = 10000

# draws a sample from the redshift-stellar mass distribution of the parent sample
zd_popsamp, ms_popsamp, arcsec2kpc_popsamp = mz_distribution.draw_mz(npop)

# assigns sizes to the parent sample sample
sigma_r = s19_sigma_r

mu_r_popsamp = hb09quad_mu_r_func(ms_popsamp)
r_popsamp = mu_r_popsamp + np.random.normal(0., sigma_r, npop)

# defines the subset of galaxies used for the fit of the fundamental plane
fpfitsamp = ms_popsamp > 11.
ms_fpfitsamp = ms_popsamp[fpfitsamp]
dr_fpfitsamp = r_popsamp[fpfitsamp] - hb09quad_mu_r_func(ms_fpfitsamp)

# draws scale-free values of m5, gamma and zs
m5_popscat = np.random.normal(0., 1., npop)
gamma_popscat = np.random.normal(0., 1., npop)
zs_popscat = np.random.normal(0., 1., npop)

# draws errors to be added to the model velocity dispersions
sigma_relerr_popsamp = np.random.normal(0., slacs_median_sigma_relerr, npop)

# prepares the lensing and dynamics grids
s2_grid_file = h5py.File('sigma2_grid.hdf5', 'r')
gamma_dyn_grid = s2_grid_file['gamma_grid'][()]
r_dyn_grid = s2_grid_file['lreff_grid'][()]
z_dyn_grid = s2_grid_file['z_grid'][()]

dyn_axes = {0: splrep(z_dyn_grid, np.arange(len(z_dyn_grid))), 1: splrep(r_dyn_grid, np.arange(len(r_dyn_grid))), 2: splrep(gamma_dyn_grid, np.arange(len(gamma_dyn_grid)))}
s2_interp = ndinterp.ndInterp(dyn_axes, s2_grid_file['s2_grid'][()])

rein_file = h5py.File('rein_grid.hdf5', 'r')
m5_rein_grid = rein_file['m5_grid'][()]
gamma_rein_grid = rein_file['gamma_grid'][()]
s_cr_rein_grid = rein_file['s_cr_grid'][()]

rein_axes = {0: splrep(s_cr_rein_grid, np.arange(len(s_cr_rein_grid))), 1: splrep(m5_rein_grid, np.arange(len(m5_rein_grid))), 2: splrep(gamma_rein_grid, np.arange(len(gamma_rein_grid)))}
rein_interp = ndinterp.ndInterp(rein_axes, rein_file['rein_grid'][()])

# prepares the comoving distance grids, to compute the estimated Einstein radius
# of parent sample galaxies
nzs = 101
zs_grid = np.linspace(zmin, zs_max, nzs)
comovd_grid = np.zeros(nzs)
dang_grid = np.zeros(nzs)
for i in range(nzs):
    comovd_grid[i] = sl_cosmology.comovd(zs_grid[i])
    dang_grid[i] = sl_cosmology.Dang(zs_grid[i])

dang_spline = splrep(zs_grid, dang_grid)
comovd_spline = splrep(zs_grid, comovd_grid)

def dds_arrfunc(zd, zs):
    comovdd = splev(zd, comovd_spline)
    comovds = splev(zs, comovd_spline)

    comovdds = comovds - comovdd
    return comovdds/(1. + zs)

def s_cr_arrfunc(zd, zs):
    ds = splev(zs, dang_spline)
    dd = splev(zd, dang_spline)
    dds = dds_arrfunc(zd, zs)
    return c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2

# prepares the lensing cross-section grid
cs_file = h5py.File('fibre_crosssect_grid.hdf5', 'r')
cs_grid = cs_file['mufibre3_cs_grid'][()]
cs_tein_grid = cs_file['tein_grid'][()]
cs_gamma_grid = cs_file['gamma_grid'][()]
cs_axes = {0: splrep(cs_tein_grid, np.arange(len(cs_tein_grid))), 1: splrep(cs_gamma_grid, np.arange(len(cs_gamma_grid)))}
cs_interp = ndinterp.ndInterp(cs_axes, cs_grid)

# prepares arrays for MC-integration of SLACS likelihood terms
ms_impsamp = np.zeros((nslacs, nis))

ms_terms = np.zeros((nslacs, nis))
r_terms = np.zeros((nslacs, nis))

# reads the lensing and dynamics grids of the SLACS lenses
lensgrid_file = h5py.File('slacs_lensing_grids.hdf5', 'r')
gamma_lens_grid = lensgrid_file['gamma_grid'][()]
ngamma = len(gamma_lens_grid)

jeans_file = h5py.File('slacs_jeans_grids.hdf5', 'r')
gamma_jeans_grid = jeans_file['gamma_grid'][()]

gamma_impsamp = np.random.normal(0., 1., (nslacs, nis))

m5_splines = []
dm5drein_splines = []
s2_splines = []
cs_lens_splines = []

for n in range(nslacs):
    m5_grid = lensgrid_file[slacs_names[n]]['m5_grid'][()]
    m5_splines.append(splrep(gamma_lens_grid, m5_grid))
    dm5drein_grid = lensgrid_file[slacs_names[n]]['dm5drein_grid'][()]
    dm5drein_splines.append(splrep(gamma_lens_grid, dm5drein_grid))
    s2_grid = jeans_file[slacs_names[n]]['s2_grid'][()]
    s2_spline = splrep(gamma_jeans_grid, s2_grid)
    s2_splines.append(s2_spline)
    cs_lens_grid = lensgrid_file[slacs_names[n]]['mufibre3_cs_grid'][()]
    cs_lens_spline = splrep(gamma_lens_grid, cs_lens_grid)
    cs_lens_splines.append(cs_lens_spline)

    ms_impsamp_here = np.random.normal(slacs_ms_obs[n], slacs_ms_err[n], nis)
    ms_impsamp[n, :] = ms_impsamp_here

    ms_terms[n, :] = mz_distribution.msdist(slacs_zd[n], ms_impsamp_here)
    r_terms[n, :] = 1./sigma_r * np.exp(-0.5*(slacs_r[n] - hb09quad_mu_r_func(ms_impsamp_here))**2/sigma_r**2)

mu_r_impsamp = hb09quad_mu_r_func(ms_impsamp)

rein_file.close()
cs_file.close()
s2_grid_file.close()
lensgrid_file.close()
jeans_file.close()

# defines the model parameters

mu_m5_par = {'name': 'mu_m5', 'lower': 9., 'upper': 12., 'guess': 11.33, 'spread': 0.01}
sigma_m5_par = {'name': 'sigma_m5', 'lower': 0., 'upper': 1., 'guess': 0.07, 'spread': 0.01}
beta_m5_par = {'name': 'beta_m5', 'lower': -3., 'upper': 3., 'guess': 0.62, 'spread': 0.01}
xi_m5_par = {'name': 'xi_m5', 'lower': -3., 'upper': 3., 'guess': -0.13, 'spread': 0.01}

mu_gamma_par = {'name': 'mu_gamma', 'lower': 1.8, 'upper': 2.2, 'guess': 2., 'spread': 0.01}
sigma_gamma_par = {'name': 'sigma_gamma', 'lower': 0., 'upper': 0.5, 'guess': 0.18, 'spread': 0.01}
beta_gamma_par = {'name': 'beta_gamma', 'lower': -3., 'upper': 3., 'guess': 0.31, 'spread': 0.01}
xi_gamma_par = {'name': 'xi_gamma', 'lower': -3., 'upper': 3., 'guess': -0.78, 'spread': 0.01}

mu_zs_par = {'name': 'mu_zs', 'lower': 0., 'upper': 2.5, 'guess': slacs_zs.mean(), 'spread': 0.01}
sigma_zs_par = {'name': 'sigma_zs', 'lower': 0., 'upper': 2., 'guess': slacs_zs.std(), 'spread': 0.01}

t_find_par = {'name': 't_find', 'lower': 0., 'upper': 3., 'guess': 0.77, 'spread': 0.01}
la_find_par = {'name': 'la_find', 'lower': -1., 'upper': 3., 'guess': 1.37, 'spread': 0.01}

pars = [mu_m5_par, sigma_m5_par, beta_m5_par, xi_m5_par, mu_gamma_par, sigma_gamma_par, beta_gamma_par, xi_gamma_par, mu_zs_par, sigma_zs_par, t_find_par, la_find_par]

npars = len(pars)

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

# defines the lens finding probability function
def pfind_func(tein_est, t, a):
    return 1./(1. + np.exp(-a*(tein_est - t)))

# log-posterior probability
def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300, np.zeros(4)

    mu_m5, sigma_m5, beta_m5, xi_m5, mu_gamma, sigma_gamma, beta_gamma, xi_gamma, mu_zs, sigma_zs, t_find, la_find = p

    mu_m5_pop = mu_m5 + beta_m5*(ms_popsamp - mpiv_slacs) + xi_m5*(r_popsamp - mu_r_popsamp)
    m5_popsamp = mu_m5_pop + sigma_m5 * m5_popscat

    mu_gamma_pop = mu_gamma + beta_gamma*(ms_popsamp - mpiv_slacs) + xi_gamma*(r_popsamp - mu_r_popsamp)
    scaled_gamma_min = (gamma_min - mu_gamma_pop)/sigma_gamma
    scaled_gamma_max = (gamma_max - mu_gamma_pop)/sigma_gamma

    gamma_popsamp = truncnorm.rvs(scaled_gamma_min, scaled_gamma_max, loc=mu_gamma_pop, scale=sigma_gamma, size=npop, random_state=0)

    zs_popsamp = truncnorm.rvs(-mu_zs/sigma_zs, np.inf, loc=mu_zs, scale=sigma_zs, size=npop, random_state=nslacs+2)

    bkg = zs_popsamp > zd_popsamp + 0.05

    s_cr_popsamp = s_cr_arrfunc(zd_popsamp, zs_popsamp)

    rein_point = np.array([s_cr_popsamp, m5_popsamp, gamma_popsamp]).T

    rein_popsamp = rein_interp.eval(rein_point)
    tein_popsamp = rein_popsamp / arcsec2kpc_popsamp

    xlarge_tein = tein_popsamp > cs_tein_grid[-1]

    cs_point = np.array((tein_popsamp, gamma_popsamp)).T
    cs_popsamp = cs_interp.eval(cs_point)

    nxl = xlarge_tein.sum()
    if nxl > 0:
        cs_popsamp[xlarge_tein] = 0.

    s2_point = np.array([zd_popsamp, r_popsamp, gamma_popsamp]).T

    s2_popmodel = 10.**m5_popsamp * s2_interp.eval(s2_point)
    sigma_popsamp = s2_popmodel**0.5
    sigma_obs_popsamp = sigma_popsamp * (1. + sigma_relerr_popsamp)

    sigma_fpfitsamp = sigma_popsamp[fpfitsamp]

    # fits for the fundamental plane 
    def fitfunc(p):
        return p[0] + p[1]*(ms_fpfitsamp - mpiv_slacs) + p[2]*dr_fpfitsamp

    def errfunc(p):
        return fitfunc(p) - np.log10(sigma_fpfitsamp)

    p0 = (2.38, 0.3, -0.1)
    pfit = leastsq(errfunc, p0)

    fpfit_scat = errfunc(pfit[0]).std()

    dds_popsamp = dds_arrfunc(zd_popsamp, zs_popsamp)
    ds_popsamp = splev(zs_popsamp, dang_spline)

    drat_popsamp = dds_popsamp/ds_popsamp

    tein_est_popsamp = np.rad2deg(4.*np.pi * (sigma_obs_popsamp/3e5)**2 * drat_popsamp) * 3600.

    good = bkg & (s_cr_popsamp < s_cr_rein_grid[-1]) & (zs_popsamp < zs_max) & (zs_popsamp > zmin) & (m5_popsamp > m5_rein_grid[0]) & (m5_popsamp < m5_rein_grid[-1]) & (gamma_popsamp > gamma_min) & (gamma_popsamp < gamma_max) & (r_popsamp > r_dyn_grid[0]) & (r_popsamp < r_dyn_grid[-1])

    popint = cs_popsamp * pfind_func(tein_est_popsamp, t_find, 10.**la_find)

    pop_norm = popint[good].sum()

    if good.sum() < 0.2 * npop:
        return -1e300, np.zeros(4)

    # adds the fundamental plane prior: intrinsic scatter and mass-veldisp relation
    sumlogp = -0.5*(fpfit_scat - fiducial_fpscat)**2/err_fpscat**2
    sumlogp += -0.5*(pfit[0][0] - mu_v_prior)**2/err_mu_v**2
    sumlogp += -0.5*(pfit[0][1] - beta_v_prior)**2/err_beta_v**2

    for n in range(nslacs):

        zs_term = 1./sigma_zs * np.exp(-0.5*(slacs_zs[n] - mu_zs)**2/sigma_zs**2)

        mu_m5_here = mu_m5 + beta_m5*(ms_impsamp[n, :] - mpiv_slacs) + xi_m5*(slacs_r[n] - mu_r_impsamp[n, :])
        mu_gamma_here = mu_gamma + beta_gamma*(ms_impsamp[n, :] - mpiv_slacs) + xi_gamma*(slacs_r[n] - mu_r_impsamp[n, :])

        scaled_gamma_min_here = (gamma_min - mu_gamma_here)/sigma_gamma
        scaled_gamma_max_here = (gamma_max - mu_gamma_here)/sigma_gamma

        gamma_impsamp_here = truncnorm.rvs(scaled_gamma_min_here, scaled_gamma_max_here, loc=mu_gamma_here, scale=sigma_gamma, size=nis, random_state=n+1)
        m5_impsamp_here = splev(gamma_impsamp_here, m5_splines[n])
        dm5drein_impsamp_here = splev(gamma_impsamp_here, dm5drein_splines[n])

        m5_term = 1./sigma_m5 * np.exp(-0.5*(m5_impsamp_here - mu_m5_here)**2/sigma_m5**2)

        sigma_model = (10.**m5_impsamp_here * splev(gamma_impsamp_here, s2_splines[n]))**0.5
        sigma_like = 1./slacs_sigma_err[n] * np.exp(-0.5*(sigma_model - slacs_sigma_obs[n])**2/slacs_sigma_err[n]**2)

        find_term = pfind_func(slacs_tein_est[n], t_find, 10.**la_find)

        cs_impsamp_here = splev(gamma_impsamp_here, cs_lens_splines[n])

        integrand = sigma_like * ms_terms[n] * r_terms[n] * zs_term * m5_term * dm5drein_impsamp_here * find_term * cs_impsamp_here / pop_norm
    
        sumlogp += np.log(integrand.sum())

    if sumlogp != sumlogp:
        return -1e300, (pfit[0][0], pfit[0][1], pfit[0][2], fpfit_scat)

    return sumlogp, (pfit[0][0], pfit[0][1], pfit[0][2], fpfit_scat)

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

output_file = h5py.File('full_inference.hdf5', 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])
    print('%s %3.2f'%(pars[n]['name'], np.median(sampler.chain[:, -1, n].flatten())))
output_file.create_dataset('fpfit', data=blobchain.T)


