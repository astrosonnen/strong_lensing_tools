import numpy as np
import h5py
import ndinterp
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import leastsq
from fitpars import *
from parent_sample_pars import *
from read_slacs import *
import mz_distribution
import sl_cosmology
from sl_cosmology import Dang, Mpc, M_Sun, c, G, kpc
from sl_profiles import powerlaw
from scipy.stats import truncnorm
from scipy.special import erf
import sys


fitname = 'full'

npp = 1000 # number of draws from the posterior
npop = 100000 # number of galaxies from the parent population
nsub = 100 # number of parent population galaxies that are stored in the output file
nlens = nslacs

# defines bins in velocity dispersion
nsigmabins = 25
sigma_bins = np.linspace(145., 405., nsigmabins+1)

# defines bins in stellar mass
nmsbins = 19
ms_bins = np.linspace(10.15, 12.05, nmsbins+1)

# draws samples in redshift and stellar mass from the parent population
zd_popsamp, ms_popsamp, arcsec2kpc_popsamp = mz_distribution.draw_mz(npop)

# assigns sizes to parent population galaxies
sigma_r = s19_sigma_r
mu_r_popsamp = hb09quad_mu_r_func(ms_popsamp)
r_popsamp = mu_r_popsamp + np.random.normal(0., sigma_r, npop)

# defines a subset of galaxies for which to fit the fundamental plane relation
fpfitsamp = ms_popsamp > 11.
ms_fpfitsamp = ms_popsamp[fpfitsamp]
r_fpfitsamp = r_popsamp[fpfitsamp]

# draws scale-free values of m5, gamma and zs
m5_popscat = np.random.normal(0., 1., npop)
gamma_popscat = np.random.normal(0., 1., npop)

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

ngfit = 161
gamma_fit_grid = np.linspace(1.2, 2.8, ngfit)

# prepares comoving distance grids, to compute Einstein radii
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

rein_file.close()
cs_file.close()
s2_grid_file.close()

# defines the lens finding probability
def pfind_func(tein_est, t, a):
    return 1./(1. + np.exp(-a*(tein_est - t)))

chain = h5py.File('%s_inference.hdf5'%fitname, 'r')
nwalkers, nstep = chain['logp'][()].shape

mu_m5_pp = np.zeros(npp)
sigma_m5_pp = np.zeros(npp)
beta_m5_pp = np.zeros(npp)
xi_m5_pp = np.zeros(npp)
mu_gamma_pp = np.zeros(npp)
sigma_gamma_pp = np.zeros(npp)
beta_gamma_pp = np.zeros(npp)
xi_gamma_pp = np.zeros(npp)
mu_zs_pp = np.zeros(npp)
sigma_zs_pp = np.zeros(npp)
t_find_pp = np.zeros(npp)
la_find_pp = np.zeros(npp)

tein_pp = np.zeros((npp, nlens))
rein_pp = np.zeros((npp, nlens))
zd_pp = np.zeros((npp, nlens))
zs_pp = np.zeros((npp, nlens))
ms_pp = np.zeros((npp, nlens))
m5_pp = np.zeros((npp, nlens))
gamma_pp = np.zeros((npp, nlens))
gamma_obs_pp = np.zeros((npp, nlens))
r_pp = np.zeros((npp, nlens))
sigma_pp = np.zeros((npp, nlens))
sigma_obs_pp = np.zeros((npp, nlens))
s_cr_pp = np.zeros((npp, nlens))
tein_est_pp = np.zeros((npp, nlens))
tein_sis_pp = np.zeros((npp, nlens))

sub_zd_pp = np.zeros((npp, nsub))
sub_ms_pp = np.zeros((npp, nsub))
sub_m5_pp = np.zeros((npp, nsub))
sub_gamma_pp = np.zeros((npp, nsub))
sub_r_pp = np.zeros((npp, nsub))
sub_sigma_pp = np.zeros((npp, nsub))
sub_sigma_obs_pp = np.zeros((npp, nsub))

gfit_scat = np.zeros(npp)
gfit_mu_gamma = np.zeros(npp)
gfit_beta_gamma = np.zeros(npp)
gfit_xi_gamma = np.zeros(npp)

fpfit_scat = np.zeros(npp)
fpfit_mu = np.zeros(npp)
fpfit_beta = np.zeros(npp)
fpfit_xi = np.zeros(npp)

zd_pop_sbin_pp = np.zeros((npp, nsigmabins))
zs_pop_sbin_pp = np.zeros((npp, nsigmabins))
ms_pop_sbin_pp = np.zeros((npp, nsigmabins))
m5_pop_sbin_pp = np.zeros((npp, nsigmabins))
gamma_pop_sbin_pp = np.zeros((npp, nsigmabins))
r_pop_sbin_pp = np.zeros((npp, nsigmabins))
sigma_pop_sbin_pp = np.zeros((npp, nsigmabins))
sigma_obs_pop_sbin_pp = np.zeros((npp, nsigmabins))

zd_lens_sbin_pp = np.zeros((npp, nsigmabins))
zs_lens_sbin_pp = np.zeros((npp, nsigmabins))
ms_lens_sbin_pp = np.zeros((npp, nsigmabins))
m5_lens_sbin_pp = np.zeros((npp, nsigmabins))
gamma_lens_sbin_pp = np.zeros((npp, nsigmabins))
r_lens_sbin_pp = np.zeros((npp, nsigmabins))
sigma_lens_sbin_pp = np.zeros((npp, nsigmabins))
sigma_obs_lens_sbin_pp = np.zeros((npp, nsigmabins))

zd_pop_sobs_pp = np.zeros((npp, nsigmabins))
zs_pop_sobs_pp = np.zeros((npp, nsigmabins))
ms_pop_sobs_pp = np.zeros((npp, nsigmabins))
m5_pop_sobs_pp = np.zeros((npp, nsigmabins))
gamma_pop_sobs_pp = np.zeros((npp, nsigmabins))
r_pop_sobs_pp = np.zeros((npp, nsigmabins))
sigma_pop_sobs_pp = np.zeros((npp, nsigmabins))
sigma_obs_pop_sobs_pp = np.zeros((npp, nsigmabins))

zd_lens_sobs_pp = np.zeros((npp, nsigmabins))
zs_lens_sobs_pp = np.zeros((npp, nsigmabins))
ms_lens_sobs_pp = np.zeros((npp, nsigmabins))
m5_lens_sobs_pp = np.zeros((npp, nsigmabins))
gamma_lens_sobs_pp = np.zeros((npp, nsigmabins))
r_lens_sobs_pp = np.zeros((npp, nsigmabins))
sigma_lens_sobs_pp = np.zeros((npp, nsigmabins))
sigma_obs_lens_sobs_pp = np.zeros((npp, nsigmabins))

zd_pop_mbin_pp = np.zeros((npp, nmsbins))
zs_pop_mbin_pp = np.zeros((npp, nmsbins))
ms_pop_mbin_pp = np.zeros((npp, nmsbins))
m5_pop_mbin_pp = np.zeros((npp, nmsbins))
gamma_pop_mbin_pp = np.zeros((npp, nmsbins))
r_pop_mbin_pp = np.zeros((npp, nmsbins))
sigma_pop_mbin_pp = np.zeros((npp, nmsbins))
sigma_obs_pop_mbin_pp = np.zeros((npp, nmsbins))

zd_lens_mbin_pp = np.zeros((npp, nmsbins))
zs_lens_mbin_pp = np.zeros((npp, nmsbins))
ms_lens_mbin_pp = np.zeros((npp, nmsbins))
m5_lens_mbin_pp = np.zeros((npp, nmsbins))
gamma_lens_mbin_pp = np.zeros((npp, nmsbins))
r_lens_mbin_pp = np.zeros((npp, nmsbins))
sigma_lens_mbin_pp = np.zeros((npp, nmsbins))
sigma_obs_lens_mbin_pp = np.zeros((npp, nmsbins))

all_indices = np.arange(npop)

output_file = h5py.File('%s_nopfind_pp.hdf5'%fitname, 'w')

output_file.attrs['npp'] = npp
output_file.attrs['npop'] = npop

output_file.create_dataset('ms_bins', data=ms_bins)
output_file.create_dataset('sigma_bins', data=sigma_bins)

for i in range(npp):

    print(i)

    ind1 = i%nwalkers
    ind2 = nstep-i//nwalkers-1

    mu_m5 = chain['mu_m5'][ind1, ind2]
    sigma_m5 = chain['sigma_m5'][ind1, ind2]
    beta_m5 = chain['beta_m5'][ind1, ind2]
    xi_m5 = chain['xi_m5'][ind1, ind2]
    mu_gamma = chain['mu_gamma'][ind1, ind2]
    sigma_gamma = chain['sigma_gamma'][ind1, ind2]
    beta_gamma = chain['beta_gamma'][ind1, ind2]
    xi_gamma = chain['xi_gamma'][ind1, ind2]
    mu_zs = chain['mu_zs'][ind1, ind2]
    sigma_zs = chain['sigma_zs'][ind1, ind2]
    t_find = chain['t_find'][ind1, ind2]
    la_find = chain['la_find'][ind1, ind2]

    mu_m5_pp[i] = mu_m5
    sigma_m5_pp[i] = sigma_m5
    beta_m5_pp[i] = beta_m5
    xi_m5_pp[i] = xi_m5
    mu_gamma_pp[i] = mu_gamma
    sigma_gamma_pp[i] = sigma_gamma
    beta_gamma_pp[i] = beta_gamma
    xi_gamma_pp[i] = xi_gamma
    mu_zs_pp[i] = mu_zs
    sigma_zs_pp[i] = sigma_zs
    t_find_pp[i] = t_find
    la_find_pp[i] = la_find

    mu_m5_pop = mu_m5 + beta_m5*(ms_popsamp - mpiv_slacs) + xi_m5*(r_popsamp - mu_r_popsamp)
    m5_popsamp = mu_m5_pop + sigma_m5 * m5_popscat

    mu_gamma_pop = mu_gamma + beta_gamma*(ms_popsamp - mpiv_slacs) + xi_gamma*(r_popsamp - mu_r_popsamp)
    scaled_gamma_min = (gamma_min - mu_gamma_pop)/sigma_gamma
    scaled_gamma_max = (gamma_max - mu_gamma_pop)/sigma_gamma

    gamma_popsamp = truncnorm.rvs(scaled_gamma_min, scaled_gamma_max, loc=mu_gamma_pop, scale=sigma_gamma, size=npop, random_state=0)

    zs_popsamp = truncnorm.rvs(-mu_zs/sigma_zs, np.inf, loc=mu_zs, scale=sigma_zs, size=npop, random_state=1)

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

    sub_indices = np.random.choice(all_indices, size=nsub)

    sub_zd_pp[i, :] = zd_popsamp[sub_indices]
    sub_ms_pp[i, :] = ms_popsamp[sub_indices]
    sub_m5_pp[i, :] = m5_popsamp[sub_indices]
    sub_gamma_pp[i, :] = gamma_popsamp[sub_indices]
    sub_r_pp[i, :] = r_popsamp[sub_indices]
    sub_sigma_pp[i, :] = sigma_popsamp[sub_indices]
    sub_sigma_obs_pp[i, :] = sigma_obs_popsamp[sub_indices]

    sigma_fpfitsamp = sigma_popsamp[fpfitsamp]

    # fits for the fundamental plane
    def fitfunc(p):
        return p[0] + p[1]*(ms_fpfitsamp - mpiv_slacs) + p[2]*(r_fpfitsamp - hb09quad_mu_r_func(ms_fpfitsamp))

    def errfunc(p):
        return fitfunc(p) - np.log10(sigma_fpfitsamp)

    p0 = (2.3, 0.2, -0.4)
    pfit = leastsq(errfunc, p0)

    fpfit_mu[i] = pfit[0][0]
    fpfit_beta[i] = pfit[0][1]
    fpfit_xi[i] = pfit[0][2]

    fpfit_scat[i] = errfunc(pfit[0]).std()

    dds_popsamp = dds_arrfunc(zd_popsamp, zs_popsamp)
    ds_popsamp = splev(zs_popsamp, dang_spline)

    drat_popsamp = dds_popsamp/ds_popsamp

    tein_est_popsamp = np.rad2deg(4.*np.pi * (sigma_obs_popsamp/3e5)**2 * drat_popsamp) * 3600.
    tein_sis_popsamp = np.rad2deg(4.*np.pi * (sigma_popsamp/3e5)**2 * drat_popsamp) * 3600.

    good = bkg & (s_cr_popsamp < s_cr_rein_grid[-1]) & (zs_popsamp < zs_max) & (zs_popsamp > zmin) & (m5_popsamp > m5_rein_grid[0]) & (m5_popsamp < m5_rein_grid[-1]) & (gamma_popsamp > gamma_min) & (gamma_popsamp < gamma_max) & (r_popsamp > r_dyn_grid[0]) & (r_popsamp < r_dyn_grid[-1])

    popprob = cs_popsamp 
    popprob[popprob<0.] = 0.

    popprob /= popprob.sum()

    lens_indices = np.random.choice(all_indices, size=nlens, p=popprob)

    zd_pp[i, :] = zd_popsamp[lens_indices]
    zs_pp[i, :] = zs_popsamp[lens_indices]
    ms_pp[i, :] = ms_popsamp[lens_indices]
    m5_pp[i, :] = m5_popsamp[lens_indices]
    gamma_pp[i, :] = gamma_popsamp[lens_indices]
    r_pp[i, :] = r_popsamp[lens_indices]
    tein_pp[i, :] = tein_popsamp[lens_indices]
    rein_pp[i, :] = rein_popsamp[lens_indices]
    sigma_pp[i, :] = sigma_popsamp[lens_indices]
    sigma_obs_pp[i, :] = sigma_obs_popsamp[lens_indices]
    s_cr_pp[i, :] = s_cr_popsamp[lens_indices]
    tein_est_pp[i, :] = tein_est_popsamp[lens_indices]
    tein_sis_pp[i, :] = tein_sis_popsamp[lens_indices]

    # fits for the observed gamma
    for j in range(nlens):
        s2_point = np.array([zd_pp[i, j]*np.ones(ngfit), r_pp[i, j]*np.ones(ngfit), gamma_fit_grid]).T
        s2_model = np.pi * s_cr_pp[i, j] * rein_pp[i, j]**2 / powerlaw.M2d(rein_pp[i, j], gamma_fit_grid) * powerlaw.M2d(5., gamma_fit_grid) * s2_interp.eval(s2_point)

        sigma_model = s2_model**0.5
        gamma_obs_pp[i, j] = gamma_fit_grid[abs(sigma_model - sigma_obs_pp[i, j]).argmin()]

    # fits for the gamma distribution of the lenses
    ms_lenses = ms_popsamp[lens_indices]
    r_lenses = r_popsamp[lens_indices]
    gamma_lenses = gamma_popsamp[lens_indices]
    def fitfunc(p):
        return p[0] + p[1]*(ms_lenses - mpiv_slacs) + p[2]*(r_lenses - hb09quad_mu_r_func(ms_lenses))

    def errfunc(p):
        return fitfunc(p) - gamma_lenses

    p0 = (2., 0., -0.5)
    pfit = leastsq(errfunc, p0)

    gfit_mu_gamma[i] = pfit[0][0]
    gfit_beta_gamma[i] = pfit[0][1]
    gfit_xi_gamma[i] = pfit[0][2]

    gfit_scat[i] = errfunc(pfit[0]).std()

    for n in range(nsigmabins):
        sigma_bin = (sigma_popsamp >= sigma_bins[n]) & (sigma_popsamp < sigma_bins[n+1])

        zd_pop_sbin_pp[i, n] = zd_popsamp[sigma_bin].mean()
        zs_pop_sbin_pp[i, n] = zs_popsamp[sigma_bin].mean()
        ms_pop_sbin_pp[i, n] = ms_popsamp[sigma_bin].mean()
        m5_pop_sbin_pp[i, n] = m5_popsamp[sigma_bin].mean()
        gamma_pop_sbin_pp[i, n] = gamma_popsamp[sigma_bin].mean()
        r_pop_sbin_pp[i, n] = r_popsamp[sigma_bin].mean()
        sigma_pop_sbin_pp[i, n] = sigma_popsamp[sigma_bin].mean()
        sigma_obs_pop_sbin_pp[i, n] = sigma_obs_popsamp[sigma_bin].mean()
        
        pbsum = popprob[sigma_bin].sum()
        zd_lens_sbin_pp[i, n] = (zd_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        zs_lens_sbin_pp[i, n] = (zs_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        ms_lens_sbin_pp[i, n] = (ms_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        m5_lens_sbin_pp[i, n] = (m5_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        gamma_lens_sbin_pp[i, n] = (gamma_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        r_lens_sbin_pp[i, n] = (r_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        sigma_lens_sbin_pp[i, n] = (sigma_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum
        sigma_obs_lens_sbin_pp[i, n] = (sigma_obs_popsamp[sigma_bin]*popprob[sigma_bin]).sum()/pbsum

        sobs_bin = (sigma_obs_popsamp >= sigma_bins[n]) & (sigma_obs_popsamp < sigma_bins[n+1])

        zd_pop_sobs_pp[i, n] = zd_popsamp[sobs_bin].mean()
        zs_pop_sobs_pp[i, n] = zs_popsamp[sobs_bin].mean()
        ms_pop_sobs_pp[i, n] = ms_popsamp[sobs_bin].mean()
        m5_pop_sobs_pp[i, n] = m5_popsamp[sobs_bin].mean()
        gamma_pop_sobs_pp[i, n] = gamma_popsamp[sobs_bin].mean()
        r_pop_sobs_pp[i, n] = r_popsamp[sobs_bin].mean()
        sigma_pop_sobs_pp[i, n] = sigma_popsamp[sobs_bin].mean()
        sigma_obs_pop_sobs_pp[i, n] = sigma_obs_popsamp[sobs_bin].mean()
        
        pbsum = popprob[sobs_bin].sum()
        zd_lens_sobs_pp[i, n] = (zd_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        zs_lens_sobs_pp[i, n] = (zs_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        ms_lens_sobs_pp[i, n] = (ms_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        m5_lens_sobs_pp[i, n] = (m5_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        gamma_lens_sobs_pp[i, n] = (gamma_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        r_lens_sobs_pp[i, n] = (r_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        sigma_lens_sobs_pp[i, n] = (sigma_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum
        sigma_obs_lens_sobs_pp[i, n] = (sigma_obs_popsamp[sobs_bin]*popprob[sobs_bin]).sum()/pbsum

    for n in range(nmsbins):
        ms_bin = (ms_popsamp >= ms_bins[n]) & (ms_popsamp < ms_bins[n+1])

        zd_pop_mbin_pp[i, n] = zd_popsamp[ms_bin].mean()
        zs_pop_mbin_pp[i, n] = zs_popsamp[ms_bin].mean()
        ms_pop_mbin_pp[i, n] = ms_popsamp[ms_bin].mean()
        m5_pop_mbin_pp[i, n] = m5_popsamp[ms_bin].mean()
        gamma_pop_mbin_pp[i, n] = gamma_popsamp[ms_bin].mean()
        r_pop_mbin_pp[i, n] = r_popsamp[ms_bin].mean()
        sigma_pop_mbin_pp[i, n] = sigma_popsamp[ms_bin].mean()
        sigma_obs_pop_mbin_pp[i, n] = sigma_obs_popsamp[ms_bin].mean()
 
        pbsum = popprob[ms_bin].sum()
        zd_lens_mbin_pp[i, n] = (zd_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        zs_lens_mbin_pp[i, n] = (zs_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        ms_lens_mbin_pp[i, n] = (ms_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        m5_lens_mbin_pp[i, n] = (m5_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        gamma_lens_mbin_pp[i, n] = (gamma_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        r_lens_mbin_pp[i, n] = (r_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        sigma_lens_mbin_pp[i, n] = (sigma_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum
        sigma_obs_lens_mbin_pp[i, n] = (sigma_obs_popsamp[ms_bin]*popprob[ms_bin]).sum()/pbsum

# stores the values of the hyper-parameters corresponding to each draw
hpgroup = output_file.create_group('hyperpars')

hpgroup.create_dataset('mu_m5', data=mu_m5_pp)
hpgroup.create_dataset('sigma_m5', data=sigma_m5_pp)
hpgroup.create_dataset('beta_m5', data=beta_m5_pp)
hpgroup.create_dataset('xi_m5', data=xi_m5_pp)
hpgroup.create_dataset('mu_gamma', data=mu_gamma_pp)
hpgroup.create_dataset('sigma_gamma', data=sigma_gamma_pp)
hpgroup.create_dataset('beta_gamma', data=beta_gamma_pp)
hpgroup.create_dataset('xi_gamma', data=xi_gamma_pp)
hpgroup.create_dataset('t_find', data=t_find_pp)
hpgroup.create_dataset('la_find', data=la_find_pp)

# fundamental plane fit parameters
hpgroup.create_dataset('fpfit_scat', data=fpfit_scat)
hpgroup.create_dataset('fpfit_mu', data=fpfit_mu)
hpgroup.create_dataset('fpfit_beta', data=fpfit_beta)
hpgroup.create_dataset('fpfit_xi', data=fpfit_xi)

# galaxy parameters (redshift, stellar mass, m5, gamma
# log(Reff), veldisp (true and observed)) for a subset 
# of nsub galaxies drawn from the parent population.
subgroup = output_file.create_group('subset')

subgroup.create_dataset('zd', data=sub_zd_pp)
subgroup.create_dataset('ms', data=sub_ms_pp)
subgroup.create_dataset('m5', data=sub_m5_pp)
subgroup.create_dataset('m5', data=sub_gamma_pp)
subgroup.create_dataset('r', data=sub_r_pp)
subgroup.create_dataset('sigma', data=sub_sigma_pp)
subgroup.create_dataset('sigma_obs', data=sub_sigma_obs_pp)

# parameters for a set of 59 lenses
lensgroup = output_file.create_group('lenses')

lensgroup.create_dataset('zd', data=zd_pp)
lensgroup.create_dataset('zs', data=zs_pp)
lensgroup.create_dataset('ms', data=ms_pp)
lensgroup.create_dataset('m5', data=m5_pp)
lensgroup.create_dataset('gamma', data=gamma_pp)
lensgroup.create_dataset('gamma_obs', data=gamma_obs_pp)
lensgroup.create_dataset('r', data=r_pp)
lensgroup.create_dataset('tein', data=tein_pp)
lensgroup.create_dataset('rein', data=rein_pp)
lensgroup.create_dataset('sigma', data=sigma_pp)
lensgroup.create_dataset('sigma_obs', data=sigma_obs_pp)
lensgroup.create_dataset('s_cr', data=s_cr_pp)
lensgroup.create_dataset('tein_est', data=tein_est_pp)
lensgroup.create_dataset('tein_sis', data=tein_sis_pp)
lensgroup.create_dataset('gfit_scat', data=gfit_scat)
lensgroup.create_dataset('gfit_mu_gamma', data=gfit_mu_gamma)
lensgroup.create_dataset('gfit_beta_gamma', data=gfit_beta_gamma)
lensgroup.create_dataset('gfit_xi_gamma', data=gfit_xi_gamma)

# population-averaged galaxy properties in bins of velocity dispersion
popsbingroup = output_file.create_group('pop_sigma_bin')

popsbingroup.create_dataset('zd', data=zd_pop_sbin_pp)
popsbingroup.create_dataset('zs', data=zs_pop_sbin_pp)
popsbingroup.create_dataset('ms', data=ms_pop_sbin_pp)
popsbingroup.create_dataset('m5', data=m5_pop_sbin_pp)
popsbingroup.create_dataset('gamma', data=gamma_pop_sbin_pp)
popsbingroup.create_dataset('r', data=r_pop_sbin_pp)
popsbingroup.create_dataset('sigma', data=sigma_pop_sbin_pp)
popsbingroup.create_dataset('sigma_obs', data=sigma_obs_pop_sbin_pp)

# population-averaged lens properties in bins of velocity dispersion
lenssbingroup = output_file.create_group('lens_sigma_bin')

lenssbingroup.create_dataset('zd', data=zd_lens_sbin_pp)
lenssbingroup.create_dataset('zs', data=zs_lens_sbin_pp)
lenssbingroup.create_dataset('ms', data=ms_lens_sbin_pp)
lenssbingroup.create_dataset('m5', data=m5_lens_sbin_pp)
lenssbingroup.create_dataset('gamma', data=gamma_lens_sbin_pp)
lenssbingroup.create_dataset('r', data=r_lens_sbin_pp)
lenssbingroup.create_dataset('sigma', data=sigma_lens_sbin_pp)
lenssbingroup.create_dataset('sigma_obs', data=sigma_obs_lens_sbin_pp)

# population-averaged galaxy properties in bins of observed velocity dispersion
popsobsgroup = output_file.create_group('pop_sigma_obs_bin')

popsobsgroup.create_dataset('zd', data=zd_pop_sobs_pp)
popsobsgroup.create_dataset('zs', data=zs_pop_sobs_pp)
popsobsgroup.create_dataset('ms', data=ms_pop_sobs_pp)
popsobsgroup.create_dataset('m5', data=m5_pop_sobs_pp)
popsobsgroup.create_dataset('gamma', data=gamma_pop_sobs_pp)
popsobsgroup.create_dataset('r', data=r_pop_sobs_pp)
popsobsgroup.create_dataset('sigma', data=sigma_pop_sobs_pp)
popsobsgroup.create_dataset('sigma_obs', data=sigma_obs_pop_sobs_pp)

# population-averaged lens properties in bins of observed velocity dispersion
lenssobsgroup = output_file.create_group('lens_sigma_obs_bin')

lenssobsgroup.create_dataset('zd', data=zd_lens_sobs_pp)
lenssobsgroup.create_dataset('zs', data=zs_lens_sobs_pp)
lenssobsgroup.create_dataset('ms', data=ms_lens_sobs_pp)
lenssobsgroup.create_dataset('m5', data=m5_lens_sobs_pp)
lenssobsgroup.create_dataset('gamma', data=gamma_lens_sobs_pp)
lenssobsgroup.create_dataset('r', data=r_lens_sobs_pp)
lenssobsgroup.create_dataset('sigma', data=sigma_lens_sobs_pp)
lenssobsgroup.create_dataset('sigma_obs', data=sigma_obs_lens_sobs_pp)

# population-averaged galaxy properties in bins of stellar mass
popmbingroup = output_file.create_group('pop_ms_bin')

popmbingroup.create_dataset('zd', data=zd_pop_mbin_pp)
popmbingroup.create_dataset('zs', data=zs_pop_mbin_pp)
popmbingroup.create_dataset('ms', data=ms_pop_mbin_pp)
popmbingroup.create_dataset('m5', data=m5_pop_mbin_pp)
popmbingroup.create_dataset('gamma', data=gamma_pop_mbin_pp)
popmbingroup.create_dataset('r', data=r_pop_mbin_pp)
popmbingroup.create_dataset('sigma', data=sigma_pop_mbin_pp)
popmbingroup.create_dataset('sigma_obs', data=sigma_obs_pop_mbin_pp)

# population-averaged lens properties in bins of stellar mass
lensmbingroup = output_file.create_group('lens_ms_bin')

lensmbingroup.create_dataset('zd', data=zd_lens_mbin_pp)
lensmbingroup.create_dataset('zs', data=zs_lens_mbin_pp)
lensmbingroup.create_dataset('ms', data=ms_lens_mbin_pp)
lensmbingroup.create_dataset('m5', data=m5_lens_mbin_pp)
lensmbingroup.create_dataset('gamma', data=gamma_lens_mbin_pp)
lensmbingroup.create_dataset('r', data=r_lens_mbin_pp)
lensmbingroup.create_dataset('sigma', data=sigma_lens_mbin_pp)
lensmbingroup.create_dataset('sigma_obs', data=sigma_obs_lens_mbin_pp)

