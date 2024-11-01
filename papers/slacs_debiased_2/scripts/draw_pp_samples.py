import numpy as np
import h5py
from read_slacs import *
from masssize import *
from gnfw_lensingfuncs import *
from scipy.optimize import leastsq
import ndinterp


chain = h5py.File('adcontr_teinerr_csinterp_inference.hdf5', 'r')

nwalkers, nstep = chain['lasps'][()].shape

npp = 1000
nsub = 100 # number of random subsamples from the parent population

# least squares fit
def do_leastsq(sstar, gammapl, piv=9.):
    x = sstar - piv
    y = gammapl
    n = len(sstar)
    m = (n * (x*y).sum() - (y.sum())*(x.sum()))/(n * (x**2).sum() - (x.sum())**2)
    b = (y.sum() - m*x.sum())/n
    scat = (y - b - m*x).std()
    return m, b, scat

# loads the individual cross-section grids
pop_file = h5py.File('npop1e+04_teincs_grids.hdf5', 'r')
seedno = pop_file.attrs['seedno']
npop = pop_file.attrs['npop']

mh_scat = pop_file['mh_scat'][()]
ms_pop = pop_file['ms'][()]
ms_obs_pop = ms_pop + np.random.normal(0., slacs_median_ms_err, npop)
zd_pop = pop_file['zd'][()]
zs_pop = pop_file['zs'][()]
re_pop = pop_file['re'][()]
drat_pop = pop_file['drat'][()]
rhoc_pop = pop_file['rhoc'][()]
s_cr_pop = pop_file['s_cr'][()]
bkg_pop = pop_file['bkg'][()]
arcsec2kpc_pop = pop_file['arcsec2kpc'][()]
lsigma_scat_pop = pop_file['lsigma_scat']
sigma_relerr_pop = pop_file['sigma_relerr'][()]

mu_r_pop = hb09quad_mu_r_func(ms_pop)

eps_pop_grid = pop_file['eps_grid'][()]
lasps_pop_grid = pop_file['lasps_grid'][()]

cs_pop_interps = []
tein_pop_interps = []
rs_pop_interps = []
gamma_pop_interps = []
for n in range(npop):
    group = pop_file['%05d'%n]
    mh_pop_grid = group['mh_grid'][()]
    cs_pop_grid = group['cs_grid'][()]
    tein_pop_grid = group['tein_grid'][()]
    rs_pop_grid = group['rs_grid'][()]
    gamma_pop_grid = group['gamma_grid'][()]

    axes = {0: splrep(eps_pop_grid, np.arange(len(eps_pop_grid))), 1: splrep(mh_pop_grid, np.arange(len(mh_pop_grid))), 2: splrep(lasps_pop_grid, np.arange(len(lasps_pop_grid)))}
    cs_pop_interp = ndinterp.ndInterp(axes, cs_pop_grid, order=1)
    cs_pop_interps.append(cs_pop_interp)

    tein_pop_interp = ndinterp.ndInterp(axes, tein_pop_grid, order=1)
    tein_pop_interps.append(tein_pop_interp)

    rs_pop_interp = ndinterp.ndInterp(axes, rs_pop_grid, order=1)
    rs_pop_interps.append(rs_pop_interp)

    gamma_pop_interp = ndinterp.ndInterp(axes, gamma_pop_grid, order=1)
    gamma_pop_interps.append(gamma_pop_interp)

dx = 0.001

lasps_pp = np.zeros(npp)
eps_pp = np.zeros(npp)
mu_mh_pp = np.zeros(npp)
beta_mh_pp = np.zeros(npp)
sigma_mh_pp = np.zeros(npp)
mu_sigma_pp = np.zeros(npp)
beta_sigma_pp = np.zeros(npp)
xi_sigma_pp = np.zeros(npp)
nu_sigma_pp = np.zeros(npp)
sigma_sigma_pp = np.zeros(npp)

lens_mu_sigma_pp = np.zeros(npp)
lens_beta_sigma_pp = np.zeros(npp)
lens_xi_sigma_pp = np.zeros(npp)
lens_nu_sigma_pp = np.zeros(npp)
lens_sigma_sigma_pp = np.zeros(npp)

lens_mu_sobs_pp = np.zeros(npp)
lens_beta_sobs_pp = np.zeros(npp)
lens_xi_sobs_pp = np.zeros(npp)
lens_nu_sobs_pp = np.zeros(npp)
lens_sigma_sobs_pp = np.zeros(npp)

amy_gpl_err = 0.5*(amy_gpl_uperr + amy_gpl_dwerr)
shajib_gpl_err = 0.5*(shajib_gpl_uperr + shajib_gpl_dwerr)

amy_mu_gammapl_pp = np.zeros(npp)
amy_beta_gammapl_pp = np.zeros(npp)
amy_sigma_gammapl_pp = np.zeros(npp)

shajib_mu_gammapl_pp = np.zeros(npp)
shajib_beta_gammapl_pp = np.zeros(npp)
shajib_sigma_gammapl_pp = np.zeros(npp)

tan_mu_gammapl_pp = np.zeros(npp)
tan_beta_gammapl_pp = np.zeros(npp)
tan_sigma_gammapl_pp = np.zeros(npp)

tein_lens_pp = np.zeros((npp, nslacs))
tein_est_lens_pp = np.zeros((npp, nslacs))
rein_lens_pp = np.zeros((npp, nslacs))
gammapl_lens_pp = np.zeros((npp, nslacs))
lmst_lens_pp = np.zeros((npp, nslacs))
rs_lens_pp = np.zeros((npp, nslacs))
gamma_lens_pp = np.zeros((npp, nslacs))
ms_lens_pp = np.zeros((npp, nslacs))
ms_obs_lens_pp = np.zeros((npp, nslacs))
lmstar_lens_pp = np.zeros((npp, nslacs))
re_lens_pp = np.zeros((npp, nslacs))
zd_lens_pp = np.zeros((npp, nslacs))
zs_lens_pp = np.zeros((npp, nslacs))
r200_lens_pp = np.zeros((npp, nslacs))
mh_lens_pp = np.zeros((npp, nslacs))
lsigma_lens_pp = np.zeros((npp, nslacs))
sigma_err_lens_pp = np.zeros((npp, nslacs))
sigma_obs_lens_pp = np.zeros((npp, nslacs))

tein_nopf_pp = np.zeros((npp, nslacs))
tein_est_nopf_pp = np.zeros((npp, nslacs))
rein_nopf_pp = np.zeros((npp, nslacs))
gammapl_nopf_pp = np.zeros((npp, nslacs))
lmst_nopf_pp = np.zeros((npp, nslacs))
rs_nopf_pp = np.zeros((npp, nslacs))
gamma_nopf_pp = np.zeros((npp, nslacs))
ms_nopf_pp = np.zeros((npp, nslacs))
lmstar_nopf_pp = np.zeros((npp, nslacs))
re_nopf_pp = np.zeros((npp, nslacs))
zd_nopf_pp = np.zeros((npp, nslacs))
r200_nopf_pp = np.zeros((npp, nslacs))
mh_nopf_pp = np.zeros((npp, nslacs))
lsigma_nopf_pp = np.zeros((npp, nslacs))
sigma_obs_nopf_pp = np.zeros((npp, nslacs))

tein_sub_pp = np.zeros((npp, nsub))
tein_est_sub_pp = np.zeros((npp, nsub))
rein_sub_pp = np.zeros((npp, nsub))
lmst_sub_pp = np.zeros((npp, nsub))
rs_sub_pp = np.zeros((npp, nsub))
gamma_sub_pp = np.zeros((npp, nsub))
ms_sub_pp = np.zeros((npp, nsub))
lmstar_sub_pp = np.zeros((npp, nsub))
re_sub_pp = np.zeros((npp, nsub))
zd_sub_pp = np.zeros((npp, nsub))
r200_sub_pp = np.zeros((npp, nsub))
mh_sub_pp = np.zeros((npp, nsub))
lsigma_sub_pp = np.zeros((npp, nsub))
sigma_obs_sub_pp = np.zeros((npp, nsub))

for i in range(npp):

    ind1 = i%nwalkers
    ind2 = nstep-i//nwalkers-1
    print(i, ind1, ind2)

    lasps = chain['lasps'][ind1, ind2]
    eps = chain['eps'][ind1, ind2]
    mu_mh = chain['mu_mh'][ind1, ind2]
    beta_mh = chain['beta_mh'][ind1, ind2]
    sigma_mh = chain['sigma_mh'][ind1, ind2]
    mu_sigma = chain['mu_sigma'][ind1, ind2]
    beta_sigma = chain['beta_sigma'][ind1, ind2]
    xi_sigma = chain['xi_sigma'][ind1, ind2]
    nu_sigma = chain['nu_sigma'][ind1, ind2]
    sigma_sigma = chain['sigma_sigma'][ind1, ind2]
    t_find = chain['t_find'][ind1, ind2]
    la_find = chain['la_find'][ind1, ind2]

    lasps_pp[i] = lasps
    eps_pp[i] = eps
    mu_mh_pp[i] = mu_mh
    beta_mh_pp[i] = beta_mh
    sigma_mh_pp[i] = sigma_mh
    mu_sigma_pp[i] = mu_sigma
    beta_sigma_pp[i] = beta_sigma
    xi_sigma_pp[i] = xi_sigma
    nu_sigma_pp[i] = nu_sigma
    sigma_sigma_pp[i] = sigma_sigma

    mu_mh_pop = mu_mh + beta_mh * (ms_pop - 11.3)
    mh_pop = mu_mh_pop + sigma_mh * mh_scat

    r200_pop = (10.**mh_pop * 3./200./(4*np.pi)/rhoc_pop)**(1./3.) * 1000.
    rs_pop = r200_pop/c200_func(mh_pop)

    mu_sigma_pop = mu_sigma + beta_sigma * (ms_pop - mpiv_slacs) + xi_sigma * (re_pop - mu_r_pop) + nu_sigma * (mh_pop - mu_mh_pop)

    lsigma_pop = mu_sigma_pop + sigma_sigma * lsigma_scat_pop

    sigma_err_pop = 10.**lsigma_pop * sigma_relerr_pop
    sigma_obs_pop = 10.**lsigma_pop * (1. + sigma_relerr_pop)
    lsigma_obs_pop = np.log10(sigma_obs_pop)

    tein_est_pop = np.rad2deg(4.*np.pi * (sigma_obs_pop/3e5)**2 * drat_pop) * 3600.

    lasps_pop = lasps * np.ones(npop)
    # computes the lensing cross-section of each galaxy
    mstar_pop = 10.**(ms_pop + lasps_pop)
    cs_pop = np.zeros(npop)
    rs_pop = np.zeros(npop)
    gamma_pop = np.zeros(npop)
    for j in range(npop):
        point = np.array([eps, mh_pop[j], lasps_pop[j]]).reshape((1, 3))
        rs_pop[j] = rs_pop_interps[j].eval(point)
        gamma_pop[j] = gamma_pop_interps[j].eval(point)
        if bkg_pop[j]:
            cs_pop[j] = cs_pop_interps[j].eval(point)

    psel = cs_pop * pfind_func(tein_est_pop, t_find, 10.**la_find)
    ind_lenses = np.random.choice(np.arange(npop, dtype=int), size=nslacs, p=psel/psel.sum())

    mh_lens = mh_pop[ind_lenses]
    r200_lens = r200_pop[ind_lenses]
    rs_lens = rs_pop[ind_lenses]
    gamma_lens = gamma_pop[ind_lenses]
    cvir_lens = r200_lens/rs_lens
    mstar_lens = 10.**(ms_pop[ind_lenses] + lasps)
    reff_lens = 10.**re_pop[ind_lenses]
    s_cr_lens = s_cr_pop[ind_lenses]
    arcsec2kpc_lens = arcsec2kpc_pop[ind_lenses]

    # computes lambda_int
    rein_lens = np.zeros(nslacs)
    psi2_lens = np.zeros(nslacs)
    psi3_lens = np.zeros(nslacs)
    for j in range(nslacs):
        gnfw_norm = 10.**mh_lens[j] / gnfw.M3d(r200_lens[j], rs_lens[j], gamma_lens[j])
        rein_here = get_rein_kpc(mstar_lens[j], reff_lens[j], gnfw_norm, rs_lens[j], gamma_lens[j], s_cr_lens[j])
        rein_lens[j] = rein_here
        alpha_up = alpha_kpc(rein_here+dx, gnfw_norm, rs_lens[j], gamma_lens[j], mstar_lens[j], reff_lens[j], s_cr_lens[j])
        alpha_dw = alpha_kpc(rein_here-dx, gnfw_norm, rs_lens[j], gamma_lens[j], mstar_lens[j], reff_lens[j], s_cr_lens[j])

        psi2_lens[j] = (alpha_up - alpha_dw)/(2.*dx)
        psi3_lens[j] = (alpha_up + alpha_dw - 2.*rein_here)/dx**2

    tein_lens = rein_lens / arcsec2kpc_lens

    psi2_pl = -psi3_lens/(1. - psi2_lens) * rein_lens
    lmst = (1. - psi2_lens)/(1. - psi2_pl)

    # fits the fundamental hyper-plane relation to the lenses
    ms_lens = ms_pop[ind_lenses]
    re_lens = re_pop[ind_lenses]
    mh_lens = mh_pop[ind_lenses]
    mu_mh_lens = mu_mh + beta_mh * (ms_lens - 11.3)

    def fitfunc(p):
        return p[0] + p[1]*(ms_lens - mpiv_slacs) + p[2]*(re_lens - hb09quad_mu_r_func(ms_lens)) + p[3] * (mh_lens - mu_mh_lens)

    def errfunc(p):
        return fitfunc(p) - lsigma_pop[ind_lenses]

    p0 = (2.37, 0.3, -0.4, 0.1)
    pfit = leastsq(errfunc, p0)

    lens_sigma_sigma_pp[i] = fpfit_scat = errfunc(pfit[0]).std()

    lens_mu_sigma_pp[i] = pfit[0][0]
    lens_beta_sigma_pp[i] = pfit[0][1]
    lens_xi_sigma_pp[i] = pfit[0][2]
    lens_nu_sigma_pp[i] = pfit[0][3]

    def sobs_errfunc(p):
        return fitfunc(p) - lsigma_obs_pop[ind_lenses]

    p0 = (2.37, 0.3, -0.4, 0.1)
    pfit = leastsq(sobs_errfunc, p0)

    lens_sigma_sobs_pp[i] = fpfit_scat = sobs_errfunc(pfit[0]).std()

    lens_mu_sobs_pp[i] = pfit[0][0]
    lens_beta_sobs_pp[i] = pfit[0][1]
    lens_xi_sobs_pp[i] = pfit[0][2]
    lens_nu_sobs_pp[i] = pfit[0][3]

    zd_lens_pp[i, :] = zd_pop[ind_lenses]
    zs_lens_pp[i, :] = zs_pop[ind_lenses]
    r200_lens_pp[i, :] = r200_pop[ind_lenses]
    tein_lens_pp[i, :] = tein_lens
    rein_lens_pp[i, :] = rein_lens
    lmst_lens_pp[i, :] = lmst
    gammapl_lens_pp[i, :] = 2. - psi2_pl
    ms_lens_pp[i, :] = ms_pop[ind_lenses]
    ms_obs_lens_pp[i, :] = ms_obs_pop[ind_lenses]
    lmstar_lens_pp[i, :] = np.log10(mstar_pop[ind_lenses])
    re_lens_pp[i, :] = re_pop[ind_lenses]
    mh_lens_pp[i, :] = mh_lens
    rs_lens_pp[i, :] = rs_lens
    gamma_lens_pp[i, :] = gamma_lens
    lsigma_lens_pp[i, :] = lsigma_pop[ind_lenses]
    sigma_obs_lens_pp[i, :] = sigma_obs_pop[ind_lenses]
    sigma_err_lens_pp[i, :] = sigma_err_pop[ind_lenses]
    tein_est_lens_pp[i, :] = tein_est_pop[ind_lenses]

    # now simulates the noisy gammpl measurements and fits the Sigma_*-gammapl relation

    ind_amy = np.random.choice(np.arange(nslacs), size=n_amy, replace=False)
    sstar_amy = ms_obs_pop[ind_lenses][ind_amy] - 2.*re_pop[ind_lenses][ind_amy] - np.log10(2.*np.pi)
    gammapl_amy = (2. - psi2_pl)[ind_amy] + np.random.normal(0., amy_gpl_err, n_amy)
    m_amy, b_amy, scat_amy = do_leastsq(sstar_amy, gammapl_amy)

    amy_mu_gammapl_pp[i] = b_amy
    amy_beta_gammapl_pp[i] = m_amy
    amy_sigma_gammapl_pp[i] = scat_amy

    ind_shajib = np.random.choice(np.arange(nslacs), size=n_shajib, replace=False)
    sstar_shajib = ms_obs_pop[ind_lenses][ind_shajib] - 2.*re_pop[ind_lenses][ind_shajib] - np.log10(2.*np.pi)
    gammapl_shajib = (2. - psi2_pl)[ind_shajib] + np.random.normal(0., shajib_gpl_err, n_shajib)
    m_shajib, b_shajib, scat_shajib = do_leastsq(sstar_shajib, gammapl_shajib)

    shajib_mu_gammapl_pp[i] = b_shajib
    shajib_beta_gammapl_pp[i] = m_shajib
    shajib_sigma_gammapl_pp[i] = scat_shajib

    ind_tan = np.random.choice(np.arange(nslacs), size=n_tan, replace=False)
    sstar_tan = ms_obs_pop[ind_lenses][ind_tan] - 2.*re_pop[ind_lenses][ind_tan] - np.log10(2.*np.pi)
    gammapl_tan = (2. - psi2_pl)[ind_tan] + np.random.normal(0., tan_gpl_err, n_tan)
    m_tan, b_tan, scat_tan = do_leastsq(sstar_tan, gammapl_tan)

    tan_mu_gammapl_pp[i] = b_tan
    tan_beta_gammapl_pp[i] = m_tan
    tan_sigma_gammapl_pp[i] = scat_tan

    # draws a sample of lenses, with no pfind term
    #ind_nopfind, lsigma, sigma_obs, tein_nopf, tein_est_nopf, rs_nopf, gamma = draw_nopfind_lenses(pop, p)
    psel = cs_pop.copy()
    ind_nopfind = np.random.choice(np.arange(npop), size=nslacs, p=psel/psel.sum())

    mh_nopf = mh_pop[ind_nopfind]
    r200_nopf = r200_pop[ind_nopfind]
    rs_nopf = rs_pop[ind_nopfind]
    gamma_nopf = gamma_pop[ind_nopfind]
    cvir_nopf = r200_nopf/rs_nopf
    mstar_nopf = 10.**(ms_pop[ind_nopfind] + lasps)
    reff_nopf = 10.**re_pop[ind_nopfind]
    s_cr_nopf = s_cr_pop[ind_nopfind]
    arcsec2kpc_nopf = arcsec2kpc_pop[ind_nopfind]

    # computes lambda_int
    rein_nopf = np.zeros(nslacs)
    psi2_nopf = np.zeros(nslacs)
    psi3_nopf = np.zeros(nslacs)
    for j in range(nslacs):
        gnfw_norm = 10.**mh_nopf[j] / gnfw.M3d(r200_nopf[j], rs_nopf[j], gamma_nopf[j])
        rein_here = get_rein_kpc(mstar_nopf[j], reff_nopf[j], gnfw_norm, rs_nopf[j], gamma_nopf[j], s_cr_nopf[j])
        rein_nopf[j] = rein_here
        alpha_up = alpha_kpc(rein_here+dx, gnfw_norm, rs_nopf[j], gamma_nopf[j], mstar_nopf[j], reff_nopf[j], s_cr_nopf[j])
        alpha_dw = alpha_kpc(rein_here-dx, gnfw_norm, rs_nopf[j], gamma_nopf[j], mstar_nopf[j], reff_nopf[j], s_cr_nopf[j])

        psi2_nopf[j] = (alpha_up - alpha_dw)/(2.*dx)
        psi3_nopf[j] = (alpha_up + alpha_dw - 2.*rein_here)/dx**2

    tein_nopf = rein_nopf / arcsec2kpc_nopf
    psi2_pl = -psi3_nopf/(1. - psi2_nopf) * rein_nopf
    lmst = (1. - psi2_nopf)/(1. - psi2_pl)

    zd_nopf_pp[i, :] = zd_pop[ind_nopfind]
    r200_nopf_pp[i, :] = r200_pop[ind_nopfind]
    tein_nopf_pp[i, :] = tein_nopf
    rein_nopf_pp[i, :] = rein_nopf
    gammapl_nopf_pp[i, :] = 2. - psi2_pl
    lmst_nopf_pp[i, :] = lmst
    ms_nopf_pp[i, :] = ms_pop[ind_nopfind]
    lmstar_nopf_pp[i, :] = np.log10(mstar_pop[ind_nopfind])
    re_nopf_pp[i, :] = re_pop[ind_nopfind]
    mh_nopf_pp[i, :] = mh_nopf
    rs_nopf_pp[i, :] = rs_nopf
    gamma_nopf_pp[i, :] = gamma_nopf
    lsigma_nopf_pp[i, :] = lsigma_pop[ind_nopfind]
    sigma_obs_nopf_pp[i, :] = sigma_obs_pop[ind_nopfind]
    tein_est_nopf_pp[i, :] = tein_est_pop[ind_nopfind]

    # draws a random sample from the population
    ind_sub = np.random.choice(np.arange(npop, dtype=int), size=nsub)

    zd_sub_pp[i, :] = zd_pop[ind_sub]
    r200_sub_pp[i, :] = r200_pop[ind_sub]
    tein_est_sub_pp[i, :] = tein_est_pop[ind_sub]
    ms_sub_pp[i, :] = ms_pop[ind_sub]
    lmstar_sub_pp[i, :] = np.log10(mstar_pop[ind_sub])
    re_sub_pp[i, :] = re_pop[ind_sub]
    mh_sub_pp[i, :] = mh_pop[ind_sub]
    rs_sub_pp[i, :] = rs_pop[ind_sub]
    gamma_sub_pp[i, :] = gamma_pop[ind_sub]
    lsigma_sub_pp[i, :] = lsigma_pop[ind_sub]
    sigma_obs_sub_pp[i, :] = sigma_obs_pop[ind_sub]

output = h5py.File('short_adcontr_pp.hdf5', 'w')
output.attrs['npp'] = npp
output.attrs['nsub'] = nsub

hp_group = output.create_group('hyperpars')

hp_group.create_dataset('lasps', data=lasps_pp)
hp_group.create_dataset('eps', data=eps_pp)
hp_group.create_dataset('mu_mh', data=mu_mh_pp)
hp_group.create_dataset('beta_mh', data=beta_mh_pp)
hp_group.create_dataset('sigma_mh', data=sigma_mh_pp)
hp_group.create_dataset('mu_sigma', data=mu_sigma_pp)
hp_group.create_dataset('beta_sigma', data=beta_sigma_pp)
hp_group.create_dataset('xi_sigma', data=xi_sigma_pp)
hp_group.create_dataset('nu_sigma', data=nu_sigma_pp)
hp_group.create_dataset('sigma_sigma', data=sigma_sigma_pp)

hp_group = output.create_group('lens_hyperplane')

hp_group.create_dataset('mu_sigma', data=lens_mu_sigma_pp)
hp_group.create_dataset('beta_sigma', data=lens_beta_sigma_pp)
hp_group.create_dataset('xi_sigma', data=lens_xi_sigma_pp)
hp_group.create_dataset('nu_sigma', data=lens_nu_sigma_pp)
hp_group.create_dataset('sigma_sigma', data=lens_sigma_sigma_pp)

hp_group = output.create_group('lens_sapobsplane')

hp_group.create_dataset('mu_sigma', data=lens_mu_sobs_pp)
hp_group.create_dataset('beta_sigma', data=lens_beta_sobs_pp)
hp_group.create_dataset('xi_sigma', data=lens_xi_sobs_pp)
hp_group.create_dataset('nu_sigma', data=lens_nu_sobs_pp)
hp_group.create_dataset('sigma_sigma', data=lens_sigma_sobs_pp)

amy_group = output.create_group('amy_pars')

amy_group.create_dataset('mu_gammapl', data=amy_mu_gammapl_pp)
amy_group.create_dataset('beta_gammapl', data=amy_beta_gammapl_pp)
amy_group.create_dataset('sigma_gammapl', data=amy_sigma_gammapl_pp)

shajib_group = output.create_group('shajib_pars')

shajib_group.create_dataset('mu_gammapl', data=shajib_mu_gammapl_pp)
shajib_group.create_dataset('beta_gammapl', data=shajib_beta_gammapl_pp)
shajib_group.create_dataset('sigma_gammapl', data=shajib_sigma_gammapl_pp)

tan_group = output.create_group('tan_pars')

tan_group.create_dataset('mu_gammapl', data=tan_mu_gammapl_pp)
tan_group.create_dataset('beta_gammapl', data=tan_beta_gammapl_pp)
tan_group.create_dataset('sigma_gammapl', data=tan_sigma_gammapl_pp)

lens_group = output.create_group('lenses')

lens_group.create_dataset('tein', data=tein_lens_pp)
lens_group.create_dataset('tein_est', data=tein_est_lens_pp)
lens_group.create_dataset('rein', data=rein_lens_pp)
lens_group.create_dataset('ms', data=ms_lens_pp)
lens_group.create_dataset('ms_obs', data=ms_obs_lens_pp)
lens_group.create_dataset('lmstar', data=lmstar_lens_pp)
lens_group.create_dataset('re', data=re_lens_pp)
lens_group.create_dataset('zd', data=zd_lens_pp)
lens_group.create_dataset('zs', data=zs_lens_pp)
lens_group.create_dataset('mh', data=mh_lens_pp)
lens_group.create_dataset('r200', data=r200_lens_pp)
lens_group.create_dataset('rs', data=rs_lens_pp)
lens_group.create_dataset('gamma', data=gamma_lens_pp)
lens_group.create_dataset('sigma', data=10.**lsigma_lens_pp)
lens_group.create_dataset('sigma_obs', data=sigma_obs_lens_pp)
lens_group.create_dataset('sigma_err', data=sigma_err_lens_pp)
lens_group.create_dataset('gammapl', data=gammapl_lens_pp)
lens_group.create_dataset('lmst', data=lmst_lens_pp)

nopf_group = output.create_group('nopfind')

nopf_group.create_dataset('tein', data=tein_nopf_pp)
nopf_group.create_dataset('tein_est', data=tein_est_nopf_pp)
nopf_group.create_dataset('rein', data=rein_nopf_pp)
nopf_group.create_dataset('gammapl', data=gammapl_nopf_pp)
nopf_group.create_dataset('lmst', data=lmst_nopf_pp)
nopf_group.create_dataset('rs', data=rs_nopf_pp)
nopf_group.create_dataset('gamma', data=gamma_nopf_pp)
nopf_group.create_dataset('ms', data=ms_nopf_pp)
nopf_group.create_dataset('lmstar', data=lmstar_nopf_pp)
nopf_group.create_dataset('re', data=re_nopf_pp)
nopf_group.create_dataset('zd', data=zd_nopf_pp)
nopf_group.create_dataset('r200', data=r200_nopf_pp)
nopf_group.create_dataset('mh', data=mh_nopf_pp)
nopf_group.create_dataset('sigma', data=10.**lsigma_nopf_pp)
nopf_group.create_dataset('sigma_obs', data=sigma_obs_nopf_pp)

sub_group = output.create_group('subset')

sub_group.create_dataset('rein', data=rein_sub_pp)
sub_group.create_dataset('rs', data=rs_sub_pp)
sub_group.create_dataset('gamma', data=gamma_sub_pp)
sub_group.create_dataset('ms', data=ms_sub_pp)
sub_group.create_dataset('lmstar', data=lmstar_sub_pp)
sub_group.create_dataset('re', data=re_sub_pp)
sub_group.create_dataset('zd', data=zd_sub_pp)
sub_group.create_dataset('r200', data=r200_sub_pp)
sub_group.create_dataset('mh', data=mh_sub_pp)
sub_group.create_dataset('sigma', data=10.**lsigma_sub_pp)
sub_group.create_dataset('sigma_obs', data=sigma_obs_sub_pp)

