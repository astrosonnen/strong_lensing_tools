import numpy as np
import h5py
import ndinterp
from scipy.interpolate import splrep, splev, splint
from fitpars import *
from parent_sample_pars import *
from read_slacs import *
from masssize import *
from halo_pars import *
from adcontr_funcs import *
from gnfw_lensingfuncs import *
import sl_cosmology
from sl_cosmology import Dang, Mpc, M_Sun, c, G, kpc
from scipy.stats import truncnorm
import sys


nzd = 36
zd_grid = np.linspace(zmin, zmax, nzd)
arcsec2kpc_grid = np.zeros(nzd)
rhoc_grid = np.zeros(nzd)
dvdz_grid = np.zeros(nzd)
for n in range(nzd):
    dvdz_grid[n] = sl_cosmology.comovd(zd_grid[n])**2 * sl_cosmology.dcomovdz(zd_grid[n])
    arcsec2kpc_grid[n] = sl_cosmology.arcsec2kpc(zd_grid[n])
    rhoc_grid[n] = sl_cosmology.rhoc(zd_grid[n])

dvdz_spline = splrep(zd_grid, dvdz_grid)
arcsec2kpc_spline = splrep(zd_grid, arcsec2kpc_grid)
rhoc_spline = splrep(zd_grid, rhoc_grid)

nms = 101
ms_grid = np.linspace(lmchab_min, lmchab_max, nms)

def mtfunc(z):
    return mt0 + mt1*z + mt2*z**2 + mt3*z**3 + mt4*z**4 + mt5*z**5

def ftfunc(z, ms):
    return 1./np.pi * np.arctan((ms - mtfunc(z))/sigmat) + 0.5

def msdist(z, ms):
    return splev(z, dvdz_spline) * ftfunc(z, ms) * (10.**(ms - mbar))**(alpha + 1) * np.exp(-10.**(ms - mbar))

msint_grid = np.zeros(nzd)
for n in range(nzd):
    integrand_grid = msdist(zd_grid[n], ms_grid)
    integrand_spline = splrep(ms_grid, integrand_grid)
    msint_grid[n] = splint(ms_grid[0], ms_grid[-1], integrand_spline)

msint_spline = splrep(zd_grid, msint_grid) # this is the marginal z distribution

invnorm = splint(zd_grid[0], zd_grid[-1], msint_spline)

pzcum_grid = np.zeros(nzd)
for n in range(nzd):
    pzcum_grid[n] = splint(zmin, zd_grid[n], msint_spline)/invnorm

invpzcum_spline = splrep(pzcum_grid, zd_grid)

def get_pop_crosssect(pop, lasps, eps):

    npop = len(pop['ms'])

    cs_samp = np.zeros(npop)
    tein_samp = np.zeros(npop)
    rs_samp = np.zeros(npop)
    gamma_samp = np.zeros(npop)

    mstar_samp = 10.**(pop['ms'] + lasps)

    for i in range(npop):

        reff = 10.**pop['re'][i]
        rs_gnfw, gamma = find_gnfw(mstar_samp[i], reff, 10.**pop['mh'][i], pop['rs_0'][i], pop['r200'][i], eps=eps)
        rs_samp[i] = rs_gnfw
        gamma_samp[i] = gamma

        if pop['bkg'][i]:
            gnfw_norm = 10.**pop['mh'][i] / gnfw.M3d(pop['r200'][i], rs_gnfw, gamma)

            rein_here = get_rein_kpc(mstar_samp[i], reff, gnfw_norm, rs_gnfw, gamma, pop['s_cr'][i])
            if rein_here > 0.1:
                xrad_kpc, radcaust, xA_max = get_radcaust(mstar_samp[i], reff, gnfw_norm, rs_gnfw, gamma, pop['s_cr'][i], rein_here)

                cs_samp[i] = get_crosssect(mstar_samp[i], reff, gnfw_norm, rs_gnfw, gamma, pop['s_cr'][i], rein_here, xrad_kpc, xA_max, pop['arcsec2kpc'][i])
                tein_samp[i] = rein_here / pop['arcsec2kpc'][i]

    return cs_samp, tein_samp, rs_samp, gamma_samp

def get_msonly_pop(npop=10000, seedno=0):

    np.random.seed(seedno)

    # draw values of z
    zd_samp = splev(np.random.rand(npop), invpzcum_spline)
    arcsec2kpc_samp = splev(zd_samp, arcsec2kpc_spline)
    t_samp = np.random.rand(npop)
    
    rhoc_samp = splev(zd_samp, rhoc_spline)
    
    ms_samp = np.zeros(npop)
    
    # draws values of ms
    for i in range(npop):
        msfunc_grid = msdist(zd_samp[i], ms_grid)
        msfunc_spline = splrep(ms_grid, msfunc_grid, k=1)
    
        cump_grid = 0. * msfunc_grid
        for j in range(nms):
            cump_grid[j] = splint(ms_grid[0], ms_grid[j], msfunc_spline)
    
        cump_grid /= cump_grid[-1]
    
        invcump_spline = splrep(cump_grid, ms_grid, k=1)
      
        ms_samp[i] = splev(t_samp[i], invcump_spline)
    
    re_samp = hb09quad_mu_r_func(ms_samp) + np.random.normal(0., s19_sigma_r, npop)
    zs_samp = np.random.normal(mu_zs, sigma_zs, npop)
    
    # draws errors to be added to the model velocity dispersions
    sigma_relerr_samp = np.random.normal(0., slacs_median_sigma_relerr, npop)
    
    # prepares the comoving distance grid
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
    
    bkg_samp = zs_samp > zd_samp + 0.05
    
    s_cr_samp = s_cr_arrfunc(zd_samp, zs_samp)
    
    dds_samp = dds_arrfunc(zd_samp, zs_samp)
    ds_samp = splev(zs_samp, dang_spline)
    drat_samp = dds_samp/ds_samp

    # draws errors to be added to the model velocity dispersions
    sigma_relerr_samp = np.random.normal(0., slacs_median_sigma_relerr, npop)
    # scale-free scatter in velocity dispersion
    lsigma_scat_samp = np.random.normal(0., 1., npop)
    # scale-free scatter in halo mass
    mh_scat_samp = np.random.normal(0., 1., npop)

    pop = {}
    pop['ms'] = ms_samp
    pop['re'] = re_samp
    pop['zd'] = zd_samp
    pop['zs'] = zs_samp
    pop['rhoc'] = rhoc_samp
    pop['drat'] = drat_samp
    pop['s_cr'] = s_cr_samp
    pop['bkg'] = bkg_samp
    pop['arcsec2kpc'] = arcsec2kpc_samp
    pop['lsigma_scat'] = lsigma_scat_samp
    pop['sigma_relerr'] = sigma_relerr_samp
    pop['mh_scat'] = mh_scat_samp

    return pop

