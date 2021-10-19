import numpy as np
from sl_profiles import nfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun, yr
import sl_cosmology
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.special import erf
import emcee
import h5py
import sys


mockname = '1e4mock_0'

sampdir = './'

mockfile = h5py.File('%s_pop.hdf5'%mockname, 'r')

nwalkers = 100
nstep = 200
burnin = 100

zd = mockfile.attrs['zd']
zs_grid = mockfile['Grids/zs_grid'][()]
pz_grid = mockfile['Grids/pz_grid'][()]
mstar_sch_grid = mockfile['Grids/mstar_sch_grid'][()]
s_cr_grid = mockfile['Grids/s_cr_grid'][()]
ms_grid = mockfile['Grids/ms_grid'][()]
dvol_dz_grid = mockfile['Grids/dvol_dz_grid'][()]

pz_spline = splrep(zs_grid, pz_grid)
dvol_dz_spline = splrep(zs_grid, dvol_dz_grid)
s_cr_spline = splrep(zs_grid, s_cr_grid)
mstar_sch_spline = splrep(zs_grid, mstar_sch_grid)

alpha_sch = mockfile.attrs['alpha_sch']

c200 = mockfile.attrs['c200']

maxmagB_det = mockfile.attrs['maxmagB_det']
ms_min = mockfile.attrs['ms_min']
ms_max = mockfile.attrs['ms_max']
mag_err = mockfile.attrs['mag_err']

lenses = mockfile['Lenses']

lm200_true = mockfile['Lenses/lm200'][()]

lreff = lenses['lreff'][()]
llum = lenses['llum'][()]
xA = lenses['xA'][()]
xB = lenses['xB'][()]
magA_obs = lenses['magA_obs'][()]
magB_obs = lenses['magB_obs'][()]

nlens = len(xA)

lm200_min = 11.
lm200_max = 15.

lustar_min = 0.
lustar_max = 1.

rhoc = sl_cosmology.rhoc(zd)

# defines lensing-related functions
def alpha_dm(s_cr, x, nfw_norm, rs):
    # deflection angle (in kpc)
    return nfw_norm * nfw.M2d(abs(x), rs) / np.pi/x/s_cr

def alpha_star(s_cr, x, mstar, reff): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(s_cr, x, nfw_norm, rs, mstar, reff):
    return alpha_dm(s_cr, x, nfw_norm, rs) + alpha_star(s_cr, x, mstar, reff)

def kappa(s_cr, x, nfw_norm, rs, mstar, reff): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + nfw_norm * nfw.Sigma(abs(x), rs))/s_cr
   
def mu_r(s_cr, x, nfw_norm, rs, mstar, reff):
    # radial magnification
    return (1. + alpha(s_cr, x, nfw_norm, rs, mstar, reff)/x - 2.*kappa(s_cr, x, nfw_norm, rs, mstar, reff))**(-1)

def mu_t(s_cr, x, nfw_norm, rs, mstar, reff):
    # tangential magnification
    return (1. - alpha(s_cr, x, nfw_norm, rs, mstar, reff)/x)**(-1)

def mu_tot(s_cr, x, nfw_norm, rs, mstar, reff):
    return mu_r(s_cr, x, nfw_norm, rs, mstar, reff) * mu_t(s_cr, x, nfw_norm, rs, mstar, reff)

dx = 0.0001

xmin = 0.01

mstar_min = 1e9
niter_max = 10

samps_file = h5py.File(sampdir+'/%s_lensmodel_samps.hdf5'%mockname, 'w')

samps_file.attrs['c200'] = c200
samps_file.attrs['lm200_min'] = lm200_min
samps_file.attrs['lm200_max'] = lm200_max
samps_file.attrs['lustar_min'] = lustar_min
samps_file.attrs['lustar_max'] = lustar_max

for i in range(nlens):

    reff = 10.**lreff[i]

    xA_up = xA[i] + dx
    xA_dw = xA[i] - dx
    
    xB_up = xB[i] + dx
    xB_dw = xB[i] - dx

    lm200_par = {'name': 'lm200', 'lower': lm200_min, 'upper': lm200_max, 'guess': lm200_true[i]-0.1, 'spread': 0.01}
    zs_par = {'name': 'zs', 'lower': zs_grid[0], 'upper': zs_grid[-1], 'guess': 2., 'spread': 1.}
    ms_par = {'name': 'ms', 'lower': ms_grid[0], 'upper': ms_grid[-1], 'guess': 25., 'spread': 1.}
    
    pars = [lm200_par, zs_par, ms_par]
    npars = len(pars)
    
    bounds = []
    for par in pars:
        bounds.append((par['lower'], par['upper']))
    
    # picks a reasonable starting guess
    zs_start = np.random.rand(nwalkers) * (zs_grid[-1] - zs_grid[0]) + zs_grid[0]
    ms_start = np.random.rand(nwalkers) * (ms_grid[-1] - ms_grid[0]) + ms_grid[0]

    s_cr_start = splev(zs_start, s_cr_spline)

    lm200_start = 0. * zs_start
    for j in range(nwalkers):

        alpha_star_diff = alpha_star(s_cr_start[j], xA[i], mstar_min, reff) - alpha_star(s_cr_start[j], xB[i], mstar_min, reff)
        def zerofunc(lm200):
            r200 = (10.**lm200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
            rs = r200/c200
            nfw_norm = 10.**lm200/nfw.M3d(r200, rs)
            return xA[i] - xB[i] - alpha_star_diff - alpha_dm(s_cr_start[j], xA[i], nfw_norm, rs) + alpha_dm(s_cr_start[j], xB[i], nfw_norm, rs)

        if zerofunc(lm200_max) > 0.:
            lm200_max_here = lm200_max
        else:
            lm200_max_here = brentq(zerofunc, lm200_min, lm200_max)

        mur_ok = False
        niter = 0

        while not mur_ok and niter < niter_max:

            lm200_here = np.random.rand(1) * (lm200_max_here - lm200_min) + lm200_min
            r200_here = (10.**lm200_here*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
            rs_here = r200_here/c200
            nfw_norm_here = 10.**lm200_here/nfw.M3d(r200_here, rs_here)
       
            alpha_star_xA = alpha_star(s_cr_start[j], xA[i], 1., reff)
            alpha_star_xB = alpha_star(s_cr_start[j], xB[i], 1., reff)

            alpha_dm_xA = alpha_dm(s_cr_start[j], xA[i], nfw_norm_here, rs_here)
            alpha_dm_xB = alpha_dm(s_cr_start[j], xB[i], nfw_norm_here, rs_here)
           
            mstar_here = (xA[i] - xB[i] - alpha_dm_xA + alpha_dm_xB)/(alpha_star_xA - alpha_star_xB)

            mur_B_here = mu_r(s_cr_start[j], xB[i], nfw_norm_here, rs_here, mstar_here, reff)

            if mur_B_here > 0.:
               mur_ok = True

            niter += 1

            lm200_start[j] = lm200_here

        if niter >= niter_max:
            print('reached maximum number of iterations. Aborting.')
            df

    start = np.array((lm200_start, zs_start, ms_start)).T

    def logphifunc(m, mstar):
        return -0.4*(m - mstar) * (alpha_sch+1.) * np.log(10.) -10.**(-0.4*(m - mstar))
    
    def logpz(zs):
        return np.log(splev(zs, pz_spline))

    def logpzms(zs, ms, mstar):
        return logphifunc(ms, mstar) + np.log(splev(zs, dvol_dz_spline))

    def logprior(p):
        for i in range(npars):
            if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
                return -1e300

        lm200, zs, ms = p
        mstar_sch_here = splev(zs, mstar_sch_spline)

        #return logpz(zs) + logphifunc(ms, mstar_sch_here)
        return logpzms(zs, ms, mstar_sch_here)

    def logpfunc(p):

        lprior = logprior(p)
        if lprior <= -1e300:
            return -1e300, -np.inf, 0., np.inf

        lm200, zs, ms = p

        r200 = (10.**lm200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
        rs = r200/c200

        nfw_norm = 10.**lm200/nfw.M3d(r200, rs)

        s_cr = splev(zs, s_cr_spline)

        alpha_star_xA = alpha_star(s_cr, xA[i], 1., reff)
        alpha_star_xA_up = alpha_star(s_cr, xA[i]+dx, 1., reff)
        alpha_star_xA_dw = alpha_star(s_cr, xA[i]-dx, 1., reff)
    
        alpha_star_xB = alpha_star(s_cr, xB[i], 1., reff)
        alpha_star_xB_up = alpha_star(s_cr, xB[i]+dx, 1., reff)
        alpha_star_xB_dw = alpha_star(s_cr, xB[i]-dx, 1., reff)

        # given the halo mass, calculates stellar mass needed to obtain the observed image positions

        alpha_dm_xA = alpha_dm(s_cr, xA[i], nfw_norm, rs)
        alpha_dm_xA_up = alpha_dm(s_cr, xA[i]+dx, nfw_norm, rs)
        alpha_dm_xA_dw = alpha_dm(s_cr, xA[i]-dx, nfw_norm, rs)
    
        alpha_dm_xB = alpha_dm(s_cr, xB[i], nfw_norm, rs)
        alpha_dm_xB_up = alpha_dm(s_cr, xB[i]+dx, nfw_norm, rs)
        alpha_dm_xB_dw = alpha_dm(s_cr, xB[i]-dx, nfw_norm, rs)
       
        mstar_here = (xA[i] - xB[i] - alpha_dm_xA + alpha_dm_xB)/(alpha_star_xA - alpha_star_xB)
   
        mu_xA = mu_tot(s_cr, xA[i], nfw_norm, rs, mstar_here, reff) 
        mu_xB = mu_tot(s_cr, xB[i], nfw_norm, rs, mstar_here, reff) 
        mur_xB = mu_r(s_cr, xB[i], nfw_norm, rs, mstar_here, reff) 

        magA = ms - 2.5*np.log10(abs(mu_xA))
        magB = ms - 2.5*np.log10(abs(mu_xB))

        if mstar_here < mstar_min or mur_xB < 0.:
            return -1e300, -np.inf, 0., magB

        beta = xA[i] - alpha_dm_xA - mstar_here*alpha_star_xA

        mstar_xA_up = (xA_up - xB[i] - alpha_dm_xA_up + alpha_dm_xB)/(alpha_star_xA_up - alpha_star_xB)
        mstar_xA_dw = (xA_dw - xB[i] - alpha_dm_xA_dw + alpha_dm_xB)/(alpha_star_xA_dw - alpha_star_xB)
    
        mstar_xB_up = (xA[i] - xB_up - alpha_dm_xA + alpha_dm_xB_up)/(alpha_star_xA - alpha_star_xB_up)
        mstar_xB_dw = (xA[i] - xB_dw - alpha_dm_xA + alpha_dm_xB_dw)/(alpha_star_xA - alpha_star_xB_dw)
    
        beta_xA_up = xA_up - alpha_dm_xA_up - mstar_xA_up * alpha_star_xA_up
        beta_xA_dw = xA_dw - alpha_dm_xA_dw - mstar_xA_dw * alpha_star_xA_dw
    
        beta_xB_up = xA[i] - alpha_dm_xA - mstar_xB_up * alpha_star_xA
        beta_xB_dw = xA[i] - alpha_dm_xA - mstar_xB_dw * alpha_star_xA
    
        dlmstar_dxA = (np.log10(mstar_xA_up) - np.log10(mstar_xA_dw))/(2.*dx)
        dlmstar_dxB = (np.log10(mstar_xB_up) - np.log10(mstar_xB_dw))/(2.*dx)
    
        dbeta_dxA = (beta_xA_up - beta_xA_dw)/(2.*dx)
        dbeta_dxB = (beta_xB_up - beta_xB_dw)/(2.*dx)
    
        detJ = abs(dlmstar_dxA*dbeta_dxB - dlmstar_dxB*dbeta_dxA)

        #beta_term = beta * 0.5 * (1. - erf((magB - maxmagB_det)/2.**0.5/mag_err))

        mags_logp = -0.5*(magA - magA_obs[i])**2/mag_err**2 - 0.5*(magB - magB_obs[i])**2/mag_err**2

        sumlogp = lprior + mags_logp + np.log(detJ)

        if sumlogp != sumlogp:
            return -1e300, -np.inf, beta, magB

        return sumlogp, np.log10(mstar_here), beta, magB

    print(i)

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=1)

    sampler.run_mcmc(start, nstep)

    blobs_chain = sampler.blobs

    lmstar_chain = blobs_chain[:, :, 0].T
    beta_chain = blobs_chain[:, :, 1].T
    magB_chain = blobs_chain[:, :, 2].T

    group = samps_file.create_group('lens_%04d'%i)

    group.create_dataset('lm200_samp', data=sampler.chain[:, burnin:, 0])
    group.create_dataset('zs_samp', data=sampler.chain[:, burnin:, 1])
    group.create_dataset('ms_samp', data=sampler.chain[:, burnin:, 2])
    group.create_dataset('lustar_samp', data=lmstar_chain[:, burnin:] - llum[i])
    group.create_dataset('beta_samp', data=beta_chain[:, burnin:])
    group.create_dataset('magB_samp', data=magB_chain[:, burnin:])
    group.create_dataset('logp', data=sampler.lnprobability[:, burnin:])

samps_file.close()

