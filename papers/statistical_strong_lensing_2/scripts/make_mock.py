import numpy as np
from sl_profiles import nfw, gnfw, deVaucouleurs as deV, eaglecontr
from sl_cosmology import Mpc, c, G, M_Sun, yr
import sl_cosmology
from scipy.optimize import brentq, minimize_scalar, leastsq
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev, splint, interp1d
from scipy.integrate import quad
import h5py
import sys


if len(sys.argv) > 1:
    seedno = int(sys.argv[1])
else:
    seedno = 0
mockname = 'mock_%d'%seedno

h = 0.7
cosmo = sl_cosmology.default_cosmo 
cosmo['h'] = h

ngal = 100 # sample size

np.random.seed(seedno)

lmsps_piv = 11.5 # pivot point of stellar mass-halo mass and stellar mass-size relation
lmsps_err = 0.15 # uncertainty on logM* from stellar population synthesis
rmur_err = 0.05 # uncertainty on the radial magnification ratio between image 1 and 2
dt_err = 5. # uncertainty on the time delay, in days

max_asymm = 0.5 # maximum image configuration asymmetry

zd = 0.4 # lens redshift
zs = 1.5 # source redshift
c200 = 5. # halo concentration
gnfw_c200 = 5. # halo concentration assumed in the gNFW fit
rs_fixed = 100. # scale radius assumed in the fit with fixed mdm5

lmsps_mu = 11.4 # average value of logM*^(sps)
lmsps_sig = 0.3 # intrinsic scatter in logM*^(sps)

laimf_mu = 0.1 # average value of log(alpha_IMF)
laimf_sig = 0. # intrinsic scatter in log(alpha_IMF)

lreff_mu = 1. # average value of log(Reff) at logM*=lmsps_piv
lreff_beta = 0.8 # slope of mass-size relation
lreff_sig = 0.15 # intrinsic scatter in Reff at fixed logM*

lm200_mu = 13. # average logM200 at logM*=lmsps_piv
lm200_sig = 0.2 # intrinsic scatter in logM200
lm200_beta = 1.5 # slope of stellar mass-halo mass relation

# scale-free source position: uniform distribution in a circle of radius 1
beta_scalefree_samp = np.random.rand(ngal)**0.5

# generate the values of stellar mass, size, IMF, halo mass
lmsps_samp = np.random.normal(lmsps_mu, lmsps_sig, ngal)

lreff_samp = lreff_mu + lreff_beta * (lmsps_samp - lmsps_piv) + np.random.normal(0., lreff_sig, ngal)

lm200_samp = lm200_mu + lm200_beta * (lmsps_samp - lmsps_piv) + np.random.normal(0., lm200_sig, ngal)

laimf_samp = laimf_mu + laimf_sig * np.random.normal(0., 1., ngal)
lmstar_samp = lmsps_samp + laimf_samp

# adds observational errors to the stellar mass measurements
lmsps_obs = lmsps_samp + np.random.normal(0., lmsps_err, ngal)

# generates observational errors on the radial magnification ratios (adds them later)
rmur_deltas = np.random.normal(0., rmur_err, ngal)

# generates observational errors on the time delay (adds them later)
dt_deltas = np.random.normal(0., dt_err, ngal)

# prepares arrays to store lensing info
rein_samp = np.zeros(ngal)

beta_samp = np.zeros(ngal)
beta_max_samp = np.zeros(ngal)
xA_samp = np.zeros(ngal)
xB_samp = np.zeros(ngal)
lmdm5_samp = np.zeros(ngal)
rmur_samp = np.zeros(ngal)
dt_samp = np.zeros(ngal)
gnfwfit_gammadm_samp = np.zeros(ngal)
gnfwfit_rs_samp = np.zeros(ngal)
mdm5fit_gammadm_samp = np.zeros(ngal)
xradcrit_samp = np.zeros(ngal)

kpc = Mpc/1000.
day = yr/365.
arcsec2rad = np.deg2rad(1./3600.)

dd = sl_cosmology.Dang(zd, cosmo=cosmo)
ds = sl_cosmology.Dang(zs, cosmo=cosmo)
dds = sl_cosmology.Dang(zs, zd, cosmo=cosmo)
kpc2rad = 0.001/dd

s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2

ddt = (1. + zd)*dd*ds/dds # time-delay distance

rhoc = sl_cosmology.rhoc(zd, cosmo=cosmo) # critical density of the Universe at z=zd. Halo masses are defined as M200 wrt rhoc.

r200_samp = (10.**lm200_samp*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
rs_samp = r200_samp/c200

arcsec2kpc = arcsec2rad * dd * 1000.

nr3d = 1001
r3d_scalefree = np.logspace(-3., 3., nr3d) # radial grid, from 1/100 to 100 times Reff
deV_rho_scalefree = deV.rho(r3d_scalefree, 1.)
deV_m3d_unitmass = deV.fast_M3d(r3d_scalefree)

nR2d = 101
R2d_scalefree = np.logspace(-3., 2., nR2d)

R2d_fit_grid = np.logspace(0., np.log10(30.), nR2d) # radial grid over which to fit the gNFW model

def get_halo_splines(m200, mstar, reff):
    r200 = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs = r200/c200
    r3d_grid = r3d_scalefree * reff
    R2d_grid = R2d_scalefree * reff

    xmin = 1.01*R2d_grid[0]
    xmax = 0.99*R2d_grid[-1]

    nfw_norm = m200/nfw.M3d(r200, rs)
    nfw_rho = nfw_norm * nfw.rho(r3d_grid, rs)
    nfw_m3d = nfw_norm * nfw.M3d(r3d_grid, rs)

    deV_rho = mstar * deV_rho_scalefree / reff**3
    deV_m3d = mstar * deV_m3d_unitmass

    density_ratio = eaglecontr.contract_density(nfw_rho, deV_rho, nfw_m3d, deV_m3d)/ nfw_rho
    density_ratio[density_ratio < 1.] = 1.
    density_increase = interp1d(r3d_grid, density_ratio, bounds_error=False, \
                                                    fill_value=(density_ratio[0], density_ratio[-1]) )

    contr_Sigma = eaglecontr.projected_density_NFW_contracted(R2d_grid, m200, c200, rs, density_increase)
    contr_Sigma_spline = splrep(R2d_grid, contr_Sigma)
    contr_SigmaR_spline = splrep(R2d_grid, contr_Sigma * R2d_grid)

    return contr_Sigma_spline, contr_SigmaR_spline, R2d_grid

def gnfw_fit(halo_Sigma_spline, m200):

    r200_here = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs_here = r200_here/gnfw_c200

    halo_Sigma_fit = splev(R2d_fit_grid, halo_Sigma_spline)

    def fitfunc(p):
        gammadm, rs = p
        gnfw_norm = m200 / gnfw.M3d(r200_here, rs, gammadm)
        return gnfw_norm * gnfw.fast_Sigma(R2d_fit_grid, rs, gammadm)

    def errfunc(p):
        return (fitfunc(p) - halo_Sigma_fit)/halo_Sigma_fit

    pfit = leastsq(errfunc, (1.4, rs_here))[0]
    gammadm_fit, rs_fit = pfit

    return gammadm_fit, rs_fit

def mdm5_fit(halo_Sigma_spline, mdm5):

    halo_Sigma_fit = splev(R2d_fit_grid, halo_Sigma_spline)

    def fitfunc(gammadm):
        gnfw_norm = mdm5 / gnfw.fast_M2d(5., rs_fixed, gammadm)
        return gnfw_norm * gnfw.fast_Sigma(R2d_fit_grid, rs_fixed, gammadm)

    def errfunc(gammadm):
        return (fitfunc(gammadm) - halo_Sigma_fit)/halo_Sigma_fit

    pfit = leastsq(errfunc, 1.4)[0]
    gammadm_fit = pfit

    return gammadm_fit

eps = 1e-4
def lensconfig(halo_Sigma_spline, halo_SigmaR_spline, mstar, reff, beta_scalefree, xmin, xmax):

    def halo_M2d(x):
        xarr = np.atleast_1d(x)
        out = 0.*xarr
        for i in range(len(xarr)):
            out[i] = 2.*np.pi*splint(0., abs(xarr[i]), halo_SigmaR_spline)
        return out

    def halo_lenspot(x):
        xarr = np.atleast_1d(x)
        out = 0.*xarr
        for i in range(len(xarr)):
            out[i] = 2.*quad(lambda x: splev(x, halo_SigmaR_spline)*np.log(abs(xarr[i])/x), 0., abs(xarr[i]))[0]
        return out

    # defines lensing-related functions
    def alpha(x): 
        # deflection angle (in kpc)
        return (halo_M2d(x) + mstar * deV.fast_M2d(abs(x)/reff)) / np.pi/x/s_cr

    def kappa(x): 
        # dimensionless surface mass density
        return (mstar * deV.Sigma(abs(x), reff) + splev(abs(x), halo_Sigma_spline))/s_cr
    
    def lenspot(x):
        return (halo_lenspot(x) + mstar * deV.fast_lenspot(abs(x), reff)) / s_cr

    def mu_r(x):
        # radial magnification
        return (1. + alpha(x)/x - 2.*kappa(x))**(-1)
    
    def mu_t(x):
        # tangential magnification
        return (1. - alpha(x)/x)**(-1)
    
    def absmu(x):
        # total magnification
        return abs(mu_r(x) * mu_t(x))
    
    def zerofunc(x):
        return alpha(x) - x
        
    if zerofunc(xmin) < 0.:
        rein = 0.
    elif zerofunc(xmax) > 0.:
        rein = np.inf
    else:
        rein = brentq(zerofunc, xmin, xmax)

    def radial_invmag(x):
        return 1. + alpha(x)/x - 2.*kappa(x)

    # finds the radial caustic
    if radial_invmag(xmin)*radial_invmag(xmax) > 0.:
        xradcrit = xmin
    else:
        xradcrit = brentq(radial_invmag, xmin, xmax)

    ycaust = -(xradcrit - alpha(xradcrit))

    # finds the main image for a source on the radial caustic
    imageeq = lambda x: x - alpha(x) - ycaust
    xA_caust = brentq(imageeq, rein, xmax)

    # calculates the asymmetry of the most extreme case
    asymm_caust = (xA_caust + xradcrit)/(xA_caust - xradcrit)

    if asymm_caust < max_asymm:
        beta_max = ycaust
    else:
        def sourcepos_eq(xB_here):
            xA_here = -xB_here * (1. + max_asymm)/(1. - max_asymm)
            return xA_here - alpha(xA_here) - xB_here + alpha(xB_here)
        xB_betamax = brentq(sourcepos_eq, -rein, -xradcrit)
        beta_max = xB_betamax - alpha(xB_betamax)

        """
        def asymmfunc_eq(y):
            imageeq_here = lambda x: x - alpha(x) - y
            xA_here = brentq(imageeq_here, rein, 2.*xA_caust)
            xB_here = brentq(imageeq_here, -rein, -xradcrit)
            return (xA_here + xB_here)/(xA_here - xB_here) - max_asymm
        beta_max = brentq(asymmfunc_eq, xmin, ycaust)
        """
        
    beta = beta_scalefree * beta_max

    # finds the images
    imageeq = lambda x: x - alpha(x) - beta
    if imageeq(xradcrit)*imageeq(xmax) >= 0. or imageeq(-xmax)*imageeq(-xradcrit) >= 0.:
        xA, xB = -np.inf, np.inf
    else:
        xA = brentq(imageeq, rein, xmax)#, xtol=xtol)
        xB = brentq(imageeq, -rein, -xradcrit)#, xtol=xtol)

    rmuA = mu_r(xA)
    rmuB = mu_r(xB)

    dt = ddt*Mpc/c*kpc2rad**2 * (0.5*(xB-beta)**2 - lenspot(xB) - 0.5*(xA-beta)**2 + lenspot(xA))/day

    return xA, xB, beta, beta_max, rmuA/rmuB, rein, xradcrit, dt

for i in range(ngal):

    reff = 10.**lreff_samp[i]

    halo_Sigma_spline, halo_SigmaR_spline, R2d_grid = get_halo_splines(10.**lm200_samp[i], 10.**lmstar_samp[i], reff)
    xmin = 1.01*R2d_grid[0]
    xmax = 0.99*R2d_grid[-1]

    xA, xB, beta, beta_max, rmur, rein, xradcrit, dt = lensconfig(halo_Sigma_spline, halo_SigmaR_spline, 10.**lmstar_samp[i], reff, beta_scalefree_samp[i], xmin, xmax)
    
    lmdm5_samp[i] = np.log10(2.*np.pi*splint(0., 5., halo_SigmaR_spline))

    xA_samp[i] = xA
    xB_samp[i] = xB
    beta_samp[i] = beta
    beta_max_samp[i] = beta_max
    rmur_samp[i] = rmur
    xradcrit_samp[i] = xradcrit
    dt_samp[i] = dt
    rein_samp[i] = rein

    print('%d, Rein: %3.2f kpc, Dt=%3.2f days'%(i, rein, dt))

    gammadm_fit, rs_fit = gnfw_fit(halo_Sigma_spline, 10.**lm200_samp[i])
    gnfwfit_gammadm_samp[i] = gammadm_fit
    gnfwfit_rs_samp[i] = rs_fit

    gammadm_fit = mdm5_fit(halo_Sigma_spline, 10.**lmdm5_samp[i])
    mdm5fit_gammadm_samp[i] = gammadm_fit

tein_samp = rein_samp/arcsec2kpc

# adds observational errors to the radial magnification ratios and time delays
rmur_obs = rmur_samp + rmur_deltas
dt_obs = dt_samp + dt_deltas

output = h5py.File('%s_pop.hdf5'%mockname, 'w')

# individual lens parameters
output.create_dataset('rein', data=rein_samp)
output.create_dataset('tein', data=rein_samp/arcsec2kpc)
output.create_dataset('xradcrit', data=xradcrit_samp)
output.create_dataset('impos', data=np.array((xA_samp, xB_samp)).T)
output.create_dataset('rmur_true', data=rmur_samp)
output.create_dataset('rmur_obs', data=rmur_obs)
output.create_dataset('dt_true', data=dt_samp)
output.create_dataset('dt_obs', data=dt_obs)
output.create_dataset('beta', data=beta_samp)
output.create_dataset('beta_max', data=beta_max_samp)
output.create_dataset('lmsps_true', data=lmsps_samp)
output.create_dataset('lmsps_obs', data=lmsps_obs)
output.create_dataset('lmstar', data=lmstar_samp)
output.create_dataset('laimf', data=laimf_samp)
output.create_dataset('lm200', data=lm200_samp)
output.create_dataset('lmdm5', data=lmdm5_samp)
output.create_dataset('r200', data=r200_samp)
output.create_dataset('rs_nfw', data=rs_samp)
output.create_dataset('gammadm_gnfw', data=gnfwfit_gammadm_samp)
output.create_dataset('rs_gnfw', data=gnfwfit_rs_samp)
output.create_dataset('gammadm_fixedrs', data=mdm5fit_gammadm_samp)
output.create_dataset('lreff', data=lreff_samp)

# fixed parameters
output.attrs['ngal'] = ngal
output.attrs['zd'] = zd
output.attrs['zs'] = zs
output.attrs['c200'] = c200
output.attrs['max_asymm'] = max_asymm
output.attrs['h'] = h
output.attrs['s_cr'] = s_cr
output.attrs['dd'] = dd
output.attrs['ddt'] = ddt

# hyper-parameters
output.attrs['lmsps_mu'] = lmsps_mu
output.attrs['lmsps_sig'] = lmsps_sig
output.attrs['lmsps_err'] = lmsps_err
output.attrs['lmsps_piv'] = lmsps_piv
output.attrs['rmur_err'] = rmur_err
output.attrs['dt_err'] = dt_err
output.attrs['laimf_mu'] = laimf_mu
output.attrs['laimf_sig'] = laimf_sig
output.attrs['lm200_mu'] = lm200_mu
output.attrs['lm200_sig'] = lm200_sig
output.attrs['lm200_beta'] = lm200_beta
output.attrs['lreff_mu'] = lreff_mu
output.attrs['lreff_sig'] = lreff_sig
output.attrs['lreff_beta'] = lreff_beta

