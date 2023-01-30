import numpy as np
from scipy.interpolate import splrep, splev, splint, interp1d
from scipy.stats import poisson, beta
from scipy.optimize import leastsq, brentq
import sl_cosmology
from sl_profiles import deVaucouleurs as deV, gnfw, nfw, eaglecontr
import h5py
import ndinterp
from simpars import *


seedno = 0
np.random.seed(seedno)

sky_area_deg2 = 1000.
sky_area = sky_area_deg2 * np.deg2rad(1.)**2

modelname = 'lowscatter_%dsqdeg'%sky_area_deg2
sigma_h = 0.1
sigma_sps = 0.05

print('Sky area: %d square degrees'%sky_area_deg2)
print('Expected number of foreground galaxies: %2.1e'%(ngal_1Mpc3 * fourpi_volume * sky_area / (4.*np.pi)))
# draws the number of galaxies
nsamp = poisson.rvs(ngal_1Mpc3 * fourpi_volume * sky_area / (4.*np.pi), 1)
print('%d galaxies drawn'%nsamp)

lmobs_samp = splev(np.random.rand(nsamp), invcum_lmobs_spline)
z_samp = splev(np.random.rand(nsamp), invcum_z_spline)

rhoc_samp = sl_cosmology.rhoc(z_samp)

lasps_samp = np.random.normal(mu_sps, sigma_sps, nsamp)
lmstar_samp = lmobs_samp + lasps_samp

lreff_samp = mu_R + beta_R * (lmobs_samp - lmobs_piv) + np.random.normal(0., sigma_R, nsamp)

dlreff_samp = lreff_samp - (mu_R + beta_R * (lmstar_samp - mu_sps - lmobs_piv))

dlm200_samp = np.random.normal(0., sigma_h, nsamp)
lm200_samp = mu_h + beta_h * (lmstar_samp - lmstar_piv) + dlm200_samp
r200_samp = (10.**lm200_samp*3./200./(4.*np.pi)/rhoc_samp)**(1./3.) * 1000.

q_samp = beta.rvs(alpha_q, beta_q, size=nsamp)

# uses the grid built previously to get the parameters of the gnfw profile

# loads the gnfw parameters grid
gnfwpar_file = h5py.File('gnfwpar_grid.hdf5', 'r')

lmstar_grid = gnfwpar_file['lmstar_grid'][()]
nmstar = len(lmstar_grid)

dlreff_grid = gnfwpar_file['dlreff_grid'][()]
nreff = len(dlreff_grid)

dlm200_grid = gnfwpar_file['dlm200_grid'][()]
nm200 = len(dlm200_grid)

z_grid = gnfwpar_file['z_grid'][()]
nz = len(z_grid)

rs_grid = gnfwpar_file['rs_grid'][()]
gammadm_grid = gnfwpar_file['gammadm_grid'][()]

axes = {0: splrep(lmstar_grid, np.arange(nmstar)), 1: splrep(dlreff_grid, np.arange(nreff)), 2: splrep(dlm200_grid, np.arange(nm200)), 3: splrep(z_grid, np.arange(nz))}

gammadm_interp = ndinterp.ndInterp(axes, gammadm_grid, order=3)
rs_interp = ndinterp.ndInterp(axes, rs_grid, order=3)

point = np.array([lmstar_samp, dlreff_samp, dlm200_samp, z_samp]).T

gammadm_samp = gammadm_interp.eval(point)
rs_samp = rs_interp.eval(point)

good = (dlm200_samp > dlm200_grid[0]) & (dlm200_samp < dlm200_grid[-1]) & (dlreff_samp > dlreff_grid[0]) & (dlreff_samp < dlreff_grid[-1])
bad = np.logical_not(good)

nbad = bad.sum()
bad_indices = np.arange(nsamp)[bad]

# the galaxies that fall outside of the grid are treated separately

# prepares stuff for the contracted profile
nr3d = 1001
r3d_scalefree = np.logspace(-3., 3., nr3d) # radial grid, from 1/100 to 100 times Reff
deV_rho_scalefree = deV.rho(r3d_scalefree, 1.)
deV_m3d_unitmass = deV.fast_M3d(r3d_scalefree)

nR2d = 101
R2d_scalefree = np.logspace(-3., 2., nR2d)

R2d_fit_grid = np.logspace(0., np.log10(30.), nR2d) # radial grid over which to fit the gNFW model

def get_halo_splines(m200, mstar, reff, rhoc):
    r200 = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs = r200/c200_0
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

    contr_Sigma = eaglecontr.projected_density_NFW_contracted(R2d_grid, m200, c200_0, rs, density_increase)
    contr_Sigma_spline = splrep(R2d_grid, contr_Sigma)
    contr_SigmaR_spline = splrep(R2d_grid, contr_Sigma * R2d_grid)

    return contr_Sigma_spline, contr_SigmaR_spline, R2d_grid

def gnfw_fit(halo_Sigma_spline, m200, rhoc):

    r200_here = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.

    halo_Sigma_fit = splev(R2d_fit_grid, halo_Sigma_spline)

    def fitfunc(p):
        gammadm, rs = p
        gnfw_norm = m200 / gnfw.M3d(r200_here, rs, gammadm)
        return gnfw_norm * gnfw.fast_Sigma(R2d_fit_grid, rs, gammadm)

    def errfunc(p):
        return (fitfunc(p) - halo_Sigma_fit)/halo_Sigma_fit

    pfit = leastsq(errfunc, (1.4, r200_here/c200_0))[0]
    gammadm_fit, rs_fit = pfit

    return gammadm_fit, rs_fit

# loops over the galaxies, calculates the gnfw-equivalent parameters
for n in range(nbad):
    i = bad_indices[n]
    print(i)
    Sigma_spline, SigmaR_spline, R2d_grid = get_halo_splines(10.**lm200_samp[i], 10.**lmstar_samp[i], 10.**lreff_samp[i], rhoc_samp[i])
    gammadm, rs = gnfw_fit(Sigma_spline, 10.**lm200_samp[i], rhoc_samp[i])

    rs_samp[i] = rs
    gammadm_samp[i] = gammadm

# calculates the projected dark matter mass enclosed within 5kpc
gnfw_norm_samp = 10.**lm200_samp / gnfw.fast_M3d(r200_samp, rs_samp, gammadm_samp)
lmdm5_samp = np.log10(gnfw_norm_samp * gnfw.fast_M2d(5., rs_samp, gammadm_samp))

# calculates Einstein radii and radial caustic size (in axisymmetric case)

tein_samp = np.zeros(nsamp)
tcaust_samp = np.zeros(nsamp)

# defines lensing-related functions
def alpha_dm(x, gnfw_norm, rs, gammadm, s_cr):
    # deflection angle (in kpc)
    return gnfw_norm * gnfw.fast_M2d(abs(x), rs, gammadm) / np.pi/x/s_cr

def alpha_star(x, mstar, reff, s_cr): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr):
    return alpha_dm(x, gnfw_norm, rs, gammadm, s_cr) + alpha_star(x, mstar, reff, s_cr)

def kappa(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + gnfw_norm * gnfw.fast_Sigma(abs(x), rs, gammadm))/s_cr
   
def mu_r(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr):
    # radial magnification
    return (1. + alpha(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr)/x - 2.*kappa(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr))**(-1)

def mu_t(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr):
    # tangential magnification
    return (1. - alpha(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr)/x)**(-1)

dx = 0.0001
dx_search = 0.001

Rfrac_min = gnfw.R_grid[0]
Rfrac_max = gnfw.R_grid[-1]

def get_rein_kpc(mstar, reff, gnfw_norm, rs, gammadm, s_cr):

    xmin = max(deV.rgrid_min*reff, Rfrac_min*rs)

    def zerofunc(x):
        return alpha(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr) - x

    rein_kpc = brentq(zerofunc, 0.01, 100.)
    return rein_kpc

def get_radcaust(mstar, reff, gnfw_norm, rs, gammadm, s_cr, rein_kpc):

    def zerofunc(x):
        return 1./mu_r(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr)

    xmin = max(deV.rgrid_min*reff, Rfrac_min*rs)
    
    if zerofunc(-xmin) > 0.:
        xrad = xmin
    else:
        xrad = -brentq(zerofunc, -rein_kpc, -xmin)

    radcaust_kpc = -(xrad - alpha(xrad, gnfw_norm, rs, gammadm, mstar, reff, s_cr))
    return xrad, radcaust_kpc

for i in range(nsamp):

    s_cr_here = splev(z_samp[i], s_cr_spline)
    arcsec2kpc = np.deg2rad(1./3600.) * splev(z_samp[i], dd_spline) * 1000.

    rein_kpc = get_rein_kpc(10.**lmstar_samp[i], 10.**lreff_samp[i], gnfw_norm_samp[i], rs_samp[i], gammadm_samp[i], s_cr_here)
    xrad_kpc, radcaust_kpc = get_radcaust(10.**lmstar_samp[i], 10.**lreff_samp[i], gnfw_norm_samp[i], rs_samp[i], gammadm_samp[i], s_cr_here, rein_kpc)

    tein_samp[i] = rein_kpc / arcsec2kpc
    tcaust_samp[i] = radcaust_kpc / arcsec2kpc

output_file = h5py.File('%s_galaxies.hdf5'%modelname, 'w')

output_file.attrs['nsamp'] = nsamp
output_file.attrs['sigma_h'] = sigma_h
output_file.attrs['sigma_sps'] = sigma_sps
output_file.attrs['seedno'] = seedno

output_file.create_dataset('z', data=z_samp)
output_file.create_dataset('lmobs', data=lmobs_samp)
output_file.create_dataset('lmstar', data=lmstar_samp)
output_file.create_dataset('lasps', data=lasps_samp)
output_file.create_dataset('lm200', data=lm200_samp)
output_file.create_dataset('gnfw_norm', data=gnfw_norm_samp)
output_file.create_dataset('rhoc', data=rhoc_samp)
output_file.create_dataset('lmdm5', data=lmdm5_samp)
output_file.create_dataset('r200', data=r200_samp)
output_file.create_dataset('lreff', data=lreff_samp)
output_file.create_dataset('tein', data=tein_samp)
output_file.create_dataset('tcaust', data=tcaust_samp)
output_file.create_dataset('q', data=q_samp)
output_file.create_dataset('rs', data=rs_samp)
output_file.create_dataset('gammadm', data=gammadm_samp)

