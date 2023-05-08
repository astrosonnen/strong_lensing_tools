import numpy as np
from scipy.interpolate import splrep, splev, splint, interp1d
from scipy.stats import poisson, beta
from scipy.optimize import leastsq, brentq
import sl_cosmology
from sl_profiles import deVaucouleurs as deV, gnfw
import h5py
import ndinterp
from simpars import *


seedno = 0
np.random.seed(seedno)

sky_area_deg2 = 1000.
sky_area = sky_area_deg2 * np.deg2rad(1.)**2

modelname = 'gnfwdev_%dsqdeg'%sky_area_deg2

# Setting values for the dark matter and SPS mismatch parameters.
# The slope and normalization of the SHMR is defined in simpars.py
sigma_h = 0.2
sigma_sps = 0.
mu_gammadm = 1.4
sigma_gammadm = 0.

print('Sky area: %d square degrees'%sky_area_deg2)
print('Expected number of foreground galaxies: %2.1e'%(ngal_1Mpc3 * fourpi_volume * sky_area / (4.*np.pi)))
# draws the number of galaxies
nsamp = poisson.rvs(ngal_1Mpc3 * fourpi_volume * sky_area / (4.*np.pi), 1)
print('%d galaxies drawn'%nsamp)

lmobs_samp = splev(np.random.rand(nsamp), invcum_lmobs_spline)
z_samp = splev(np.random.rand(nsamp), invcum_z_spline)

rhoc_samp = sl_cosmology.rhoc(z_samp)

lasps_samp = mu_sps + sigma_sps * np.random.normal(0., 1., nsamp)
lmstar_samp = lmobs_samp + lasps_samp

lreff_samp = mu_R + beta_R * (lmobs_samp - lmobs_piv) + np.random.normal(0., sigma_R, nsamp)

lm200_samp = mu_h + beta_h * (lmstar_samp - lmstar_piv) + sigma_h * np.random.normal(0., 1., nsamp)
r200_samp = (10.**lm200_samp*3./200./(4.*np.pi)/rhoc_samp)**(1./3.) * 1000.
c200_samp = c200_0 * np.ones(nsamp) # YOU MIGHT WANT TO CHANGE THIS
rs_samp = r200_samp / c200_samp

gammadm_samp = mu_gammadm + sigma_gammadm * np.random.normal(0., 1., nsamp)

q_samp = beta.rvs(alpha_q, beta_q, size=nsamp)

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
output_file.create_dataset('c200', data=c200_samp)
output_file.create_dataset('r200', data=r200_samp)
output_file.create_dataset('lreff', data=lreff_samp)
output_file.create_dataset('tein', data=tein_samp)
output_file.create_dataset('tcaust', data=tcaust_samp)
output_file.create_dataset('q', data=q_samp)
output_file.create_dataset('rs', data=rs_samp)
output_file.create_dataset('gammadm', data=gammadm_samp)

