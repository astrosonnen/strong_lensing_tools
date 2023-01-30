import numpy as np
from sl_profiles import gnfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun
import sl_cosmology
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
import h5py


# lensing cross-section of a composite (deV + gNFW) model,
# as a function of: half-light radius, stellar mass, dark matter 
# fraction and inner slope.

reff_fixed = 7.
lmstar_fixed = 11.5
lmdm5_fixed = 11.
gammadm_fixed = 1.5
rs_fixed = 100.

zd_fixed = 0.3
zs_fixed = 1.5

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

dd = sl_cosmology.Dang(zd_fixed)
ds = sl_cosmology.Dang(zs_fixed)
dds = sl_cosmology.Dang(zs_fixed, zd_fixed)

arcsec2kpc = arcsec2rad * dd * 1000.

s_cr_fixed = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2

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

ndms = 6
dms_grid = np.linspace(-3., 2., ndms)

gammadm_min = 0.4 
gammadm_max = 2.2 

#ngammadm = 19
ngammadm = 10
gammadm_grid = np.linspace(gammadm_min, gammadm_max, ngammadm)

lmdm5_min = 10.5
lmdm5_max = 11.5
nmdm5 = 9
lmdm5_grid = np.linspace(lmdm5_min, lmdm5_max, nmdm5)

lmstar_min = 11.
lmstar_max = 12.
nmstar = 11
lmstar_grid = np.linspace(lmstar_min, lmstar_max, nmstar)

lreff_min = 0.5
lreff_max = 1.3
nreff = 9
lreff_grid = np.linspace(lreff_min, lreff_max, nreff)

# calculates dark matter fractions within Re, given lmdm5, lmstar and reff
gnfw_norm_grid = 10.**lmdm5_grid / gnfw.M2d(5., rs_fixed, gammadm_fixed)
mdm_reff_grid = gnfw_norm_grid * gnfw.M2d(reff_fixed, rs_fixed, gammadm_fixed)
fdm_grid = mdm_reff_grid/(mdm_reff_grid + 0.5*10.**lmstar_fixed)

grids_file = h5py.File('composite_physical_grid.hdf5', 'w')

grids_file.attrs['lmstar_fixed'] = lmstar_fixed
grids_file.attrs['reff_fixed'] = reff_fixed
grids_file.attrs['rs_fixed'] = rs_fixed
grids_file.attrs['lmdm5_fixed'] = lmdm5_fixed
grids_file.attrs['gammadm_fixed'] = gammadm_fixed
grids_file.create_dataset('lreff_grid', data=lreff_grid)
grids_file.create_dataset('lmstar_grid', data=lmstar_grid)
grids_file.create_dataset('lmdm5_grid', data=lmdm5_grid)
grids_file.create_dataset('fdm_grid', data=fdm_grid)
grids_file.create_dataset('gammadm_grid', data=gammadm_grid)
grids_file.create_dataset('dms_grid', data=dms_grid)

rein_vs_gammadm = np.zeros(ngammadm)
cs_vs_gammadm = np.zeros((ngammadm, ndms))

rein_vs_lmdm5 = np.zeros(nmdm5)
cs_vs_lmdm5 = np.zeros((nmdm5, ndms))

rein_vs_lmstar = np.zeros(nmstar)
cs_vs_lmstar = np.zeros((nmstar, ndms))

rein_vs_lreff = np.zeros(nreff)
cs_vs_lreff = np.zeros((nreff, ndms))

def get_rein_kpc(mstar, reff, mdm5, gammadm, s_cr):

    xmin = max(deV.rgrid_min*reff, Rfrac_min*rs_fixed)

    gnfw_norm = mdm5 / gnfw.fast_M2d(5., rs_fixed, gammadm)

    def zerofunc(x):
        return alpha(x, gnfw_norm, rs_fixed, gammadm, mstar, reff, s_cr) - x

    rein_kpc = brentq(zerofunc, 0.1, 100.)
    return rein_kpc

def get_cs(mstar, reff, mdm5, gammadm, rein_kpc, s_cr, dms):

    xmin = max(deV.rgrid_min*reff, Rfrac_min*rs_fixed)

    gnfw_norm = mdm5 / gnfw.fast_M2d(5., rs_fixed, gammadm)

    xcimg_max_search = np.arange(-rein_kpc+dx_search, -xmin, dx_search)
    xcimg_indarr = np.arange(len(xcimg_max_search))
    mu_r_search = mu_r(xcimg_max_search, gnfw_norm, rs_fixed, gammadm, mstar, reff, s_cr)
    mu_t_search = mu_t(xcimg_max_search, gnfw_norm, rs_fixed, gammadm, mstar, reff, s_cr)

    cimg = mu_r_search > 0. # only looks outside of the radial critical curve
    mu_search = abs(mu_r_search * mu_t_search)
    minmu_here = 10.**(dms/2.5)

    beta_arr = xcimg_max_search[cimg] - alpha(xcimg_max_search[cimg], gnfw_norm, rs_fixed, gammadm, mstar, reff, s_cr)
    beta_arr[0] = 0.

    cs_integrand = 2.*np.pi*beta_arr
    cs_integrand[mu_search[cimg] < minmu_here] = 0.

    cs_spline = splrep(beta_arr, cs_integrand, k=1)
    cs_kpc2 = splint(0., beta_arr[-1], cs_spline)

    return cs_kpc2

for i in range(nmstar):
    print(i)
    rein_kpc = get_rein_kpc(10.**lmstar_grid[i], reff_fixed, 10.**lmdm5_fixed, gammadm_fixed, s_cr_fixed)
    rein_vs_lmstar[i] = rein_kpc / arcsec2kpc
    for j in range(ndms):
        cs_vs_lmstar[i, j] = get_cs(10.**lmstar_grid[i], reff_fixed, 10.**lmdm5_fixed, gammadm_fixed, rein_kpc, s_cr_fixed, dms_grid[j]) / arcsec2kpc**2

for i in range(nreff):
    print(i)
    rein_kpc = get_rein_kpc(10.**lmstar_fixed, 10.**lreff_grid[i], 10.**lmdm5_fixed, gammadm_fixed, s_cr_fixed)
    rein_vs_lreff[i] = rein_kpc / arcsec2kpc
    for j in range(ndms):
        cs_vs_lreff[i, j] = get_cs(10.**lmstar_fixed, 10.**lreff_grid[i], 10.**lmdm5_fixed, gammadm_fixed, rein_kpc, s_cr_fixed, dms_grid[j]) / arcsec2kpc**2

for i in range(nmdm5):
    print(i)
    rein_kpc = get_rein_kpc(10.**lmstar_fixed, reff_fixed, 10.**lmdm5_grid[i], gammadm_fixed, s_cr_fixed)
    rein_vs_lmdm5[i] = rein_kpc / arcsec2kpc
    for j in range(ndms):
        cs_vs_lmdm5[i, j] = get_cs(10.**lmstar_fixed, reff_fixed, 10.**lmdm5_grid[i], gammadm_fixed, rein_kpc, s_cr_fixed, dms_grid[j]) / arcsec2kpc**2

for i in range(ngammadm):
    print(i)
    rein_kpc = get_rein_kpc(10.**lmstar_fixed, reff_fixed, 10.**lmdm5_fixed, gammadm_grid[i], s_cr_fixed)
    rein_vs_gammadm[i] = rein_kpc / arcsec2kpc
    for j in range(ndms):
        cs_vs_gammadm[i, j] = get_cs(10.**lmstar_fixed, reff_fixed, 10.**lmdm5_fixed, gammadm_grid[i], rein_kpc, s_cr_fixed, dms_grid[j]) / arcsec2kpc**2

grids_file.create_dataset('cs_vs_lmstar', data=cs_vs_lmstar)
grids_file.create_dataset('rein_vs_lmstar', data=rein_vs_lmstar)

grids_file.create_dataset('cs_vs_lreff', data=cs_vs_lreff)
grids_file.create_dataset('rein_vs_lreff', data=rein_vs_lreff)

grids_file.create_dataset('cs_vs_gammadm', data=cs_vs_gammadm)
grids_file.create_dataset('rein_vs_gammadm', data=rein_vs_gammadm)

grids_file.create_dataset('cs_vs_lmdm5', data=cs_vs_lmdm5)
grids_file.create_dataset('rein_vs_lmdm5', data=rein_vs_lmdm5)

