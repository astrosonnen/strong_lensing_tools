import numpy as np
from sl_profiles import gnfw, deVaucouleurs as deV
from scipy.interpolate import splrep, splev, splint
import h5py


# lensing cross-section of a composite (deV + gNFW) model,
# as a function of dark matter fraction and inner slope.
# The ratio between the Einstein radius and half-light radius
# is kept constant. This means that Sigma_cr varies implicitly

reff = 1.
rein = reff
rs_fixed = 10. * reff

s_cr = 1. # I shouldn't need to change this

# defines lensing-related functions
def alpha_dm(x, gnfw_norm, rs, gammadm):
    # deflection angle (in kpc)
    return gnfw_norm * gnfw.fast_M2d(abs(x), rs, gammadm) / np.pi/x/s_cr

def alpha_star(x, mstar, reff): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(x, gnfw_norm, rs, gammadm, mstar, reff):
    return alpha_dm(x, gnfw_norm, rs, gammadm) + alpha_star(x, mstar, reff)

def kappa(x, gnfw_norm, rs, gammadm, mstar, reff): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + gnfw_norm * gnfw.fast_Sigma(abs(x), rs, gammadm))/s_cr
   
def mu_r(x, gnfw_norm, rs, gammadm, mstar, reff):
    # radial magnification
    return (1. + alpha(x, gnfw_norm, rs, gammadm, mstar, reff)/x - 2.*kappa(x, gnfw_norm, rs, gammadm, mstar, reff))**(-1)

def mu_t(x, gnfw_norm, rs, gammadm, mstar, reff):
    # tangential magnification
    return (1. - alpha(x, gnfw_norm, rs, gammadm, mstar, reff)/x)**(-1)

dx = 0.0001
dx_search = 0.001

Rfrac_min = gnfw.R_grid[0]
Rfrac_max = gnfw.R_grid[-1]

mstar_einfrac = deV.M2d(rein, reff)

# with s_cr = 1, the total mass enclosed within the Einstein radius must be
# equal to s_cr * np.pi * rein**2

ndms = 6
dms_grid = np.linspace(-3., 2., ndms)

fdm_fixed = 0.5
gammadm_fixed = 1.5

gammadm_min = 0.4 
gammadm_max = 2.2 

ngammadm = 91
gammadm_grid = np.linspace(gammadm_min, gammadm_max, ngammadm)

fdm_min = 0.1
fdm_max = 0.9
nfdm = 81
fdm_grid = np.linspace(fdm_min, fdm_max, nfdm)

grids_file = h5py.File('composite_rein%2.1freff_grid.hdf5'%rein, 'w')

grids_file.attrs['rein'] = rein
grids_file.attrs['reff'] = reff
grids_file.attrs['rs'] = rs_fixed
grids_file.attrs['fdm_fixed'] = fdm_fixed
grids_file.attrs['gammadm_fixed'] = gammadm_fixed
grids_file.create_dataset('fdm_grid', data=fdm_grid)
grids_file.create_dataset('gammadm_grid', data=gammadm_grid)
grids_file.create_dataset('dms_grid', data=dms_grid)

cs_vs_gammadm = np.zeros((ngammadm, ndms))
cs_vs_fdm = np.zeros((nfdm, ndms))

xmin = max(deV.rgrid_min*reff, Rfrac_min*rs_fixed)

def get_cs(gammadm, fdm, dms):

    mstar_here = np.pi * rein**2 / mstar_einfrac * (1. - fdm)
    mdm_ein_here = np.pi * rein**2 * fdm

    gnfw_norm = mdm_ein_here / gnfw.fast_M2d(rein, rs_fixed, gammadm)

    xcimg_max_search = np.arange(-rein+dx_search, -xmin, dx_search)
    xcimg_indarr = np.arange(len(xcimg_max_search))
    mu_r_search = mu_r(xcimg_max_search, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)
    mu_t_search = mu_t(xcimg_max_search, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    cimg = mu_r_search > 0. # only looks outside of the radial critical curve
    mu_search = abs(mu_r_search * mu_t_search)
    minmu_here = 10.**(dms/2.5)

    beta_arr = xcimg_max_search[cimg] - alpha(xcimg_max_search[cimg], gnfw_norm, rs_fixed, gammadm, mstar_here, reff)
    beta_arr[0] = 0.

    cs_integrand = 2.*np.pi*beta_arr
    cs_integrand[mu_search[cimg] < minmu_here] = 0.

    cs_spline = splrep(beta_arr, cs_integrand, k=1)
    cs_int = splint(0., beta_arr[-1], cs_spline)

    return cs_int

for i in range(ngammadm):
    print(i)
    for j in range(ndms):
        cs_vs_gammadm[i, j] = get_cs(gammadm_grid[i], fdm_fixed, dms_grid[j])

for i in range(nfdm):
    print(i)
    for j in range(ndms):
        cs_vs_fdm[i, j] = get_cs(gammadm_fixed, fdm_grid[i], dms_grid[j])

grids_file.create_dataset('cs_vs_gammadm', data=cs_vs_gammadm)
grids_file.create_dataset('cs_vs_fdm', data=cs_vs_fdm)


