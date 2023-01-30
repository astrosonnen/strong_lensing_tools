import numpy as np
from sl_profiles import gnfw, deVaucouleurs as deV
from scipy.optimize import brentq
import h5py
import sys
import pylab
from labellines import labelLine, labelLines
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18

reff = 1.
rein = reff
rs_fixed = 10. * reff

s_cr = 1. # I shouldn't need to change this

gammadm_plot = [1., 1.5, 2.]
ngammadm = len(gammadm_plot)
fdm_fixed = 0.5

fdm_plot = [0.2, 0.8]
gammadm_fixed = 1.5
nfdm = len(fdm_plot)

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

xmin = max(deV.rgrid_min*reff, Rfrac_min*rs_fixed)
nx = 1001

leftm = 0.14

fig, ax = pylab.subplots(1, 1, figsize=(6, 6))
pylab.subplots_adjust(left=leftm, right=1.00, bottom=0.12, top=0.98, wspace=0.)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(ngammadm):

    gammadm = gammadm_plot[i]

    mstar_here = np.pi * rein**2 / mstar_einfrac * (1. - fdm_fixed)
    mdm_ein_here = np.pi * rein**2 * fdm_fixed

    gnfw_norm = mdm_ein_here / gnfw.fast_M2d(rein, rs_fixed, gammadm)

    # looks for radial critical curve
    def zerofunc(x):
        return 1./mu_r(x, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    x_rad = brentq(zerofunc, -rein, -0.01)
    beta_rad = x_rad - alpha(x_rad, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    xarr = np.linspace(-1., x_rad, nx)

    beta_arr = xarr - alpha(xarr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)
    mu_arr = mu_r(xarr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff) * mu_t(xarr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    ax.plot(beta_arr, abs(mu_arr), color=colseq[i], label='$f_{\mathrm{DM}} = %2.1f$, $\gamma_{\mathrm{DM}} = %2.1f$'%(fdm_fixed, gammadm))
    ax.axvline(beta_rad, linestyle=':', color=colseq[i])

for i in range(nfdm):

    gammadm = gammadm_fixed

    mstar_here = np.pi * rein**2 / mstar_einfrac * (1. - fdm_plot[i])
    mdm_ein_here = np.pi * rein**2 * fdm_plot[i]

    gnfw_norm = mdm_ein_here / gnfw.fast_M2d(rein, rs_fixed, gammadm)

    # looks for radial critical curve
    def zerofunc(x):
        return 1./mu_r(x, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    x_rad = brentq(zerofunc, -rein, -0.01)
    beta_rad = x_rad - alpha(x_rad, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    xarr = np.linspace(-1., x_rad, nx)

    beta_arr = xarr - alpha(xarr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)
    mu_arr = mu_r(xarr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff) * mu_t(xarr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    ax.plot(beta_arr, abs(mu_arr), label='$f_{\mathrm{DM}} = %2.1f$, $\gamma_{\mathrm{DM}} = %2.1f$'%(fdm_plot[i], gammadm), color=colseq[i+ngammadm])
    ax.axvline(beta_rad, linestyle=':', color=colseq[i+ngammadm])

ax.set_xlabel('$\\beta/\\theta_{\mathrm{Ein}}$', fontsize=fsize)
ax.set_ylabel('$|\mu_2|$', fontsize=fsize)

ax.tick_params(axis='both', which='both', labelsize=fsize, direction='in', top=True, right=True)

ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))

#ax.yaxis.set_major_locator(MultipleLocator(1))
#ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.set_yscale('log')
ax.set_ylim(0.025, 12.)

ax.legend(loc='lower left', fontsize=fsize, framealpha=1.)
#ax.legend(loc='upper right', fontsize=fsize, framealpha=1.)

pylab.savefig('../../paper/composite_fixedap_mu2.eps')
pylab.show()
