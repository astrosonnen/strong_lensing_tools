import numpy as np
import pylab
from sl_profiles import deVaucouleurs as deV, gnfw
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy.optimize import brentq
rc('text', usetex=True)


fsize = 18

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

reff = 1.
rein = reff
theta_E = rein
rs_fixed = 10. * reff

s_cr = 1. # I shouldn't need to change this

gammadm_plot = [1., 1.5, 2.]
ngamma = len(gammadm_plot)
fdm_fixed = 0.5
gammadm_fixed = 1.5
fdm_plot = [0.2, 0.8]
nfdm = len(fdm_plot)

Rfrac_min = gnfw.R_grid[0]
Rfrac_max = gnfw.R_grid[-1]

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

hwidth = 3.
t_finearr = np.linspace(-hwidth, hwidth, 1001)
t_arr = np.linspace(-hwidth, hwidth, 201)

tick_height = 0.1
xlabel_height = -0.5
xlabel_shift = -0.1

ymax = 1.5

fig, ax = pylab.subplots(figsize=(5.5, 5.5))

pylab.subplots_adjust(left=0., right=1., bottom=0., top=1.)

mstar_einfrac = deV.M2d(rein, reff)

# with s_cr = 1, the total mass enclosed within the Einstein radius must be
# equal to s_cr * np.pi * rein**2

xmin = max(deV.rgrid_min*reff, Rfrac_min*rs_fixed)

for i in range(ngamma):

    gammadm = gammadm_plot[i]

    mstar_here = np.pi * rein**2 / mstar_einfrac * (1. - fdm_fixed)
    mdm_ein_here = np.pi * rein**2 * fdm_fixed

    gnfw_norm = mdm_ein_here / gnfw.fast_M2d(rein, rs_fixed, gammadm)

    ax.plot(t_finearr, t_finearr - alpha(t_finearr, gnfw_norm, rs_fixed, gammadm, mstar_here, reff), label='$f_{\mathrm{DM}} = %2.1f$, $\gamma_{\mathrm{DM}}=%2.1f$'%(fdm_fixed, gammadm), color=colseq[i])

    # looks for radial critical curve
    def zerofunc(x):
        return 1./mu_r(x, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)

    x_rad = brentq(zerofunc, -rein, -0.01)
    beta_rad = x_rad - alpha(x_rad, gnfw_norm, rs_fixed, gammadm, mstar_here, reff)
    ax.axhline(beta_rad, linestyle=':', color=colseq[i])
    ax.axhline(-beta_rad, linestyle=':', color=colseq[i])

for i in range(nfdm):

    fdm = fdm_plot[i]

    mstar_here = np.pi * rein**2 / mstar_einfrac * (1. - fdm)
    mdm_ein_here = np.pi * rein**2 * fdm

    gnfw_norm = mdm_ein_here / gnfw.fast_M2d(rein, rs_fixed, gammadm_fixed)

    ax.plot(t_finearr, t_finearr - alpha(t_finearr, gnfw_norm, rs_fixed, gammadm_fixed, mstar_here, reff), label='$f_{\mathrm{DM}}=%2.1f$, $\gamma_{\mathrm{DM}}=%2.1f$'%(fdm, gammadm_fixed), color=colseq[i+ngamma])

    # looks for radial critical curve
    def zerofunc(x):
        return 1./mu_r(x, gnfw_norm, rs_fixed, gammadm_fixed, mstar_here, reff)

    x_rad = brentq(zerofunc, -rein, -0.01)
    beta_rad = x_rad - alpha(x_rad, gnfw_norm, rs_fixed, gammadm_fixed, mstar_here, reff)
    ax.axhline(beta_rad, linestyle=':', color=colseq[i+ngamma])
    ax.axhline(-beta_rad, linestyle=':', color=colseq[i+ngamma])

#ax.set_aspect(1.)
pylab.arrow(t_arr[0], 0, t_arr[-1]-t_arr[0], 0, length_includes_head=False, head_width=0.15, color='k')
pylab.arrow(0, t_arr[0], 0, t_arr[-1]-t_arr[0]-1., length_includes_head=False, head_width=0.15, color='k')

pylab.text(-1.2, 2., '$\\theta - \\alpha(\\theta)$', fontsize=fsize)
pylab.text(2.8, xlabel_height, '$\\theta$', fontsize=fsize)

pylab.plot([theta_E, theta_E], [-tick_height, tick_height], color='k')
pylab.text(theta_E+xlabel_shift, xlabel_height, '$\\theta_E$', fontsize=fsize)

pylab.plot([-theta_E, -theta_E], [-tick_height, tick_height], color='k')
pylab.text(-theta_E-0.3, xlabel_height, '$-\\theta_E$', fontsize=fsize)
pylab.legend(loc='lower right', fontsize=fsize)

pylab.axis('off')
pylab.savefig('../../figures/composite_fixedap_scheme.eps')
pylab.show()

