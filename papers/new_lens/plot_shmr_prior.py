import numpy as np
import pylab
import h5py
from lensingfuncs import *
from halo_pars import shmr, sigmalogms, invcumhmf_spline, deltaVir, rho_c, cvir_func
from masssize_pars import *
from lenspars import s_cr, arcsec2kpc
from plotters import probcontour
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18

# draws prior from the halo mass function and SHMR (no lensing)
nsamp = 100000

lmvir_samp = splev(np.random.rand(nsamp), invcumhmf_spline)
rvir_samp = (10.**lmvir_samp*3./deltaVir/(4.*np.pi)/rho_c)**(1./3.)
cvir_samp = cvir_func(lmvir_samp)
rs_samp = rvir_samp/cvir_samp
nfw_norm_samp = 10.**lmvir_samp / nfw.M3d(rvir_samp, rs_samp)

lmsps_samp = np.random.normal(shmr(lmvir_samp), sigmalogms, nsamp)
lreff_samp = np.random.normal(masssize_mu + masssize_beta * (lmsps_samp - masssize_mpiv), masssize_sigma, nsamp)

laimf = 0.1
lmstar_samp = lmsps_samp + laimf

cs_samp = np.zeros(nsamp)
tein_samp = np.zeros(nsamp)

for i in range(nsamp):
    reff_kpc = 10.**lreff_samp[i]
    mstar = 10.**lmstar_samp[i]

    # computes Einstein radius
    rein_here = get_rein_kpc(mstar, reff_kpc, nfw_norm_samp[i], rs_samp[i], s_cr)
    tein = rein_here / arcsec2kpc
    tein_samp[i] = tein

    # computes radial caustic and critical curve (if any)
    xrad_kpc, radcaust_kpc = get_radcaust(mstar, reff_kpc, nfw_norm_samp[i], rs_samp[i], s_cr, rein_here)

    # computes the cross-section
    cs_samp[i] = get_crosssect(mstar, reff_kpc, nfw_norm_samp[i], rs_samp[i], s_cr, rein_here, xrad_kpc, arcsec2kpc)

fig, ax = pylab.subplots()
pylab.subplots_adjust(left=0.13, bottom=0.13, right=1., top=1.)

probcontour(lmvir_samp, lmsps_samp, style='solid', color='b')
probcontour(lmvir_samp, lmsps_samp, weights=cs_samp, style='lines', color='r', linewidths=2)

ax.set_xlabel('$\log{M_{\mathrm{h}}}$', fontsize=fsize)
ax.set_ylabel('$\log{M_*^{\mathrm{(sps)}}}$', fontsize=fsize)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.tick_params(which='both', axis='both', direction='in', labelsize=fsize)

pylab.show()



