import numpy as np
import pylab
import h5py
from astropy.io import fits as pyfits
from scipy.interpolate import splrep, splev
from labellines import labelLine, labelLines
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18

#xlim = (0.1, 1.)
ylim = (0.085, 1.5)

logscale = True

if logscale:
    leftm = 0.14
else:
    leftm = 0.08

cs_file = h5py.File('smallsource_crosssect.hdf5', 'r')

lfrat_grid = cs_file['lfrat_grid'][()]
nfrat = len(lfrat_grid)

logre_grid = cs_file['logre_grid'][()]
nre = len(logre_grid)

cs_grid = cs_file['cs_grid'][()]
cs_ps = cs_grid[:, 0] # approximation of point-source cross-section
re_max = (cs_ps/np.pi)**0.5
print(re_max)

fig, ax = pylab.subplots(1, 1)#, figsize=(8, 4))
pylab.subplots_adjust(left=leftm, right=0.98, bottom=0.14, top=0.97, wspace=0.)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(nfrat):
    #ax.loglog(10.**logre_grid, cs_grid[i, :]/np.pi, linewidth=2, color=colseq[i], label="$\log{\\frac{f}{\sigma_{\mathrm{sky,\\theta_{\mathrm{Ein}}^{2}}}}} = %2.1f$"%lfrat_grid[i])
    ax.loglog(10.**logre_grid, cs_grid[i, :]/np.pi, linewidth=2, color=colseq[i], label="$\log{f} = %2.1f$"%lfrat_grid[i])
    cs_spline = splrep(10.**logre_grid, cs_grid[i, :]/np.pi, k=1)

ax.tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in')

ax.set_xlabel('$\\theta_{\mathrm{e,s}}/\\theta_{\mathrm{Ein}}$', fontsize=fsize)
ax.set_ylabel('$\sigma_{\mathrm{SL}}/(\pi\\theta_{\mathrm{Ein}}^2)$', fontsize=fsize)

# now finds the detection limit for non-lensed galaxies
nre = 11
logre_grid = np.linspace(-1., 0., nre)
nsigma = 2.

for l in range(nfrat):
    lfrat = lfrat_grid[l]
    sky_rms = 10.**(1. - lfrat) 

    sn_grid = np.zeros(nre)

    for m in range(nre):
        logre = logre_grid[m]

        preamble = 'ftot200_sourceonly_logre%2.1f'%logre

        img = pyfits.open('mockdir/ftot200_sourceonly/'+preamble+'_source.fits')[0].data
        footprint = img > nsigma * sky_rms
        npix_here = footprint.sum()

        signal = img[footprint].sum()
        noise = npix_here**0.5 * sky_rms

        sn_grid[m] = signal/noise

    good = np.isfinite(sn_grid)
    sn_spline = splrep(np.flipud(sn_grid[good]), np.flipud(logre_grid[good]))

    logre_crit = splev(10., sn_spline)

    ax.axvline(10.**logre_crit, linestyle='--', color=colseq[l])

# gets the area within the caustics
afile = h5py.File('../elliptical_point/caustic_areas.hdf5', 'r')
caustic_area = afile['full_cs'][3]

ax.axhline(caustic_area/np.pi, linestyle=':', color='k', label='Bright point source')

#ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])

#ax.loglog(re_max, cs_ps/np.pi, linestyle=':', color='k')

ax.legend(loc = 'lower right', fontsize=fsize, framealpha=1.)

pylab.savefig('../../paper/ell_ext_cs.eps')
pylab.show()


