import numpy as np
from astropy.io import fits as pyfits
from scipy.interpolate import splrep, splev
import h5py
import pylab
from matplotlib import rc
rc('text', usetex=True)


colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

# this scripts looks at simulations of exponential profiles with
# different values of the half-light radius and assuming different
# background noise levels. Then, using the same criterion used for
# the detection of multiple images of a strong lens, it finds the
# largest half-light radius for which the galaxy is detected.

ftot = 200.
nre = 11
logre_grid = np.linspace(-1., 0., nre)

nfrat = 4
lfrat_grid = np.linspace(0.8, 2., nfrat)
# frat is the ratio between the source intrinsic flux and the sky rms over an area equal to the square of the Einstein radius.
# In my sims, the Einstein radius is 1 arcsec. Pixels have a 0.05" size.
# Then, the sky rms over a 1 square arcsec is 20 times that over a single pixel
# All simulations have a source intrinsic flux of 200. Then an frat=10 
# corresponds to a sky rms per pixel of 1.

nsigma = 2.

X, Y = np.meshgrid(np.arange(80), np.arange(80))

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

        print(l, m, npix_here, signal/noise)

        sn_grid[m] = signal/noise

    good = np.isfinite(sn_grid)
    sn_spline = splrep(np.flipud(sn_grid[good]), np.flipud(logre_grid[good]))

    logre_crit = splev(10., sn_spline)

    pylab.plot(10.**logre_grid, sn_grid, color=colseq[l])
    pylab.axvline(10.**logre_crit, linestyle='--', color=colseq[l])

pylab.xscale('log')
pylab.show()

