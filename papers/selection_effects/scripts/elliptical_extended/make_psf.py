import numpy as np
from astropy.io import fits as pyfits


npix = 7

fwhm = 2.

beta = 5.
alpha = 0.5*fwhm/(2.**(1./beta) - 1.)**0.5

x0 = 3.
y0 = 3.

X, Y = np.meshgrid(np.arange(npix), np.arange(npix))
R = ((X-x0)**2 + (Y-y0)**2)**0.5

psf = (beta - 1.)/(np.pi*alpha)*(1. + (R/alpha)**2)**(-beta)

psf /= psf.sum()

pyfits.PrimaryHDU(psf).writeto('psf.fits', overwrite=True)

