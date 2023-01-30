import numpy as np
from astropy.io import fits as pyfits
from scipy.signal import convolve2d
from lensdet import detect_lens
import h5py


nre = 11
logre_grid = np.linspace(-1., 0., nre)

nsim = 10000

nser = 1.

nfrat = 3
lfrat_grid = np.linspace(1., 2., nfrat)
# frat is the ratio between the source intrinsic flux and the sky rms over an area equal to the square of the Einstein radius.
# In my sims, the Einstein radius is 1 arcsec. Pixels have a 0.05" size.
# Then, the sky rms over a 1 square arcsec is 20 times that over a single pixel
# All simulations have a source intrinsic flux of 200. Then an frat=10 
# corresponds to a sky rms per pixel of 1.

nsigma = 2.

# reads the PSF (because Glafic can't seem to be able to deal with it)
psf = pyfits.open('psf.fits')[0].data

cs_grid = np.zeros((nfrat, nre))

for l in range(nfrat):
    lfrat = lfrat_grid[l]
    sky_rms = 10.**(1. - lfrat) 
    sb_min = nsigma * sky_rms

    for m in range(nre):
        print(l, m)
        logre = logre_grid[m]

        nlens = 0
        islens_sim = np.zeros(nsim, dtype=bool)
        nimages_sim = np.zeros(nsim, dtype=int)

        img_file = h5py.File('mockdir/logre%2.1f_images.hdf5'%logre, 'r')

        rmax = img_file.attrs['rmax']
        source_area = np.pi*rmax**2

        for i in range(nsim):
        
            img = img_file['lens_%04d'%i][()]
            img = convolve2d(img, psf, mode='same')

            res = detect_lens(img, sky_rms, npix_min=1)
            islens_sim[i] = res[0]

        cs_grid[l, m] = islens_sim.sum()/float(nsim) * source_area

output = h5py.File('smallsource_crosssect.hdf5', 'w')

output.create_dataset('lfrat_grid', data=lfrat_grid)
output.create_dataset('logre_grid', data=logre_grid)
output.create_dataset('cs_grid', data=cs_grid)

