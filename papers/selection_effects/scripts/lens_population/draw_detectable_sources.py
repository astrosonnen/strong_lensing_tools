import numpy as np
from simpars import *
from sl_profiles import sersic
import h5py
import os
from scipy.signal import convolve2d
from scipy.special import gamma as gfunc
from astropy.io import fits as pyfits
from skimage import measure
import sys


np.random.seed(20)

ndraw = 40000

# Skills simulation (not included in this repository, contact Shun-Sheng Li)
sourcecat = pyfits.open('skills_sourceonly_zcut.fits')[1].data

psf = pyfits.open('psf.fits')[0].data
pix_arcsec = 0.1
sb_min = nsigma_pixdet * sky_rms

nsource_tot = len(sourcecat)

sourceind = np.arange(nsource_tot, dtype=int)
# shuffles source catalog (it's originally ranked by redshift)
np.random.shuffle(sourceind)

sreff_draw = sourcecat['Re_arcsec_CM'][sourceind[:ndraw]]
nser_draw = sourcecat['sersic_n_CM'][sourceind[:ndraw]]
sq_draw = sourcecat['axis_ratio_CM'][sourceind[:ndraw]]
smag_draw = sourcecat['i_SDSS_apparent_corr'][sourceind[:ndraw]]
zs_draw = sourcecat['zobs'][sourceind[:ndraw]]
spa_draw = sourcecat['PA_random'][sourceind[:ndraw]]

f = open('preamble.input', 'r')
prelines = f.readlines()
f.close()

outlines = prelines.copy()

for i in range(ndraw):
    if sq_draw[i] > 0. and sq_draw[i] < 1.:
        ind = sourceind[i]
        outlines.append('reset_par prefix simdir/source_%08d\n'%ind)
        outlines.append('reset_extend 1 1 %f\n'%zs_draw[i])
    
        ftot = 10.**(-2./5.*(smag_draw[i] - zeropoint))
        I0 = ftot/(2.*np.pi*(sreff_draw[i]/pix_arcsec)**2*nser_draw[i]/sersic.b(nser_draw[i])**(2*nser_draw[i])*gfunc(2.*nser_draw[i]))
    
        outlines.append('reset_extend 1 2 %f\n'%I0)
        outlines.append('reset_extend 1 5 %f\n'%(1. - sq_draw[i]))
        outlines.append('reset_extend 1 6 %f\n'%spa_draw[i])
        outlines.append('reset_extend 1 7 %f\n'%sreff_draw[i])
        outlines.append('reset_extend 1 8 %f\n'%nser_draw[i])
    
        outlines.append('writeimage_ori\n')
        outlines.append('\n')

outlines.append('quit\n')

f = open('unlensed.input', 'w')
f.writelines(outlines)
f.close()

os.system('/Users/alessandro/glafic unlensed.input')

detected = np.zeros(ndraw, dtype=bool)

for i in range(ndraw):
    if sq_draw[i] > 0. and sq_draw[i] < 1.:
        ind = sourceind[i]
    
        img = pyfits.open('simdir/source_%08d_source.fits'%ind)[0].data
        img_wseeing = convolve2d(img, psf, mode='same')
    
        footprint = img_wseeing > sb_min
    
        labels = measure.label(footprint)
        nreg = labels.max()
        npix_tmp = (labels==1).sum()
        signal = img[labels==1].sum()
        noise = npix_tmp**0.5 * sky_rms
        img_sn = signal/noise
        if img_sn >= 10. and npix_tmp >= npix_min:
            detected[i] = True

output_file = h5py.File('detectable_sources.hdf5', 'w')
output_file.create_dataset('sreff', data=sreff_draw[detected])
output_file.create_dataset('smag', data=smag_draw[detected])
output_file.create_dataset('zs', data=zs_draw[detected])
output_file.create_dataset('sq', data=sq_draw[detected])
output_file.create_dataset('spa', data=spa_draw[detected])
output_file.create_dataset('nser', data=nser_draw[detected])
output_file.create_dataset('index', data=sourceind[:ndraw][detected])

