import numpy as np
from sl_profiles import sersic
from scipy.interpolate import splrep, splev
import os
import glafic
import h5py
from scipy.signal import convolve2d
from astropy.io import fits as pyfits


# keeps the source surface brightness fixed and varies the redshift
np.random.seed(0)

nrein = 5
ltein_grid = np.linspace(-0.9, -0.1, nrein)

pix = 0.05
nser = 1. # source Sersic index
reff_arcsec = 1.
reff_pix = reff_arcsec/pix
I0 = sersic.Sigma(0., nser, reff_pix)/sersic.Sigma(1./pix, nser, reff_pix)
zs_ref = 1.5

nsource = 10000

# loads the psf
psf = pyfits.open('psf.fits')[0].data

# primary parameters
omegaM = 0.3
omegaL = 0.7
weos = -1.
hubble = 0.7
prefix = 'tmp'
xmin = -2.
ymin = -2.
xmax = 2.
ymax = 2.
pix_ext = pix
pix_poi = 0.1
maxlev = 5

glafic.init(omegaM, omegaL, weos, hubble, prefix, xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb = 0)
glafic.set_secondary('flag_hodensity 2')
glafic.set_secondary('nfw_users 1')
glafic.set_secondary('hodensity 200')

# WARNING: setting ellipticity to zero, for the purpose of computing
# the Einstein radius. Need to bring it back up to 0.3 later.
glafic.startup_setnum(2, 1, 0)
glafic.set_lens(1, 'gnfw', 0.3, 2.021e12, 0.0, 0.0, 0., 90.0, 10., 1.5)
glafic.set_lens(2, 'sers', 0.3, 1.087e11, 0.0, 0.0, 0., 90.0, 1., 4.)
glafic.set_extend(1, 'sersic', 1.5, I0, 0.3, 0., 0., 0., 0.1, 1.)

glafic.model_init(verb=0)

# calculates the Einstein radius on a grid of source redshifts
nz = 21
zs_grid = np.linspace(0.32, 1.8, nz)
tein_grid = 0.*zs_grid

for i in range(nz):
    tein_grid[i] = glafic.calcein2(zs_grid[i], 0., 0.)

zs_spline = splrep(tein_grid, zs_grid)

# draws the same source positions for each value of re, so that
# different simulations can be compared more easily

# generates N sources within a circle of unit radius
r_unit = np.random.rand(nsource)**0.5 
phi = 2.*np.pi*np.random.rand(nsource)

glafic.set_lens(1, 'gnfw', 0.3, 2.021e12, 0.0, 0.0, 0.3, 90.0, 10., 1.5)
glafic.set_lens(2, 'sers', 0.3, 1.087e11, 0.0, 0.0, 0.3, 90.0, 1., 4.)

for i in range(nrein):

    # prepares the output file
    output_file = h5py.File('mockdir/ltein%2.1f_images.hdf5'%ltein_grid[i], 'w')

    tein = 10.**ltein_grid[i]

    zs = splev(tein, zs_spline)

    rmax = tein + reff_arcsec

    r = r_unit * rmax
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    output_file.attrs['ltein'] = ltein_grid[i]
    output_file.attrs['pix'] = pix
    output_file.attrs['I0'] = I0
    output_file.attrs['rmax'] = rmax

    output_file.create_dataset('x', data=x)
    output_file.create_dataset('y', data=y)

    # computes caustics and critical curves
    glafic.model_init(verb=0)
    glafic.writecrit(zs)

    f = open('tmp_crit.dat', 'r')
    crit_table = np.loadtxt(f)
    f.close()

    output_file.create_dataset('crit_table', data=crit_table)

    for n in range(nsource):
        print(i, n)
        glafic.set_extend(1, 'sersic', zs, I0, x[n], y[n], 0., 90., reff_arcsec, nser)
        glafic.model_init(verb=0)

        img = np.array(glafic.writeimage())

        output_file.create_dataset('lens_%04d'%n, data=img)
        output_file.create_dataset('lens_%04d_wseeing'%n, data=convolve2d(img, psf, mode='same'))

    output_file.close()

