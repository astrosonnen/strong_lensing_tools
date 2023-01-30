import numpy as np
from sl_profiles import sersic
from scipy.special import gamma as gfunc
import os
import glafic
import h5py


np.random.seed(0)

nre = 11
logre_grid = np.linspace(-1., 0., nre)

tein = 1. # Einstein radius

ftot = 200. # ratio between total source flux and noise within tein^2.
pix = 0.05
nser = 1. # source Sersic index

nsource = 10000

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

glafic.startup_setnum(2, 1, 0)
glafic.set_lens(1, 'gnfw', 0.3, 2.021e12, 0.0, 0.0, 0.3, 90.0, 10., 1.5)
glafic.set_lens(2, 'sers', 0.3, 1.087e11, 0.0, 0.0, 0.3, 90.0, 1., 4.)
glafic.set_extend(1, 'sersic', 1.5, 1., 0.3, 0., 0., 0., 0.1, 1.)

# draws the same source positions for each value of re, so that
# different simulations can be compared more easily

# generates N sources within a circle of unit radius
r_unit = np.random.rand(nsource)**0.5 
phi = 2.*np.pi*np.random.rand(nsource)

for i in range(nre):

    # prepares the output file
    output_file = h5py.File('mockdir/logre%2.1f_images.hdf5'%logre_grid[i], 'w')

    re = 10.**logre_grid[i]

    rmax = 1. + re

    r = r_unit * rmax
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    I0 = ftot/(2.*np.pi*(re/pix)**2*nser/sersic.b(nser)**(2*nser)*gfunc(2.*nser))
   
    output_file.attrs['logre'] = logre_grid[i]
    output_file.attrs['pix'] = pix
    output_file.attrs['ftot'] = ftot
    output_file.attrs['I0'] = I0
    output_file.attrs['rmax'] = rmax

    output_file.create_dataset('x', data=x)
    output_file.create_dataset('y', data=y)

    # computes caustics and critical curves
    glafic.model_init(verb=0)
    glafic.writecrit(1.5)

    f = open('tmp_crit.dat', 'r')
    crit_table = np.loadtxt(f)
    f.close()

    output_file.create_dataset('crit_table', data=crit_table)

    for n in range(nsource):
        print(i, n)
        glafic.set_extend(1, 'sersic', 1.5, I0, x[n], y[n], 0., 90., re, nser)
        glafic.model_init(verb=0)

        img = np.array(glafic.writeimage())

        output_file.create_dataset('lens_%04d'%n, data=img)

    output_file.close()

