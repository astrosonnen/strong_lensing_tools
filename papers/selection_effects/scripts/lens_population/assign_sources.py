import numpy as np
from scipy.stats import poisson
from simpars import *
import h5py
from astropy.io import fits as pyfits
import sys


np.random.seed(10)
# places sources in a circle enclosing the radial caustic

circ_caust_rat = 1.2 # ratio between circle radius and caustic radius

modelname = sys.argv[1]
pop = h5py.File('%s_galaxies.hdf5'%modelname, 'r')

nsamp = pop.attrs['nsamp']

# Skills source catalog (not included in this repository, contact Shun-Sheng Li)
sourcecat = pyfits.open('skills_sourceonly_zcut.fits')[1].data

nsource_tot = len(sourcecat)

sourceind = np.arange(nsource_tot)
# shuffles source catalog (it's originally ranked by redshift)
np.random.shuffle(sourceind)

rrand = np.random.rand(len(sourceind))**0.5
phirand = 2.*np.pi * np.random.rand(len(sourceind))

outlines = []
outlines.append('# lens_id rcirc nsources source_index,xpos,ypos(list)\n')

sourcecount = 0
for i in range(nsamp):
    rcirc = circ_caust_rat*pop['tcaust'][i]/pop['q'][i]**0.5

    area = np.pi*rcirc**2

    lam = area * nbkg / 3600. # expectation value of the number of sources in the circle

    nsources = poisson.rvs(lam)

    line = '%d %f %d'%(i, rcirc, nsources)
    for n in range(nsources):
        xsource = rrand[sourcecount] * rcirc * np.cos(phirand[sourcecount])
        ysource = rrand[sourcecount] * rcirc * np.sin(phirand[sourcecount])

        line += ' %d,%f,%f'%(sourceind[sourcecount], xsource, ysource)
        sourcecount += 1
    line += '\n'

    outlines.append(line)

print('Total number of sources: %d'%sourcecount)
f = open('%s_sources.cat'%modelname, 'w')
f.writelines(outlines)
f.close()

