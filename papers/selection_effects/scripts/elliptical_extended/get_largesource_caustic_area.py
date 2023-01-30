import numpy as np
from scipy.interpolate import splrep, splev
import os
import glafic
import h5py


nrein = 11
ltein_grid = np.linspace(-1., 0., nrein)

zs_ref = 1.5
pix = 0.05

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
glafic.set_extend(1, 'sersic', 1.5, 1., 0.3, 0., 0., 0., 0.1, 1.)

glafic.model_init(verb=0)


# calculates the Einstein radius on a grid of source redshifts
nz = 21
zs_grid = np.linspace(0.32, 1.8, nz)
tein_grid = 0.*zs_grid

for i in range(nz):
    tein_grid[i] = glafic.calcein2(zs_grid[i], 0., 0.)

zs_spline = splrep(tein_grid, zs_grid)

f = open('preamble.input', 'r')
prelines = f.readlines()
f.close()

nmocks = 1000

output_file = h5py.File('largesource_caustic_area.hdf5', 'w')

cs_grid = 0.*ltein_grid
cs_quads_grid = 0.*ltein_grid

for i in range(nrein):

    tein = 10.**ltein_grid[i]

    zs = splev(tein, zs_spline)

    glafic.writecrit(zs)

    f = open('tmp_crit.dat', 'r')
    table = np.loadtxt(f)
    f.close()

    xs1 = table[:, 2]
    ys1 = table[:, 3]
    xs2 = table[:, 6]
    ys2 = table[:, 7]

    caust_size = max(abs(xs1).max(), abs(ys1).max(), abs(xs2).max(), abs(ys2).max())

    print('%d, theta_Ein=%4.3f, pixel_size=%4.3f, caustic_size=%4.3f'%(i, tein, tein/10., caust_size))

    rmax = caust_size*1.01

    outlines = prelines.copy()
    outlines.append('reset_par pix_poi %f\n'%(tein/10.))
    outlines.append('reset_par prefix tmp\n')
    outlines.append('mock1 %d %f %f %f %f %f\n'%(nmocks, zs, -rmax, rmax, -rmax, rmax))
    outlines.append('quit\n')

    f = open('tmp.input', 'w')
    f.writelines(outlines)
    f.close()

    os.system('/Users/alessandro/glafic tmp.input')

    nmulti = 0
    nquads = 0

    f = open('tmp_mock.dat', 'r')
    lines = f.readlines()[1:]
    f.close()
    
    nimg = 0

    ndet_list = []

    sline = True
    for line in lines:
        if sline:
            nimg = int(line[0])
            nhere = 0
            if nimg > 0:
                sline = False
                if nimg > 1:
                    nmulti += 1
                    if nimg > 3:
                        nquads += 1
            else:
                sline = True
        elif nimg > 1:
            nhere += 1
            if nhere == nimg:
                sline = True
        else:
            sline = True

    cs_grid[i] = nmulti * 4. *rmax**2 / nmocks
    cs_quads_grid[i] = nquads * 4. *rmax**2 / nmocks

output_file.create_dataset('ltein_grid', data=ltein_grid)
output_file.create_dataset('cs_grid', data=cs_grid)
output_file.create_dataset('cs_quads_grid', data=cs_quads_grid)
 
