import numpy as np
import os
import glafic
import h5py
from simpars import *
from qsopars import *
from scipy.signal import convolve2d
from scipy.interpolate import splev
from scipy.optimize import brentq
from sl_profiles import sersic, gnfw, deVaucouleurs as deV
import sl_cosmology
from sl_cosmology import G, M_Sun, Mpc, c
from scipy.special import gamma as gfunc
import sys


modelname = sys.argv[1]
pop = h5py.File('%s_galaxies.hdf5'%modelname, 'r+')

nsamp = pop.attrs['nsamp']
islens_samp = np.zeros(nsamp, dtype=bool)
tein_zs_samp = np.zeros(nsamp)

# reads the number of sources
f = open('%s_sources.cat'%modelname, 'r')
nqso = np.loadtxt(f, usecols=(2, ), dtype=int)
f.close()

nqso_tot = nqso.sum()

# generates redshift and magnitudes

zqso_samp = splev(np.random.rand(nqso_tot), invcum_zqso_spline)
t_samp = np.random.rand(nqso_tot)

qsomag_samp = np.zeros(nqso_tot)

for i in range(nqso_tot):
    ind = ztoind(zqso_samp[i])
    qsomag_here = splev(t_samp[i], invcum_phiqso_splines[ind])
    qsomag_samp[i] = qsomag_here

# primary parameters
omegaM = 0.3
omegaL = 0.7
weos = -1.
hubble = 0.7
prefix = 'tmp'
xmin = -5.
ymin = -5.
xmax = 5.
ymax = 5.
pix_ext = 0.1
pix_poi = 0.1
maxlev = 5

glafic.init(omegaM, omegaL, weos, hubble, prefix, xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb = 0)
glafic.set_secondary('flag_hodensity 2')
glafic.set_secondary('nfw_users 1')
glafic.set_secondary('hodensity 200')

glafic.startup_setnum(2, 0, 1)
glafic.set_lens(1, 'gnfw', 0.3, 1e13, 0.0, 0.0, 0., 90.0, 10., 1.5)
glafic.set_lens(2, 'sers', 0.3, 1e11, 0.0, 0.0, 0., 90.0, 1., 4.)
glafic.set_point(1, zs_ref, 0., 0.)

zqso_list = []
xpos_list = []
ypos_list = []
qsomag_list = []
nimg_list = []
tein_zs_list = []

# defines lensing-related functions (for computation of Einstein radius)
def alpha_dm(x, gnfw_norm, rs, gammadm, s_cr):
    # deflection angle (in kpc)
    return gnfw_norm * gnfw.fast_M2d(abs(x), rs, gammadm) / np.pi/x/s_cr

def alpha_star(x, mstar, reff, s_cr): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr):
    return alpha_dm(x, gnfw_norm, rs, gammadm, s_cr) + alpha_star(x, mstar, reff, s_cr)

kpc = Mpc/1000.

f = open('%s_sources.cat'%modelname, 'r')
sourcelines = f.readlines()[1:]
f.close()

sourcecount = 0
for i in range(nsamp):
    line = sourcelines[i].split()
    nsource = int(line[2])
    rmax = float(line[1])
    islens = False
    if nsource > 0:
        arcsec2kpc = np.deg2rad(1./3600.) * splev(pop['z'][i], dd_spline) * 1000.
        reff_kpc = 10.**pop['lreff'][i]
        reff_arcsec = reff_kpc/arcsec2kpc
        rs_arcsec = pop['rs'][i]/arcsec2kpc

        glafic.set_lens(1, 'gnfw', pop['z'][i], 10.**pop['lm200'][i]*hubble, 0., 0., 1. - pop['q'][i], 90., rs_arcsec, pop['gammadm'][i])
        glafic.set_lens(2, 'sers', pop['z'][i], 10.**pop['lmstar'][i]*hubble, 0., 0., 1. - pop['q'][i], 90., reff_arcsec, 4.)

        n = 0
        while not islens and n < nsource:
            sourcestrs = line[3+n].split(',')
            sourceind = int(sourcestrs[0])
            xpos = float(sourcestrs[1])
            ypos = float(sourcestrs[2])

            zs = zqso_samp[sourcecount]
            smag = qsomag_samp[sourcecount]
            sourcecount += 1

            glafic.set_point(1, zs, xpos, ypos)

            # model_init needs to be done again whenever model parameters are changed
            glafic.model_init(verb = 0)

            glafic.findimg(1)
            f = open('tmp_point.dat', 'r')
            pointlines = f.readlines()
            f.close()

            nimg = int(pointlines[0].split()[0])
            ndetected = 0

            for j in range(nimg):
                pointline = pointlines[j+1].split()
                mu_here = abs(float(pointline[2]))
                mag_img = smag -2.5*np.log10(mu_here)
                if mag_img < pointmag_max:
                    ndetected += 1

            if ndetected > 1:
                islens = True

                ds = sl_cosmology.Dang(zs)
                dds = sl_cosmology.Dang(pop['z'][i], zs)
                dd = splev(pop['z'][i], dd_spline)
                s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2
                arcsec2kpc = np.deg2rad(1./3600.) * dd * 1000.

                def zerofunc(x):
                    return x - alpha(x, pop['gnfw_norm'][i], pop['rs'][i], pop['gammadm'][i], 10.**pop['lmstar'][i], 10.**pop['lreff'][i], s_cr)

                xmin = max(deV.rgrid_min*10.**pop['lreff'][i], gnfw.R_grid[0]*pop['rs'][i])
                if zerofunc(xmin) > 0.:
                    rein_zs = 0.
                else:
                    rein_zs = brentq(zerofunc, xmin, 100.)

                tein_zs = rein_zs/arcsec2kpc
                tein_zs_samp[i] = tein_zs

                tein_zs_list.append(tein_zs)

                print('%d is a lens'%i)

                zqso_list.append(zs)
                xpos_list.append(xpos)
                ypos_list.append(ypos)
                qsomag_list.append(smag)
                nimg_list.append(ndetected)

            n += 1

    islens_samp[i] = islens

if 'qsolens' in pop:
    data = pop['qsolens']
    data[()] = islens_samp

else:
    pop.create_dataset('qsolens', data=islens_samp)

if 'tein_zqso' in pop:
    data = pop['tein_zqso']
    data[()] = tein_zs_samp

else:
    pop.create_dataset('tein_zqso', data=tein_zs_samp)

# makes file with lenses

lens_file = h5py.File('%s_qsolenses.hdf5'%modelname, 'w')

lens_file.create_dataset('z', data=pop['z'][islens_samp])
lens_file.create_dataset('index', data=np.arange(nsamp)[islens_samp])
lens_file.create_dataset('lmobs', data=pop['lmobs'][islens_samp])
lens_file.create_dataset('lmstar', data=pop['lmstar'][islens_samp])
lens_file.create_dataset('lasps', data=pop['lasps'][islens_samp])
lens_file.create_dataset('lm200', data=pop['lm200'][islens_samp])
lens_file.create_dataset('lmdm5', data=pop['lmdm5'][islens_samp])
lens_file.create_dataset('r200', data=pop['r200'][islens_samp])
lens_file.create_dataset('lreff', data=pop['lreff'][islens_samp])
lens_file.create_dataset('tein_zref', data=pop['tein'][islens_samp])
lens_file.create_dataset('tein_zs', data=np.array(tein_zs_list))
lens_file.create_dataset('tcaust', data=pop['tcaust'][islens_samp])
lens_file.create_dataset('q', data=pop['q'][islens_samp])
lens_file.create_dataset('rs', data=pop['rs'][islens_samp])
lens_file.create_dataset('gammadm', data=pop['gammadm'][islens_samp])

lens_file.create_dataset('zqso', data=np.array(zqso_list))
lens_file.create_dataset('xpos', data=np.array(xpos_list))
lens_file.create_dataset('ypos', data=np.array(ypos_list))
lens_file.create_dataset('qsomag', data=np.array(qsomag_list))
lens_file.create_dataset('nimg', data=np.array(nimg_list))

pop.close()
lens_file.close()

