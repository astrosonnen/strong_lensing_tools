import numpy as np
import os
import glafic
import h5py
from simpars import *
from scipy.signal import convolve2d
from astropy.io import fits as pyfits
from scipy.interpolate import splev
from scipy.optimize import brentq
from sl_profiles import sersic, gnfw, deVaucouleurs as deV
import sl_cosmology
from sl_cosmology import G, M_Sun, Mpc, c
from scipy.special import gamma as gfunc
from lensdet import detect_lens
import sys


modelname = sys.argv[1]
sourcecat = pyfits.open('/Users/alessandro/catalogs/skills_sourceonly_zcut.fits')[1].data
pop = h5py.File('%s_galaxies.hdf5'%modelname, 'r+')

psf = pyfits.open('psf.fits')[0].data

modeldir = 'extended_sims/%s/'%modelname
if not os.path.isdir(modeldir):
    os.system('mkdir %s'%modeldir)

nsamp = pop.attrs['nsamp']
islens_samp = np.zeros(nsamp, dtype=bool)
tein_zs_samp = np.zeros(nsamp)

f = open('%s_sources.cat'%modelname, 'r')
sourcelines = f.readlines()[1:]
f.close()

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
pix_ext = pix_arcsec
pix_poi = 0.1
maxlev = 5

glafic.init(omegaM, omegaL, weos, hubble, prefix, xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb = 0)
glafic.set_secondary('flag_hodensity 2')
glafic.set_secondary('nfw_users 1')
glafic.set_secondary('hodensity 200')

glafic.startup_setnum(2, 1, 0)
glafic.set_lens(1, 'gnfw', 0.3, 1e13, 0.0, 0.0, 0., 90.0, 10., 1.5)
glafic.set_lens(2, 'sers', 0.3, 1e11, 0.0, 0.0, 0., 90.0, 1., 4.)
glafic.set_extend(1, 'sersic', zs_ref, 0.5, 0.3, 0., 0., 0., 0.1, 1.)

zs_list = []
xpos_list = []
ypos_list = []
nser_list = []
sreff_list = []
sq_list = []
spa_list = []
smag_list = []
avg_mu_list = []
nimg_list = []
nmax_list = []
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

for i in range(nsamp):
    print(i)
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

            nser = sourcecat['sersic_n_CM'][sourceind]
            sreff = sourcecat['Re_arcsec_CM'][sourceind]
            sq = sourcecat['axis_ratio_CM'][sourceind]
            if sq > 1.: # But why.
                sq = 1./sq
            elif sq < 0.:
                sq = -sq
            spa = sourcecat['PA_random'][sourceind]
            zs = sourcecat['zobs'][sourceind]
            smag = sourcecat['i_SDSS_apparent_corr'][sourceind]

            ftot = 10.**(-2./5.*(smag - zeropoint))
            I0 = ftot/(2.*np.pi*(sreff/pix_arcsec)**2*nser/sersic.b(nser)**(2*nser)*gfunc(2.*nser))

            glafic.set_extend(1, 'sersic', zs, I0, xpos, ypos, 1.-sq, spa, sreff, nser)

            # model_init needs to be done again whenever model parameters are changed
            glafic.model_init(verb = 0)

            img = np.array(glafic.writeimage())
            img_wseeing = convolve2d(img, psf, mode='same')

            # measures detectable source unlensed flux
            def zerofunc(R):
                return ftot * sersic.Sigma(R, nser, sreff/pix_arcsec) - nsigma_pixdet * sky_rms

            if zerofunc(0.) < 0.:
                fdet = 0.
            elif zerofunc(10.*sreff/pix_arcsec) > 0.:
                fdet = ftot
            else:
                Rmax = brentq(zerofunc, 0., 10.*sreff/pix_arcsec)
                fdet = ftot * sersic.M2d(Rmax, nser, sreff/pix_arcsec)

            detection, nimg_std, nimg_max, nholes_std, nholes_max, std_footprint, best_footprint, sb_maxlim = detect_lens(img_wseeing)

            if detection:
                islens = True

                #tein_zs = glafic.calcein2(zs, 0., 0.)

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

                zs_list.append(zs)
                xpos_list.append(xpos)
                ypos_list.append(ypos)
                nser_list.append(nser)
                sreff_list.append(sreff)
                sq_list.append(sq)
                spa_list.append(spa)
                smag_list.append(smag)
                nimg_list.append(nimg_std)
                nmax_list.append(nimg_max)

                # creates a noisy version of the image
                img_wnoise = img_wseeing + np.random.normal(0., sky_rms, img.shape)

                hdr = pyfits.Header()

                # creates an fits file for the lens
                hdr['galno'] = i
                hdr['zlens'] = pop['z'][i]
                hdr['tein_zrf'] = pop['tein'][i]
                hdr['tein_zs'] = tein_zs
                hdr['reff_ang'] = reff_arcsec
                hdr['reff_kpc'] = reff_kpc
                hdr['lm200'] = pop['lm200'][i]
                hdr['lmstar'] = pop['lmstar'][i]
                hdr['lens_q'] = pop['q'][i]
                hdr['zs'] = zs
                hdr['src_x'] = xpos
                hdr['src_y'] = ypos
                hdr['src_nser'] = nser
                hdr['src_q'] = sq
                hdr['src_pa'] = spa
                hdr['src_mag'] = smag
                hdr['src_re'] = sreff
                hdr['src_ind'] = sourceind
                hdr['nimg_std'] = nimg_std
                hdr['nimg_max'] = nimg_max
                hdr['nhol_std'] = nholes_std
                hdr['nhol_max'] = nholes_max

                # calculates the average magnification over the footprint
                #footprint = img > nsigma_pixdet * sky_rms

                avg_mu = abs(img[std_footprint]).sum()/fdet
                avg_mu_list.append(avg_mu)

                hdr['avg_mu'] = avg_mu

                phdu = pyfits.PrimaryHDU(header=hdr)

                ihdu = pyfits.ImageHDU(header=hdr, data=img_wseeing)
                nhdu = pyfits.ImageHDU(header=hdr, data=img_wnoise)

                hdulist = pyfits.HDUList([phdu, ihdu, nhdu])

                hdulist.writeto(modeldir+'/lens_%06d.fits'%i, overwrite=True)

            n += 1

    islens_samp[i] = islens

if 'islens' in pop:
    data = pop['islens']
    data[()] = islens_samp

else:
    pop.create_dataset('islens', data=islens_samp)

if 'tein_zs' in pop:
    data = pop['tein_zs']
    data[()] = tein_zs_samp

else:
    pop.create_dataset('tein_zs', data=tein_zs_samp)

# makes file with lenses

lens_file = h5py.File('%s_lenses.hdf5'%modelname, 'w')

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

lens_file.create_dataset('zs', data=np.array(zs_list))
lens_file.create_dataset('xpos', data=np.array(xpos_list))
lens_file.create_dataset('ypos', data=np.array(ypos_list))
lens_file.create_dataset('nser', data=np.array(nser_list))
lens_file.create_dataset('sreff', data=np.array(sreff_list))
lens_file.create_dataset('sq', data=np.array(sq_list))
lens_file.create_dataset('spa', data=np.array(spa_list))
lens_file.create_dataset('smag', data=np.array(smag_list))
lens_file.create_dataset('avg_mu', data=np.array(avg_mu_list))
lens_file.create_dataset('nimg', data=np.array(nimg_list))
lens_file.create_dataset('nmax', data=np.array(nmax_list))

pop.close()
lens_file.close()


