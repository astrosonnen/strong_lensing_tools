import numpy as np
import glafic
from sl_profiles import sersic
from scipy.signal import convolve2d
import pylab
from astropy.io import fits as pyfits
from lensdet import detect_lens
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)


fsize = 28

nexamples = 6

psf = pyfits.open('psf.fits')[0].data

re_list = [0.2, 0.2, 0.1, 0.3, 0.4, 0.4]
srcpos_list = [(0.4, -0.2), (-0.8, 0.4), (0.6, 0.), (0.25, 0.), (-0.1, 0.), (0.3, 0.25)]
q_list = [0.7, 0.7, 0.3, 0.7, 0.7, 0.7]
zs_list = [1.5, 1.5, 1.5, 0.4, 0.35, 0.35]

# creates a custom colormap
cm_vals = np.ones((3, 4))
cm_vals[0, :] = np.array([1., 1., 1., 1.])
cm_vals[1, :] = np.array([0.5, 0., 1., 1.])
cm_vals[2, :] = np.array([1., 0., 0.5, 1.])
my_cmap = ListedColormap(cm_vals)

pix = 0.05
sky_rms = 1. # noise per pixel
nser = 1. # source Sersic index

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
glafic.set_secondary('flag_halodensity 2')
glafic.set_secondary('nfw_users 1')
glafic.set_secondary('halodensity 200')

glafic.startup_setnum(2, 1, 0)
glafic.set_lens(1, 'gnfw', 0.3, 2.021e12, 0.0, 0.0, 0.3, 90.0, 10., 1.5)
glafic.set_lens(2, 'sers', 0.3, 1.087e11, 0.0, 0.0, 0.3, 90.0, 1., 4.)
glafic.set_extend(1, 'sersic', 1.5, 1., 0.3, 0., 0., 0., 0.1, 1.)
#glafic.set_psf(2.*pix, 0., 0., 5., 1., 0., 0., 1., 1.)

sizeone = 4.
fig, ax = pylab.subplots(nexamples, 3, figsize=(3*sizeone, nexamples*sizeone))
pylab.subplots_adjust(left=0., right=1.00, bottom=0., top=1., wspace=0., hspace=0.)

for n in range(nexamples):

    re = re_list[n]

    # normalizes the total flux so that the limiting surface brightness
    # corresponds to the half-light radius surface brightness

    sb_lim = 2. *sky_rms
    I0 = sb_lim*sersic.Sigma(0., nser, re/pix)/sersic.Sigma(re/pix, nser, re/pix)

    x, y = srcpos_list[n]
   
    glafic.set_lens(1, 'gnfw', 0.3, 2.021e12, 0.0, 0.0, 1. - q_list[n], 90.0, 10., 1.5)
    glafic.set_lens(2, 'sers', 0.3, 1.087e11, 0.0, 0.0, 1. - q_list[n], 90.0, 1., 4.)
    glafic.set_extend(1, 'sersic', zs_list[n], I0, x, y, 0., 90., re, nser)
    glafic.model_init(verb=0)

    glafic.writecrit(zs_list[n]) # writes caustics onto 'tmp_crit.dat'

    #glafic.readpsf('psf.fits')
    img = np.flipud(np.array(glafic.writeimage()))
    img = convolve2d(img, psf, mode='same')

    img_wnoise = img + np.random.normal(0., sky_rms, img.shape)

    pyfits.PrimaryHDU(img).writeto('test.fits', overwrite=True)

    ny, nx = img.shape
    x0 = nx/2. - 0.5
    y0 = ny/2. - 0.5

    xs_pix = x0 + x/pix
    ys_pix = y0 + y/pix

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    unlensed_sb = sb_lim/sersic.Sigma(re/pix, nser, re/pix) * sersic.Sigma(((X-xs_pix)**2 + (Y-ys_pix)**2)**0.5, nser, re/pix).reshape((ny, nx))

    ax[n, 0].contour(unlensed_sb, [sb_lim], colors=['b'], extent=(-2., 2., -2., 2.))

    islens, nimg_std, nimg_best, nholes_std, nholes_best, std_footprint, best_footprint, sb_maxlim = detect_lens(img, sky_rms, npix_min=5)
    print(n+1, nholes_std, nholes_best)

    f = open('tmp_crit.dat', 'r')
    table = np.loadtxt(f)
    f.close()

    xs1 = table[:, 2]
    ys1 = table[:, 3]
    xs2 = table[:, 6]
    ys2 = table[:, 7]

    nseg = len(xs1)

    for j in range(nseg):
        if n==0 and j==0:
            ax[n, 0].plot([xs1[j], xs2[j]], [ys1[j], ys2[j]], color='r', label='Caustics')
            ax[n, 0].axvline(-10., color='b', label='Source')
        else:
            ax[n, 0].plot([xs1[j], xs2[j]], [ys1[j], ys2[j]], color='r')

    fp_img = 0.*img
    fp_img[std_footprint] = 0.4
    fp_img[best_footprint] = 0.8
    ax[n, 2].imshow(fp_img, vmin=0., vmax=1., cmap=my_cmap)
    ax[n, 2].set_aspect(1.)

    ax[n, 0].set_xlim(xmin, xmax)
    ax[n, 0].set_ylim(ymin, ymax)

    if n==0:
        ax[n, 0].legend(loc='upper left', fontsize=fsize)
        ax[n, 2].axvspan(-100., -90., color=cm_vals[1, :3], label='$2\sigma$ detection')# footprint')
        ax[n, 2].axvspan(-100., -90., color=cm_vals[2, :3], label='Max. \# images')# footprint')

        ax[n, 2].set_xlim(0., nx-1.)
        ax[n, 2].set_ylim(ny-1., 0.)

        ax[n, 2].legend(loc='upper left', fontsize=fsize)

    ax[n, 1].imshow(img_wnoise, cmap='gray', vmin = -sky_rms, vmax=10.*sky_rms)
    #ax[n, 1].imshow(img, cmap='gray', vmin = -sky_rms, vmax=5.*sky_rms)

    for j in range(3):
        ax[n, j].tick_params(axis='both', which='both', direction='in', labelleft=False, labelbottom=False, left=False, bottom=False)

    if islens:
        ax[n, 2].text(10., 70., 'Lens', fontsize=fsize)
    else:
        ax[n, 2].text(10., 70., 'Not a lens', fontsize=fsize)

pylab.savefig('../../paper/example_detections.eps')
pylab.show()


