import numpy as np
import pylab
from astropy.io import fits as pyfits
from skimage import measure
from matplotlib import rc
rc('text', usetex=True)


names = ['e0.0_re0.1_x0.0_y0.0', 'e0.4_re0.1_x0.122_y0.105', 'e0.4_re0.1_x0.4_y0.0', 'e0.4_re0.1_x0.7_y0.0']

sky_rms = 0.6
thresh = 3.*sky_rms

cut = 20
x0 = 39.5
y0 = 39.5

r = 7.

points = [[(21., y0), (58., y0)], [(24., 39.), (60., 39.)], [(29., y0), (65., y0)], [(70., 48.5), (70., 30.5)]]
phis = [(0., np.pi), (np.pi + np.arctan(0.5/15.), 2.*np.pi - np.arctan(0.5/20.)), (0., np.pi), (np.arctan(-9./30.5), np.arctan(9./30.5))]
text = ['180', '177', '180', '33']
textpos = [(x0-7., y0+10.), (x0-7., y0-12.), (x0-7., y0+10.), (x0+10., y0-1.)]

success = [True, True, True, False]

fsize = 18

fig, ax = pylab.subplots(1, 4, figsize=(16, 4))
pylab.subplots_adjust(left=0., right=1.00, bottom=0., top=1., wspace=0.)

for i in range(4):
    img = pyfits.open('%s_image.fits'%names[i])[0].data[cut:-cut, cut:-cut]

    footprint = img > thresh
    labels = measure.label(footprint)
    nreg = labels.max()
    print('%d regions'%nreg)
    for n in range(nreg):
        npix_here = (labels==n+1).sum()
        signal = img[labels==n+1].sum()
        noise = npix_here**0.5 * sky_rms
        img_sn = signal/noise
        if img_sn < 10.:
            img[labels==n+1] = 0.
        print(n, img_sn)

    ax[i].contourf(img,[thresh,1e8], colors='b')
    ax[i].set_aspect(1.)
    ax[i].scatter(x0, y0, color='k')

    if i==3:
        ax[i].plot([points[i][0][0], x0], [points[i][0][1], y0], color='k')
        ax[i].plot([points[i][1][0], x0], [points[i][1][1], y0], color='k')
    
        # draws semicircle
        phi = np.linspace(phis[i][0], phis[i][1], 101)
        circ_x = r * np.cos(phi) + x0
        circ_y = r * np.sin(phi) + y0
    
        ax[i].plot(circ_x, circ_y, color='k', linewidth=0.5)
        #ax[i].text(textpos[i][0], textpos[i][1], '$\\theta_{\mathrm{s}} = %s^\circ$'%text[i], fontsize=fsize)
        ax[i].text(textpos[i][0], textpos[i][1], '$%s^\circ$'%text[i], fontsize=fsize)

    if success[i]:
        #phi = np.pi*np.linspace(-1., 1., 1001)
        #x = 3. * np.cos(phi) + 10.
        #y = 3. * np.sin(phi) + 70.
        #ax[i].plot(x, y, color=(0., 1., 0.), linewidth=3)
        ax[i].text(10., 70., 'Lens', fontsize=fsize)
    else:
        #x = np.linspace(-3., 3.) + 10.
        #y1 = np.linspace(-3., 3.) + 70.
        #y2 = np.flipud(y1)
        #ax[i].plot(x, y1, color='r', linewidth=3)
        #ax[i].plot(x, y2, color='r', linewidth=3)
        ax[i].text(10., 70., 'Not a lens', fontsize=fsize)

    ax[i].tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

pylab.savefig('../../paper/footprints.eps')
pylab.show()


