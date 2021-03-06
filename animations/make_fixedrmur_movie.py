import numpy as np
from imageSim import SBModels, SEDModels, convolve
import pyplz_rgbtools
import os
from PIL import Image
from sl_profiles import gnfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun
import sl_cosmology
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
import pylab
from matplotlib import rc
rc('text', usetex=True)


rcpix = 1./pylab.rcParams['figure.dpi'] # matplotlib pixel in inches

npix = 121
scale = 0.5
pix = scale*0.168

nframes = 20
ylim = (1e6, 5e9)

x0 = float(npix/2)
y0 = float(npix/2)

X, Y = np.meshgrid(np.arange(1.*npix), np.arange(1.*npix))

re_source_0 = 0.4/pix
q_source = 1.
pa_source = 0.

source_mags = {'g': 24.7, 'r': 24.5, 'i': 24.5}

rgbbands = ['i', 'r', 'g']
std_cuts = scale**2*np.array((1.8, 1.2, 0.5)) # RGB scales optimized for gri HSC images

zp = {}

for band in rgbbands:
    zp[band] = 27.

# lens and source redshifts
zd = 0.4
zs = 1.5

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)

# angular diameter distances
dd = sl_cosmology.Dang(zd)
ds = sl_cosmology.Dang(zs)
dds = sl_cosmology.Dang(zs, zd)

kpc2pix = 0.001/dd/arcsec2rad/pix # number of pixels corresponding to 1 kpc in the lens plane

s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2

rhoc = sl_cosmology.rhoc(zd)

# lens default parameters
lmstar_0 = 11.5
reff = 7. # half-light radius in physical kpc
lmdm5_0 = 11.
rs = 100. # halo scale radius in kpc
gammadm_0 = 1.2
beta_0 = 2. # source position in lens-plane physical kpc

gnfw_norm_0 = 10.**lmdm5_0/gnfw.M2d(5., rs, gammadm_0)

r_arr = np.logspace(0., 2.)
rho_stars_scalefree = deV.rho(r_arr, reff)

# sequence of stellar mass values
lmstar_seq = np.zeros(nframes)
lmstar_max = lmstar_0 + 0.01
lmstar_min = lmstar_max - 0.4
lmstar_seq[:5] = np.linspace(lmstar_0, lmstar_max, 5)
lmstar_seq[5:15] = np.flipud(np.linspace(lmstar_min, lmstar_max, 10))
lmstar_seq[15:] = np.linspace(lmstar_min, lmstar_0, 5)

# defines lensing-related functions
def alpha_dm(x, gnfw_norm, rs, gammadm):
    # deflection angle (in kpc)
    return gnfw_norm * gnfw.fast_M2d(abs(x), rs, gammadm) / np.pi/x/s_cr

def alpha_star(x, mstar, reff): 
    # deflection angle (in kpc)
    return mstar * deV.M2d(abs(x), reff) / np.pi/x/s_cr

def alpha1d(x, gnfw_norm, rs, gammadm, mstar, reff):
    return alpha_dm(x, gnfw_norm, rs, gammadm) + alpha_star(x, mstar, reff)

def kappa(x, gnfw_norm, rs, gammadm, mstar, reff): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + gnfw_norm * gnfw.fast_Sigma(abs(x), rs, gammadm))/s_cr

def mu_r(x, gnfw_norm, rs, gammadm, mstar, reff):
    # radial magnification
    return (1. + alpha1d(x, gnfw_norm, rs, gammadm, mstar, reff)/x - 2.*kappa(x, gnfw_norm, rs, gammadm, mstar, reff))**(-1)

def mu_t(x, gnfw_norm, rs, gammadm, mstar, reff):
    # tangential magnification
    return (1. - alpha1d(x, gnfw_norm, rs, gammadm, mstar, reff)/x)**(-1)

# calculates image positions of default lens

xmin = 1.
xmax = 100.

def absmu(x):
    # total magnification
    return abs(mu_r(x) * mu_t(x))

def zerofunc(x):
    return alpha1d(x, gnfw_norm_0, rs, gammadm_0, 10.**lmstar_0, reff) - x
    
if zerofunc(xmin) < 0.:
    rein_0 = 0.
elif zerofunc(xmax) > 0.:
    rein_0 = np.inf
else:
    rein_0 = brentq(zerofunc, xmin, xmax)

# finds the images
imageeq = lambda x: x - alpha1d(x, gnfw_norm_0, rs, gammadm_0, 10.**lmstar_0, reff) - beta_0
if imageeq(xmin)*imageeq(xmax) >= 0. or imageeq(-xmax)*imageeq(-xmin) >= 0.:
    xA, xB = -np.inf, np.inf
else:
    xA = brentq(imageeq, rein_0, xmax)
    xB = brentq(imageeq, -rein_0, -xmin)

rmuA_0 = mu_r(xA, gnfw_norm_0, rs, gammadm_0, 10.**lmstar_0, reff)
rmuB_0 = mu_r(xB, gnfw_norm_0, rs, gammadm_0, 10.**lmstar_0, reff)
rmur_0 = rmuA_0/rmuB_0

print(xA, xB, rmur_0, rein_0)

alpha_star_xA = alpha_star(xA, 1., reff)
alpha_star_xB = alpha_star(xB, 1., reff)

def alpha_func(theta, theta_0, gnfw_norm, rs, gammadm, mstar, reff):
    theta_x, theta_y = theta
    theta_x0, theta_y0 = theta_0
    theta_amp = ((theta_x-theta_x0)**2 + (theta_y - theta_y0)**2)**0.5
    t_shape = theta_amp.shape
    theta_flat = theta_amp.flatten()
    alpha_amp = (alpha1d(theta_flat/kpc2pix, gnfw_norm, rs, gammadm, mstar, reff)*kpc2pix).reshape(t_shape)
    return (alpha_amp * (theta_x-theta_x0)/theta_amp, alpha_amp * (theta_y-theta_y0)/theta_amp)

ngamma = 11
gammadm_arr = np.linspace(0.8, 1.8, ngamma)
alpha_dm_xA_arr = 0.*gammadm_arr
alpha_dm_xB_arr = 0.*gammadm_arr
for i in range(ngamma):
    alpha_dm_xA_arr[i] = alpha_dm(xA, 1., rs, gammadm_arr[i])
    alpha_dm_xB_arr[i] = alpha_dm(xB, 1., rs, gammadm_arr[i])

rgbdata = []
gammadm_old = gammadm_0
for n in range(nframes):

    mstar_here = 10.**lmstar_seq[n]

    # calculates dark matter mass and slope that produces the same image positions and radial magnification ratio
    gnfw_norm_arr = (xA - xB - mstar_here*(alpha_star_xA - alpha_star_xB))/(alpha_dm_xA_arr - alpha_dm_xB_arr)
    rmu_xA_arr = mu_r(xA, gnfw_norm_arr, rs*np.ones(ngamma), gammadm_arr, mstar_here*np.ones(ngamma), reff*np.ones(ngamma))
    rmu_xB_arr = mu_r(xB, gnfw_norm_arr, rs*np.ones(ngamma), gammadm_arr, mstar_here*np.ones(ngamma), reff*np.ones(ngamma))
    rmur_arr = rmu_xA_arr/rmu_xB_arr

    rmur_spline = splrep(gammadm_arr, rmur_arr)
    #print(splev(gammadm_arr, rmur_spline))
    def zerofunc(gammadm):
        return splev(gammadm, rmur_spline) - rmur_0

    sign_change = zerofunc(gammadm_arr[1:]) * zerofunc(gammadm_arr[:-1]) < 0.
    if sign_change.sum() < 1:
        x_arr = np.linspace(-2.*rein_0, 2.*rein_0, 1001)
        for i in range(ngamma):
            pylab.plot(x_arr, x_arr - alpha1d(x_arr, gnfw_norm_arr[i], rs, gammadm_arr[i], mstar_here, reff))
        pylab.axvline(xB, color='k', linestyle='--')
        pylab.axvline(xA, color='k', linestyle='--')
        pylab.show()
        print(splev(gammadm_arr, rmur_spline))
        df
    gammadm_roots = gammadm_arr[:-1][sign_change]
    gammadm_here = gammadm_roots[abs(gammadm_roots - gammadm_old).argmin()]

    gammadm_old = gammadm_here

    alpha_dm_xA_here = alpha_dm(xA, 1., rs, gammadm_here)
    alpha_dm_xB_here = alpha_dm(xB, 1., rs, gammadm_here)

    gnfw_norm_here = (xA - xB - mstar_here*(alpha_star_xA - alpha_star_xB))/(alpha_dm_xA_here - alpha_dm_xB_here)    
    beta_here = xA - mstar_here*alpha_star_xA - gnfw_norm_here*alpha_dm_xA_here

    rmuA = mu_r(xA, gnfw_norm_here, rs, gammadm_here, mstar_here, reff)
    rmuB = mu_r(xB, gnfw_norm_here, rs, gammadm_here, mstar_here, reff)
    rmu_mean = 0.5*(rmuA + rmuB)
    print(rmuA, rmuB, rmuA/rmuB, gammadm_here)

    re_source_here = re_source_0 / rmu_mean

    alpha_X, alpha_Y = alpha_func((X, Y), (x0, y0), gnfw_norm_here, rs, gammadm_here, mstar_here, reff)
    beta_X, beta_Y = X - alpha_X, Y - alpha_Y

    #beta_x0 = x0 + theta_1 - theta_E*(theta_1/theta_E)**(2.-gammas[n])
    beta_x0 = x0 + beta_here*kpc2pix
    beta_y0 = y0

    source = SBModels.Sersic('source', {'x': beta_x0, 'y': beta_y0, 'q': q_source, 'pa': pa_source, 're': re_source_here, 'n': 1.})
    source.amp = 1.

    spix = source.pixeval(beta_X, beta_Y)

    rgbdic = {}
    for band in rgbbands:
        source_mag0 = source.Mag(zp[band])
        smodel = 10.**(-(source_mags[band] - source_mag0)/2.5) * spix / rmu_mean**2
        rgbdic[band] = smodel

    figname = 'fixedrmur_frame_%02d.png'%n

    im = Image.new('RGB', (npix, npix), 'black')
    fullim = Image.new('RGB', (4*npix, 2*npix), 'black')

    data_here = []
    for i in range(3):
        data_here.append(rgbdic[rgbbands[i]])

    im.putdata(pyplz_rgbtools.make_crazy_pil_format(data_here, std_cuts))
    #im.putdata(pyplz_rgbtools.make_crazy_pil_format(data_here, cuts_here))
    im = im.resize((2*npix, 2*npix), resample=Image.ANTIALIAS)
    #im.save(figname)
    fullim.paste(im, (0, 0))

    # now makes profile plot
    fig = pylab.figure(figsize=(2*npix*rcpix, 2*npix*rcpix))
    pylab.subplots_adjust(left=0.21, right=0.99, bottom=0.16, top=0.99)
    ax = fig.add_subplot(1, 1, 1)

    ax.loglog(r_arr, mstar_here*rho_stars_scalefree, label='Stars', color='r')
    ax.loglog(r_arr, gnfw_norm_here*gnfw.rho(r_arr, rs, gammadm_here), label='Dark matter', color='b')
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel('$\\rho(r)$ ($M_\odot$ kpc$^{-3}$')
    ax.set_xlabel('$r$ (kpc)')
    ax.legend(loc='upper right')
    #pylab.savefig('dof_plot_%02d.png'%n)
    pylab.savefig('tmp.png')
    pylab.close()

    plotim = Image.open('tmp.png')
    fullim.paste(plotim, (2*npix, 0))
    fullim.save(figname)

