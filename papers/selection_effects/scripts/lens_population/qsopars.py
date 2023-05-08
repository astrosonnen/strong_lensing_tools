import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.integrate import quad
import sl_cosmology
from sl_cosmology import Mpc, M_Sun, c, G
from astropy.io import fits as pyfits


# number density of background sources
nbkg = 70. # per square arcminute

# detection limit
pointmag_max = 23.3

# integration limits of luminosity function
ms_max = pointmag_max + 2.
ms_min = pointmag_max - 4.

# Quasar luminosity function (from Manti et al. 2017)

def lphistar_func(z):
    return -6.0991 + 0.0209*z + 0.0171*z**2

def Mstar_func(z):
    return -22.5216 - 1.6510*z + 0.2869*z**2

alpha_Q = -1.35
beta_Q = -3.23

"""
qso_hdulist = pyfits.open('optical_nir_qso_sed_001.fits')

qso_data = qso_hdulist[1].data
qso_wav = qso_data['wavelength']
qso_flambda = qso_data['flux']
"""

f = open('qso_template.txt', 'r')
qso_wav, qso_flambda = np.loadtxt(f, unpack=True)
f.close()

qso_nu = 1./qso_wav

# I assume that the normalization is such that the flux is at a distance of 10pc
qso_restfnu = qso_flambda * qso_wav**2 

ref_lam = 1450. # Wavelength at which the Manti et al. (2017) luminosity function is defined
restuv_window = (qso_wav > ref_lam-50.) & (qso_wav < ref_lam+50.)

qso_uvnorm = np.median(qso_restfnu[restuv_window])

# absolute magnitude of the template
absmag_0 = -2.5*np.log10(qso_uvnorm)

# loads the SDSS i-band filter
f = open('i_SDSS.res', 'r')
filt_wav, filt_t = np.loadtxt(f, unpack=True)
f.close()

filt_nu = 1./filt_wav
filt_spline = splrep(np.flipud(filt_nu), np.flipud(filt_t))

# at each source redshift, I calculate the transformation from rest-frame UV absolute magnitude to
# observed frame i-band magnitude

nzs = 18
zqso_grid = np.linspace(0.8, 2.5, nzs)
magapp_star_grid = 0. * zqso_grid
comovd2_grid = 0. * zqso_grid
dcomovdz_grid = 0. * zqso_grid
phinorm_grid = 0. * zqso_grid
phidet_grid = 0. * zqso_grid
dz = 0.01

invcum_phiqso_splines = []
invcum_detqso_splines = []

nms = 61
ms_grid = np.linspace(ms_min, ms_max, nms)
ndet = 41
ms_detgrid = np.linspace(ms_min, pointmag_max, ndet)

for i in range(nzs):
    dlum = sl_cosmology.Dang(zqso_grid[i]) * (1. + zqso_grid[i])**2

    dmag = Mstar_func(zqso_grid[i]) - absmag_0

    qso_fnuobs = (1. + zqso_grid[i]) * qso_restfnu / (dlum/1e-5)**2
    qso_nuobs = qso_nu/(1. + zqso_grid[i])
    qso_wavobs = qso_wav * (1. + zqso_grid[i])

    integrand_range = (qso_wavobs > filt_wav[0]) & (qso_wavobs < filt_wav[-1])

    qso_nuobs_flipped = np.flipud(qso_nuobs[integrand_range])
    qso_fnuobs_flipped = np.flipud(qso_fnuobs[integrand_range])

    num_integrand = qso_fnuobs_flipped * splev(qso_nuobs_flipped, filt_spline) / qso_nuobs_flipped
    num_spline = splrep(qso_nuobs_flipped, num_integrand)
    num_integral = splint(qso_nuobs_flipped[0], qso_nuobs_flipped[-1], num_spline)

    den_integrand = splev(qso_nuobs_flipped, filt_spline) / qso_nuobs_flipped
    den_spline = splrep(qso_nuobs_flipped, den_integrand)
    den_integral = splint(qso_nuobs_flipped[0], qso_nuobs_flipped[-1], den_spline)
    magapp_star = -2.5*np.log10(num_integral/den_integral) + dmag
    magapp_star_grid[i] = -2.5*np.log10(num_integral/den_integral) + dmag

    comovd_here = sl_cosmology.comovd(zqso_grid[i])
    comovd2_grid[i] = comovd_here**2

    comovd_up = sl_cosmology.comovd(zqso_grid[i] + dz)
    comovd_dw = sl_cosmology.comovd(zqso_grid[i] - dz)

    dcomovdz_grid[i] = (comovd_up - comovd_dw)/(2.*dz)

    # integrates the quasar luminosity function up to the limiting magnitude
    phistar_here = 10.**lphistar_func(zqso_grid[i])
    def magfunc(magapp):
        return phistar_here / (10.**(0.4*(alpha_Q + 1.) * (magapp - magapp_star)) + 10.**(0.4*(beta_Q + 1.) * (magapp - magapp_star)))

    magfunc_norm = quad(magfunc, ms_min, ms_max)[0]
    phinorm_grid[i] = magfunc_norm
    phidet_grid[i] = quad(magfunc, ms_min, pointmag_max)[0]

    # creates a spline of the inverse cumulative function of the luminosity function, on the allowed magnitude range
    magfunc_spline = splrep(ms_grid, magfunc(ms_grid))
    cumfunc_phiqso_grid = 0.*ms_grid
    for j in range(nms):
        cumfunc_phiqso_grid[j] = splint(ms_grid[0], ms_grid[j], magfunc_spline)
    cumfunc_phiqso_grid /= cumfunc_phiqso_grid[-1]
    invcum_phiqso_spline = splrep(cumfunc_phiqso_grid, ms_grid)
    invcum_phiqso_splines.append(invcum_phiqso_spline)

    # same thing, but only on the detectable magnitude range
    cumfunc_detqso_grid = 0.*ms_detgrid
    for j in range(ndet):
        cumfunc_detqso_grid[j] = splint(ms_detgrid[0], ms_detgrid[j], magfunc_spline)
    cumfunc_detqso_grid /= cumfunc_detqso_grid[-1]
    invcum_detqso_spline = splrep(cumfunc_detqso_grid, ms_detgrid)
    invcum_detqso_splines.append(invcum_detqso_spline)

dpdz_grid = comovd2_grid * dcomovdz_grid * phinorm_grid
dpdz_spline = splrep(zqso_grid, dpdz_grid)

cumfunc_zqso_grid = 0.*zqso_grid
for i in range(nzs):
    cumfunc_zqso_grid[i] = splint(zqso_grid[0], zqso_grid[i], dpdz_spline)
cumfunc_zqso_grid /= cumfunc_zqso_grid[-1]

invcum_zqso_spline = splrep(cumfunc_zqso_grid, zqso_grid)

dpdzdet_grid = comovd2_grid * dcomovdz_grid * phidet_grid
dpdzdet_spline = splrep(zqso_grid, dpdzdet_grid)

cumfunc_zqsodet_grid = 0.*zqso_grid
for i in range(nzs):
    cumfunc_zqsodet_grid[i] = splint(zqso_grid[0], zqso_grid[i], dpdzdet_spline)
cumfunc_zqsodet_grid /= cumfunc_zqsodet_grid[-1]

invcum_zqsodet_spline = splrep(cumfunc_zqsodet_grid, zqso_grid)

# maps redshift into index of list of luminosity function splines
floatind_spline = splrep(zqso_grid, np.arange(nzs))
def ztoind(zqso):
    return int(round(float(splev(zqso, floatind_spline))))

