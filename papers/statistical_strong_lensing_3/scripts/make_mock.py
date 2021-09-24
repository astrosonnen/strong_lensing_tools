import numpy as np
from sl_profiles import nfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun, yr
import sl_cosmology
from scipy.optimize import brentq, minimize_scalar, leastsq
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev, splint, interp1d
from scipy.integrate import quad
import h5py
import sys


if len(sys.argv) > 1:
    seedno = int(sys.argv[1])
else:
    seedno = 0
mockname = '1e5mock_%d'%seedno

h = 0.7
cosmo = sl_cosmology.default_cosmo 
cosmo['h'] = h

ngal = 100000 # sample size

np.random.seed(seedno)

# source distribution parameters

alpha_sch = -1.32
Mstar_sch = -19.68

zs = 2.
ds = sl_cosmology.Dang(zs)
mstar_sch = Mstar_sch + 2.5*np.log10(ds**2/1e-10*(1.+zs))

ms_min = 21.
ms_max = 26.

maxmagB_det = 25. # minimum observed magnitude of 2nd image
mag_err = 0.1 # uncertainty on the observed magnitudes

def phifunc(m):
    return (10.**(-0.4*(m - mstar_sch)))**(alpha_sch+1.) * np.exp(-10.**(-0.4*(m - mstar_sch)))
 
# calculates the normalisation
phi_invnorm = quad(phifunc, ms_min, ms_max)[0]

# obtains a spline of the inverse function of the cumulative probability
ngrid = 101
mag_grid = np.linspace(ms_min, ms_max, ngrid)
cumphi_grid = 0.*mag_grid
for i in range(ngrid):
    cumphi_grid[i] = quad(phifunc, ms_min, mag_grid[i])[0]/phi_invnorm
invcumphi_spline = splrep(cumphi_grid, mag_grid)

nbkg = 100.
nbkg_arcsec2 = nbkg / 3600.

ltein_mu = -0.3
ltein_sig = 0.2

gamma_mu = 2.
gamma_sig = 0.2

ltein_samp = ltein_mu + ltein_sig * np.random.normal(0., 1., ngal)
gamma_samp = gamma_mu + gamma_sig * np.random.normal(0., 1., ngal)

tein_samp = 10.**ltein_samp

# prepares arrays to store lensing info
rein_samp = np.zeros(ngal)
ycaust_samp = np.zeros(ngal)
xradcrit_samp = np.zeros(ngal)

day = yr/365.
arcsec2rad = np.deg2rad(1./3600.)

eps = 1e-4

def get_ycaust(tein, gamma):

    xmin = 0.01*tein

    # defines lensing-related functions
    def alpha(x): 
        # deflection angle
        return tein * x/abs(x) * (abs(x)/tein)**(2.-gamma)

    def kappa(x): 
        # dimensionless surface mass density
        return (3.-gamma)/2. * (abs(x)/tein)**(1.-gamma)

    def zerofunc(x):
        return alpha(x) - x
        
    def radial_invmag(x):
        return 1. + alpha(x)/x - 2.*kappa(x)

    # finds the radial caustic
    if radial_invmag(xmin)*radial_invmag(tein) > 0.:
        xradcrit = xmin
    else:
        xradcrit = brentq(radial_invmag, xmin, tein)

    ycaust = -(xradcrit - alpha(xradcrit))

    return ycaust, xradcrit

def get_images(tein, gamma, xradcrit, beta):

    xmin = 0.01*tein
    xmax = 100.*tein

    def alpha(x): 
        # deflection angle
        return tein * x/abs(x) * (abs(x)/tein)**(2.-gamma)

    def kappa(x): 
        # dimensionless surface mass density
        return (3.-gamma)/2. * (abs(x)/tein)**(1.-gamma)

    def mu_r(x):
        # radial magnification
        return (1. + alpha(x)/x - 2.*kappa(x))**(-1)
    
    def mu_t(x):
        # tangential magnification
        return (1. - alpha(x)/x)**(-1)
    
    def absmu(x):
        # total magnification
        return abs(mu_r(x) * mu_t(x))

    # finds the images
    imageeq = lambda x: x - alpha(x) - beta
    if imageeq(xradcrit)*imageeq(xmax) >= 0. or imageeq(-xmax)*imageeq(-xradcrit) >= 0.:
        #xA, xB = -np.inf, np.inf
        return -np.inf, np.inf, 1e-10, 1e-10, np.inf
    else:
        xA = brentq(imageeq, tein, xmax)#, xtol=xtol)
        xB = brentq(imageeq, -tein, -xradcrit)#, xtol=xtol)

    muA = absmu(xA)
    muB = absmu(xB)

    return xA, xB, muA, muB

# loops over the galaxies, calculates the lensing cross-section
for i in range(ngal):

    ycaust, xradcrit = get_ycaust(tein_samp[i], gamma_samp[i])

    ycaust_samp[i] = ycaust
    xradcrit_samp[i] = xradcrit

# draws the number of sources behind each lens
lambda_samp = np.pi*ycaust_samp**2 * nbkg_arcsec2
nsource_samp = np.random.poisson(lam=lambda_samp)
nsource_tot = nsource_samp.sum()

# generates observational errors on the magnitude of the two images (adds them later)
magA_deltas = np.random.normal(0., mag_err, nsource_tot)
magB_deltas = np.random.normal(0., mag_err, nsource_tot)

# source magnitude 
mag_samp = splev(np.random.rand(nsource_tot), invcumphi_spline)

# scale-free source position: uniform distribution in a circle of radius 1
beta_scalefree_samp = np.random.rand(nsource_tot)**0.5

xA_samp = []
xB_samp = []

beta_samp = []

nlensed_sources = []

mag_unlensed = []
magA_true = []
magB_true = []
magA_obs = []
magB_obs = []

lenses = []

source_count = 0

for i in range(ngal):
    islens = False
    nlensed_here = 0
    if nsource_samp[i] > 0:
        nsource_here = nsource_samp[i]
        for n in range(nsource_here):
            beta_here = beta_scalefree_samp[source_count+n] * ycaust_samp[i]
            mag_here = mag_samp[source_count+n]
            xA, xB, muA, muB = get_images(tein_samp[i], gamma_samp[i], xradcrit_samp[i], beta_here)
            magA = mag_here -2.5*np.log10(muA)
            magB = mag_here -2.5*np.log10(muB)

            magA_obs_here = magA + magA_deltas[source_count+n]
            magB_obs_here = magB + magB_deltas[source_count+n]
            if (magB_obs_here < maxmagB_det):
                nlensed_here += 1
                if not islens:
                    islens = True
    
                    xA_samp.append(xA)
                    xB_samp.append(xB)
    
                    magA_true.append(magA)
                    magB_true.append(magB)
    
                    magA_obs.append(magA_obs_here)
                    magB_obs.append(magB_obs_here)
    
                    mag_unlensed.append(mag_here)
    
                    beta_samp.append(beta_here)
    
        source_count += nsource_here

    nlensed_sources.append(nlensed_here)
    lenses.append(islens)

xA_samp = np.array(xA_samp)
xB_samp = np.array(xB_samp)

mag_unlensed = np.array(mag_unlensed)

magA_true = np.array(magA_true)
magB_true = np.array(magB_true)

magA_obs = np.array(magA_obs)
magB_obs = np.array(magB_obs)

beta_samp = np.array(beta_samp)

nlensed_sources = np.array(nlensed_sources)
lenses = np.array(lenses)

print('%d lenses'%lenses.sum())

lens_tein_samp = tein_samp[lenses]
lens_ycaust_samp = ycaust_samp[lenses]
lens_xradcrit_samp = xradcrit_samp[lenses]

lens_gamma_samp = gamma_samp[lenses]

output = h5py.File('%s_pop.hdf5'%mockname, 'w')

# individual galaxy parameters
galaxies = output.create_group('Galaxies')

galaxies.create_dataset('tein', data=tein_samp)
galaxies.create_dataset('ycaust', data=ycaust_samp)
galaxies.create_dataset('xradcrit', data=xradcrit_samp)
galaxies.create_dataset('gamma', data=gamma_samp)
galaxies.create_dataset('lens', data=lenses)
galaxies.create_dataset('nsource', data=nlensed_sources)

# lens parameters
lensgroup = output.create_group('Lenses')

lensgroup.create_dataset('tein', data=lens_tein_samp)
lensgroup.create_dataset('ycaust', data=lens_ycaust_samp)
lensgroup.create_dataset('xradcrit', data=lens_xradcrit_samp)
lensgroup.create_dataset('xA', data=xA_samp)
lensgroup.create_dataset('xB', data=xB_samp)
lensgroup.create_dataset('mag_unlensed', data=mag_unlensed)
lensgroup.create_dataset('magA_true', data=magA_true)
lensgroup.create_dataset('magA_obs', data=magA_obs)
lensgroup.create_dataset('magB_true', data=magB_true)
lensgroup.create_dataset('magB_obs', data=magB_obs)
lensgroup.create_dataset('beta', data=beta_samp)
lensgroup.create_dataset('gamma', data=lens_gamma_samp)

# fixed parameters
output.attrs['nbkg'] = nbkg
output.attrs['ngal'] = ngal
output.attrs['maxmagB_det'] = maxmagB_det
output.attrs['ms_min'] = ms_min
output.attrs['ms_max'] = ms_max
output.attrs['alpha_sch'] = alpha_sch
output.attrs['Mstar_sch'] = Mstar_sch
output.attrs['zs'] = zs

# hyper-parameters
output.attrs['mag_err'] = mag_err
output.attrs['ltein_mu'] = ltein_mu
output.attrs['ltein_sig'] = ltein_sig
output.attrs['gamma_mu'] = gamma_mu
output.attrs['gamma_sig'] = gamma_sig

