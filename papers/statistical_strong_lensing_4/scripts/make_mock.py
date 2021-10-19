import numpy as np
from sl_profiles import nfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun, yr
import sl_cosmology
from scipy.optimize import brentq, minimize_scalar, leastsq
from scipy.stats import truncnorm
from scipy.interpolate import splrep, splev, splint, interp1d
from scipy.integrate import quad
import emcee
import h5py
import sys
import os
from astropy.io import fits as pyfits


if len(sys.argv) > 1:
    seedno = int(sys.argv[1])
else:
    seedno = 0
mockname = '1e4mock_%d'%seedno

h = 0.7
cosmo = sl_cosmology.default_cosmo 
cosmo['h'] = h

zd = 0.4
dd = sl_cosmology.Dang(zd)

kpc = Mpc/1000.

ngal = 10000 # sample size

np.random.seed(seedno)

# source distribution parameters

zs_min = zd + 0.1
zs_max = 4.
ms_min = 22.
ms_max = 27.
alpha_sch = -1.32
Mstar_sch = -19.68
phistar_sch = 7.02 * 1e-3 # comoving number density in Mpc^-3

nms = 61
ms_grid = np.linspace(ms_min, ms_max, nms)

# loads the spectral template
spec = pyfits.open('starb1_template.fits')[1].data
temp_wav = spec['WAVELENGTH']
temp_nu = 1./temp_wav
temp_flambda = spec['FLUX']
temp_restfnu = temp_flambda * temp_wav**2 # I don't care about normalization here

restuv_window = (temp_wav > 1450.) & (temp_wav < 1550.)
temp_uvnorm = np.median(temp_restfnu[restuv_window])

absmag_0 = -2.5*np.log10(temp_uvnorm/(1e-5)**2)
scale = 10.**(1./2.5*(absmag_0 - Mstar_sch))

temp_lumnu = scale * temp_restfnu
temp_lumlambda = scale * temp_flambda

# loads the filter
f = open('i_SDSS.res', 'r')
filt_wav, filt_t = np.loadtxt(f, unpack=True)
f.close()

filt_nu = 1./filt_wav
filt_spline = splrep(np.flipud(filt_nu), np.flipud(filt_t))

nzs = 36
zs_grid = np.linspace(zs_min, zs_max, nzs)

mstar_sch_grid = 0.*zs_grid
ds_grid = 0.*zs_grid
dds_grid = 0.*zs_grid
s_cr_grid = 0.*zs_grid
comovd_grid = 0.*zs_grid
mag_integral_grid = 0.*zs_grid
dcomovd_dz = 0.*zs_grid

solid_angle = np.deg2rad(1./60.)**2

dz = 0.01

for i in range(nzs):
    ds = sl_cosmology.Dang(zs_grid[i])
    dds = sl_cosmology.Dang(zd, zs_grid[i])
    ds_grid[i] = ds
    dds_grid[i] = dds
    comovd = sl_cosmology.comovd(zs_grid[i])
    comovd_grid[i] = comovd

    comovd_up = sl_cosmology.comovd(zs_grid[i] + dz)
    comovd_dw = sl_cosmology.comovd(zs_grid[i] - dz)

    dcomovd_dz[i] = (comovd_up - comovd_dw)/(2.*dz)

    s_cr_here = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2
    s_cr_grid[i] = s_cr_here

    dlum = ds * (1.+zs_grid[i])**2 # luminosity distance in Mpc

    temp_obsflambda = temp_lumlambda / dlum**2 / (1.+zs_grid[i])
    temp_obswav = (1.+zs_grid[i]) * temp_wav
    temp_obsfnu = temp_obsflambda  * temp_obswav**2

    integrand_range = (temp_obswav > filt_wav[0]) & (temp_obswav < filt_wav[-1])

    obsnu_flipped = np.flipud(1./temp_obswav[integrand_range])
    temp_obsfnu_flipped = np.flipud(temp_obsfnu[integrand_range])

    num_integrand = temp_obsfnu_flipped * splev(obsnu_flipped, filt_spline) / obsnu_flipped
    num_spline = splrep(obsnu_flipped, num_integrand)
    num_integral = splint(obsnu_flipped[0], obsnu_flipped[-1], num_spline)

    den_integrand = splev(obsnu_flipped, filt_spline) / obsnu_flipped
    den_spline = splrep(obsnu_flipped, den_integrand)
    den_integral = splint(obsnu_flipped[0], obsnu_flipped[-1], den_spline)

    mstar_sch = -2.5*np.log10(num_integral/den_integral)
    mstar_sch_grid[i] = mstar_sch

    def integrand_func(m_s):
        return 0.4*np.log(10.)*phistar_sch*(10.**(-0.4*(m_s - mstar_sch)))**(alpha_sch+1.) * np.exp(-10.**(-0.4*(m_s - mstar_sch)))

    mag_integral = quad(integrand_func, ms_min, ms_max)[0]
    mag_integral_grid[i] = mag_integral

    print(zs_grid[i], mstar_sch)

s_cr_spline = splrep(zs_grid, s_cr_grid)

mstar_sch_spline = splrep(zs_grid, mstar_sch_grid)
comovd_integrand_grid = comovd_grid**2 * mag_integral_grid
comovd_integrand_spline = splrep(comovd_grid, comovd_integrand_grid)
nbkg = solid_angle * splint(comovd_grid[0], comovd_grid[-1], comovd_integrand_spline)

maxmagB_det = 26. # minimum observed magnitude of 2nd image
mag_err = 0.1 # uncertainty on the observed magnitudes

pz_grid = comovd_integrand_grid * dcomovd_dz
pz_spline = splrep(zs_grid, pz_grid)
pz_invnorm = splint(zs_grid[0], zs_grid[-1], pz_spline)

def pz(zs):
    return splev(zs, pz_spline) / pz_invnorm

def phifunc(m, mstar):
    return 0.4*np.log(10.)*phistar_sch*(10.**(-0.4*(m - mstar)))**(alpha_sch+1.) * np.exp(-10.**(-0.4*(m - mstar)))

dvol_dz_grid = comovd_grid**2 * dcomovd_dz
dvol_dz_spline = splrep(zs_grid, dvol_dz_grid)
def zs_ms_func(m, zs):
    return phifunc(m, splev(zs, mstar_sch_spline)) * splev(zs, dvol_dz_spline)

# obtains a spline of the inverse function of the cumulative probability in redshift
cumpz_grid = 0.*zs_grid
for i in range(nzs):
    cumpz_grid[i] = splint(zs_grid[0], zs_grid[i], pz_spline) / pz_invnorm

invcumpz_spline = splrep(cumpz_grid, zs_grid)

# calculates the integral of Phi over a broad range of magnitudes
deltam_min = -3.
deltam_max = ms_max - mstar_sch_grid.min()

dm = 0.01
deltam_grid = np.arange(deltam_min, deltam_max + dm, dm)
ndeltam = len(deltam_grid)

cumfullphi_grid = 0.*deltam_grid
fullphi_grid = phifunc(deltam_grid, 0.)
fullphi_spline = splrep(deltam_grid, fullphi_grid)
for i in range(ndeltam):
    cumfullphi_grid[i] = splint(deltam_grid[0], deltam_grid[i], fullphi_spline)

cumfullphi_spline = splrep(deltam_grid, cumfullphi_grid)
 
nbkg_arcsec2 = nbkg / 3600.

lmstar_piv = 11.5 # pivot in stellar mass, to define the mstar-reff and mstar-m200 relations

zd = 0.4 # lens redshift
c200 = 5. # halo concentration

llum_min = 10.5
llum_max = 12.
llum_llstar = 10.8 # knee of galaxy luminosity function
llum_alpha = -1. # faint-end slope of galaxy luminosity function

lustar_mu = 0.5 # average value of log(Upsilon_*)
lustar_sig = 0.1 # intrinsic scatter in log(Upsilon_*)

lreff_mu = 1. # average value of log(Reff) at logM*=lmstar_piv
lreff_beta = 0.8 # slope of mass-size relation
lreff_sig = 0.15 # intrinsic scatter in Reff at fixed logM*

lm200_mu = 13. # average logM200 at logM*=lmstar_piv
lm200_sig = 0.2 # intrinsic scatter in logM200
lm200_beta = 1.5 # slope of stellar mass-halo mass relation

# generate the values of luminosity, stellar mass, size, halo mass
def gal_phifunc(llum):
    return (10.**(llum - llum_llstar))**(llum_alpha +1.) * np.exp(-10.**(llum - llum_llstar))
 
gal_norm = 1./quad(gal_phifunc, llum_min, llum_max)[0]

# obtains a spline of the inverse function of the cumulative probability
ngrid = 101
llum_grid = np.linspace(llum_min, llum_max, ngrid)
cumgalphi_grid = 0.*llum_grid
for i in range(ngrid):
    cumgalphi_grid[i] = quad(gal_phifunc, llum_min, llum_grid[i])[0]*gal_norm
invcumgalphi_spline = splrep(cumgalphi_grid, llum_grid)

llum_samp = splev(np.random.rand(ngal), invcumgalphi_spline)

lustar_samp = lustar_mu + lustar_sig * np.random.normal(0., 1., ngal)
lmstar_samp = llum_samp + lustar_samp

lreff_samp = lreff_mu + lreff_beta * (lmstar_samp - lmstar_piv) + np.random.normal(0., lreff_sig, ngal)

lm200_samp = lm200_mu + lm200_beta * (lmstar_samp - lmstar_piv) + np.random.normal(0., lm200_sig, ngal)

# prepares arrays to store lensing info
rein_max_samp = np.zeros(ngal)
ycaust_max_samp = np.zeros(ngal)
xradcrit_max_samp = np.zeros(ngal)

kpc = Mpc/1000.
day = yr/365.
arcsec2rad = np.deg2rad(1./3600.)
arcsec2kpc = arcsec2rad * dd * 1000.
kpc2arcsec = 1./arcsec2kpc

rhoc = sl_cosmology.rhoc(zd, cosmo=cosmo) # critical density of the Universe at z=zd. Halo masses are defined as M200 wrt rhoc.

r200_samp = (10.**lm200_samp*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
rs_samp = r200_samp/c200

eps = 1e-4

xmin = 0.01
xmax = 1000.

def get_rein_ycaust(s_cr, m200, mstar, reff, xmin, xmax):

    r200 = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000. # in kpc
    rs = r200/c200

    nfw_norm = m200/nfw.M3d(r200, rs)

    # defines lensing-related functions
    def alpha(x): 
        # deflection angle (in arcsec)
        return (nfw_norm*nfw.M2d(abs(x), rs) + mstar * deV.M2d(abs(x), reff)) / np.pi/x/s_cr

    def kappa(x): 
        # dimensionless surface mass density
        return (mstar * deV.Sigma(abs(x), reff) + nfw_norm*nfw.Sigma(abs(x), rs))/s_cr

    def zerofunc(x):
        return alpha(x) - x
        
    if zerofunc(xmin) < 0.:
        rein = 0.
    elif zerofunc(xmax) > 0.:
        rein = np.inf
    else:
        rein = brentq(zerofunc, xmin, xmax)

    def radial_invmag(x):
        return 1. + alpha(x)/x - 2.*kappa(x)

    # finds the radial caustic
    if radial_invmag(xmin)*radial_invmag(xmax) > 0.:
        xradcrit = xmin
    else:
        xradcrit = brentq(radial_invmag, xmin, xmax)

    ycaust = -(xradcrit - alpha(xradcrit))

    return rein, ycaust, xradcrit

def get_images(s_cr, m200, mstar, reff, beta, rein, xradcrit, xmin, xmax):

    r200 = (m200*3./200./(4.*np.pi)/rhoc)**(1./3.) * 1000.
    rs = r200/c200

    nfw_norm = m200/nfw.M3d(r200, rs)

    def alpha(x): 
        # deflection angle (in kpc)
        return (nfw_norm*nfw.M2d(abs(x), rs) + mstar * deV.M2d(abs(x), reff)) / np.pi/x/s_cr

    def kappa(x): 
        # dimensionless surface mass density
        return (mstar * deV.Sigma(abs(x), reff) + nfw_norm*nfw.Sigma(abs(x), rs))/s_cr
    
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
        xA = brentq(imageeq, rein, xmax)#, xtol=xtol)
        xB = brentq(imageeq, -rein, -xradcrit)#, xtol=xtol)

    muA = absmu(xA)
    muB = absmu(xB)

    return xA, xB, muA, muB

# loops over the galaxies, calculates the lensing cross-section
for i in range(ngal):
    
    reff = 10.**lreff_samp[i]
    rein_max, ycaust_max, xradcrit_max = get_rein_ycaust(s_cr_grid[-1], 10.**lm200_samp[i], 10.**lmstar_samp[i], reff, xmin, xmax)
    
    rein_max_samp[i] = rein_max
    ycaust_max_samp[i] = ycaust_max
    xradcrit_max_samp[i] = xradcrit_max

# draws the number of sources behind each lens
ycaust_max_arcsec = ycaust_max_samp * kpc2arcsec
lambda_samp = np.pi*ycaust_max_arcsec**2 * nbkg_arcsec2
print(lambda_samp.max(), lambda_samp.mean(), kpc2arcsec*(rein_max_samp[rein_max_samp < 10000000.]).mean())
nsource_samp = np.random.poisson(lam=lambda_samp)
nsource_tot = nsource_samp.sum()
print(nsource_tot)

# generates observational errors on the magnitude of the two images (adds them later)
magA_deltas = np.random.normal(0., mag_err, nsource_tot)
magB_deltas = np.random.normal(0., mag_err, nsource_tot)

# source redshift
zs_allsamp = splev(np.random.rand(nsource_tot), invcumpz_spline)

# source magnitude random sample (to be used only for strongly lensed sources)
tms_allsamp = np.random.rand(nsource_tot)
# obtains ms for every source
ms_allsamp = 0.*tms_allsamp
print('Drawing source magnitudes...')
for i in range(nsource_tot):
    mstar_here = splev(zs_allsamp[i], mstar_sch_spline)
    
    deltams_grid = ms_grid - mstar_here
    deltams_cut = deltams_grid > deltam_min
    deltams_grid = deltams_grid[deltams_cut]
    
    cumphi_grid = splev(deltams_grid, cumfullphi_spline)
    cumphi_grid -= cumphi_grid[0]
    cumphi_grid /= (cumphi_grid[-1] - cumphi_grid[0])
    
    invcumphi_spline = splrep(cumphi_grid, ms_grid[deltams_cut])

    ms_here = splev(tms_allsamp[i], invcumphi_spline)
    ms_allsamp[i] = ms_here

# scale-free source position: uniform distribution in a circle of radius 1
beta_scalefree_samp = np.random.rand(nsource_tot)**0.5

xA_samp = []
xB_samp = []

rein_samp = []
ycaust_samp = []
xradcrit_samp = []

nlensed_sources = []

zs_samp = []
ms_samp = []
beta_samp = []

beta_allsamp = []

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
        reff = 10.**lreff_samp[i]
        nsource_here = nsource_samp[i]
        for n in range(nsource_here):
            beta_here = beta_scalefree_samp[source_count+n] * ycaust_max_samp[i]
            beta_allsamp.append(beta_here)
            #ms_here = mag_samp[source_count+n]
            zs_here = zs_allsamp[source_count+n]
            s_cr_here = splev(zs_here, s_cr_spline)
            # calculates the Einstein radius and radial critical curve and caustic for this source
            rein_here, ycaust_here, xradcrit_here = get_rein_ycaust(s_cr_here, 10.**lm200_samp[i], 10.**lmstar_samp[i], reff, xmin, xmax)
            if beta_here < ycaust_here: # checks if source is actually multiply imaged
                xA, xB, muA, muB = get_images(s_cr_here, 10.**lm200_samp[i], 10.**lmstar_samp[i], reff, beta_here, rein_here, xradcrit_here, xmin, xmax)

                # source magnitude
                mstar_here = splev(zs_here, mstar_sch_spline)
                
                deltams_grid = ms_grid - mstar_here
                deltams_cut = deltams_grid > deltam_min
                deltams_grid = deltams_grid[deltams_cut]
                
                cumphi_grid = splev(deltams_grid, cumfullphi_spline)
                cumphi_grid -= cumphi_grid[0]
                cumphi_grid /= (cumphi_grid[-1] - cumphi_grid[0])
                
                invcumphi_spline = splrep(cumphi_grid, ms_grid[deltams_cut])

                ms_here = splev(tms_allsamp[source_count+n], invcumphi_spline)

                magA = ms_here -2.5*np.log10(muA)
                magB = ms_here -2.5*np.log10(muB)

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
        
                        ms_samp.append(ms_here)
                        zs_samp.append(zs_here)
        
                        beta_samp.append(beta_here)
    
                        rein_samp.append(rein_here)
                        ycaust_samp.append(ycaust_here)
                        xradcrit_samp.append(xradcrit_here)
      
        source_count += nsource_here
    print(i, nsource_samp[i], nlensed_here, islens)

    nlensed_sources.append(nlensed_here)
    lenses.append(islens)

xA_samp = np.array(xA_samp)
xB_samp = np.array(xB_samp)

ms_samp = np.array(ms_samp)
zs_samp = np.array(zs_samp)
beta_samp = np.array(beta_samp)
beta_allsamp = np.array(beta_allsamp)

magA_true = np.array(magA_true)
magB_true = np.array(magB_true)

magA_obs = np.array(magA_obs)
magB_obs = np.array(magB_obs)

nlensed_sources = np.array(nlensed_sources)
lenses = np.array(lenses)

print('%d lenses'%(lenses.sum()))

rein_samp = np.array(rein_samp)
ycaust_samp = np.array(ycaust_samp)
xradcrit_samp = np.array(xradcrit_samp)

lens_ycaust_max_samp = ycaust_max_samp[lenses]

lens_lm200_samp = lm200_samp[lenses]
lens_lmstar_samp = lmstar_samp[lenses]
lens_lreff_samp = lreff_samp[lenses]

output = h5py.File('%s_pop.hdf5'%mockname, 'w')

# grids for computation of stuff
grids = output.create_group('Grids')

grids.create_dataset('ms_grid', data=ms_grid)
grids.create_dataset('zs_grid', data=zs_grid)
grids.create_dataset('pz_grid', data=pz_grid)
grids.create_dataset('cumpz_grid', data=cumpz_grid)
grids.create_dataset('ds_grid', data=ds_grid)
grids.create_dataset('dds_grid', data=dds_grid)
grids.create_dataset('s_cr_grid', data=s_cr_grid)
grids.create_dataset('mstar_sch_grid', data=mstar_sch_grid)
grids.create_dataset('deltam_grid', data=deltam_grid)
grids.create_dataset('cumfullphi_grid', data=cumfullphi_grid)
grids.create_dataset('dvol_dz_grid', data=dvol_dz_grid)

# individual galaxy parameters
galaxies = output.create_group('Galaxies')

galaxies.create_dataset('rein_max', data=rein_max_samp)
galaxies.create_dataset('tein_max', data=rein_max_samp/arcsec2kpc)
galaxies.create_dataset('ycaust_max', data=ycaust_max_samp)
galaxies.create_dataset('xradcrit_max', data=xradcrit_max_samp)
galaxies.create_dataset('llum', data=llum_samp)
galaxies.create_dataset('lmstar', data=lmstar_samp)
galaxies.create_dataset('lustar', data=lustar_samp)
galaxies.create_dataset('lm200', data=lm200_samp)
galaxies.create_dataset('r200', data=r200_samp)
galaxies.create_dataset('lreff', data=lreff_samp)
galaxies.create_dataset('lens', data=lenses)
galaxies.create_dataset('nsource', data=nlensed_sources)

# lens parameters
lensgroup = output.create_group('Lenses')

lensgroup.create_dataset('rein', data=rein_samp)
lensgroup.create_dataset('tein', data=rein_samp/arcsec2kpc)
lensgroup.create_dataset('ycaust', data=ycaust_samp)
lensgroup.create_dataset('xradcrit', data=xradcrit_samp)
lensgroup.create_dataset('xA', data=xA_samp)
lensgroup.create_dataset('xB', data=xB_samp)
lensgroup.create_dataset('zs', data=zs_samp)
lensgroup.create_dataset('ms', data=ms_samp)
lensgroup.create_dataset('magA_true', data=magA_true)
lensgroup.create_dataset('magA_obs', data=magA_obs)
lensgroup.create_dataset('magB_true', data=magB_true)
lensgroup.create_dataset('magB_obs', data=magB_obs)
lensgroup.create_dataset('beta', data=beta_samp)
lensgroup.create_dataset('llum', data=llum_samp[lenses])
lensgroup.create_dataset('lmstar', data=lens_lmstar_samp)
lensgroup.create_dataset('lustar', data=lustar_samp[lenses])
lensgroup.create_dataset('lm200', data=lens_lm200_samp)
lensgroup.create_dataset('r200', data=r200_samp[lenses])
lensgroup.create_dataset('lreff', data=lens_lreff_samp)

# source parameters
sourcegroup = output.create_group('Sources')
sourcegroup.create_dataset('zs', data=zs_allsamp)
sourcegroup.create_dataset('ms', data=ms_allsamp)
sourcegroup.create_dataset('beta', data=beta_allsamp)

# fixed parameters
output.attrs['nbkg'] = nbkg
output.attrs['ngal'] = ngal
output.attrs['Mstar_sch'] = Mstar_sch
output.attrs['alpha_sch'] = alpha_sch
output.attrs['phistar_sch'] = phistar_sch
output.attrs['zd'] = zd
output.attrs['zs_min'] = zs_min
output.attrs['zs_max'] = zs_max
output.attrs['c200'] = c200
output.attrs['maxmagB_det'] = maxmagB_det
output.attrs['ms_min'] = ms_min
output.attrs['ms_max'] = ms_max
output.attrs['dd'] = dd

# hyper-parameters
output.attrs['llum_alpha'] = llum_alpha
output.attrs['llum_llstar'] = llum_llstar
output.attrs['llum_min'] = llum_min
output.attrs['llum_max'] = llum_max
output.attrs['lmstar_piv'] = lmstar_piv
output.attrs['mag_err'] = mag_err
output.attrs['lustar_mu'] = lustar_mu
output.attrs['lustar_sig'] = lustar_sig
output.attrs['lm200_mu'] = lm200_mu
output.attrs['lm200_sig'] = lm200_sig
output.attrs['lm200_beta'] = lm200_beta
output.attrs['lreff_mu'] = lreff_mu
output.attrs['lreff_sig'] = lreff_sig
output.attrs['lreff_beta'] = lreff_beta

