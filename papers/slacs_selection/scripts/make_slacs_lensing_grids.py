import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from scipy.integrate import quad
from sl_profiles import powerlaw
from sl_cosmology import Sigma_cr
from parent_sample_pars import *
from fitpars import *
import h5py


# Einstein radius, m5 and Jacobian of SLACS power-law lenses, as a function of gamma

f = open('../SLACS_table.cat', 'r')
names = np.loadtxt(f, usecols=(0, ), dtype=str)
f.close()

nslacs = len(names)

f = open('../SLACS_table.cat', 'r')
slacs_zd, slacs_zs, slacs_tein, slacs_rein = np.loadtxt(f, usecols=(3, 4, 7, 8), unpack=True)
f.close()

muB_min = 1.
nbeta = 1001
xmax = 5.*fibre_arcsec

psf_sigma = seeing_arcsec/2.35

# defines lensing-related functions
def alpha(x, tein, gamma):
    return tein * x/abs(x) * (abs(x)/tein)**(2.-gamma)

def kappa(x, tein, gamma): 
    # dimensionless surface mass density
    return (3.-gamma)/2. * (abs(x)/tein)**(1.-gamma)

def mu_r(x, tein, gamma):
    # radial magnification
    return (1. + alpha(x, tein, gamma)/x - 2.*kappa(x, tein, gamma))**(-1)

def mu_t(x, tein, gamma):
    # tangential magnification
    return (1. - alpha(x, tein, gamma)/x)**(-1)

def pl_ycaust(tein, gamma):

    xmin = 0.01

    def radial_invmag(x):
        return 1. + alpha(x, tein, gamma)/x - 2.*kappa(x, tein, gamma)

    # finds the radial caustic
    if radial_invmag(xmin)*radial_invmag(tein) > 0.:
        xradcrit = xmin
    else:
        xradcrit = brentq(radial_invmag, xmin, tein)

    ycaust = -(xradcrit - alpha(xradcrit, tein, gamma))

    return ycaust, xradcrit

dx = 0.01

grid_file = h5py.File('slacs_lensing_grids.hdf5', 'w')

gamma_min = 1.2
gamma_max = 2.8
ngamma = 81
gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

nr = 16
r_grid = np.linspace(0., fibre_arcsec, nr)

grid_file.create_dataset('gamma_grid', data=gamma_grid)

for n in range(nslacs):

    print(names[n])
    group = grid_file.create_group(names[n])

    s_cr = Sigma_cr(slacs_zd[n], slacs_zs[n])
    rein = slacs_rein[n]
    tein = slacs_tein[n]
    mein = np.pi * s_cr * rein**2

    m5_grid = np.zeros(ngamma)
    dm5drein_grid = np.zeros(ngamma)

    mufibre2_cs_grid = np.zeros(ngamma)
    mufibre3_cs_grid = np.zeros(ngamma)

    for j in range(ngamma):
        pl_norm = mein/powerlaw.M2d(rein, gamma_grid[j])
        m5_here = np.log10(pl_norm * powerlaw.M2d(5., gamma_grid[j]))
        m5_grid[j] = m5_here

        rein_up = rein + dx
        mein_up = np.pi * s_cr * rein_up**2
        pl_norm_up = mein_up/powerlaw.M2d(rein_up, gamma_grid[j])
        m5_up = np.log10(pl_norm_up * powerlaw.M2d(5., gamma_grid[j]))

        rein_dw = rein - dx
        mein_dw = np.pi * s_cr * rein_dw**2
        pl_norm_dw = mein_dw/powerlaw.M2d(rein_dw, gamma_grid[j])
        m5_dw = np.log10(pl_norm_dw * powerlaw.M2d(5., gamma_grid[j]))

        dm5drein_grid[j] = (m5_up - m5_dw)/(rein_up - rein_dw)

        ycaust, xradcrit = pl_ycaust(tein, gamma_grid[j])

        xB_grid = np.linspace(-min(fibre_arcsec, tein), -xradcrit, nbeta)
        muB_grid = abs(mu_r(xB_grid, tein, gamma_grid[j]) * mu_t(xB_grid, tein, gamma_grid[j]))
        beta_grid = xB_grid - alpha(xB_grid, tein, gamma_grid[j])

        xA_grid = 0.*beta_grid
        muA_grid = 0.*beta_grid

        muA_seeing_grid = 0.*beta_grid
        muB_seeing_grid = 0.*beta_grid

        for k in range(1, nbeta):
            # solves the lens equation 
            def xA_zerofunc(xA):
                return xA - alpha(xA, tein, gamma_grid[j]) - beta_grid[k]

            if xA_zerofunc(xmax) >= 0.:
                xA_here = brentq(xA_zerofunc, tein, xmax)
                muA_grid[k] = abs(mu_r(xA_here, tein, gamma_grid[j]) * mu_t(xA_here, tein, gamma_grid[j]))

                def muA_func(r, phi):
                    return 1./(2.*np.pi)/psf_sigma**2 * np.exp(-0.5*((r*np.cos(phi) - xA_here)**2 + r**2*np.sin(phi)**2)/psf_sigma**2) * abs(mu_r(xA_here, tein, gamma_grid[j]))

                muA_integrand = np.zeros(nr)
                for l in range(nr):
                    muA_integrand[l] = r_grid[l] * quad(lambda phi: muA_func(r_grid[l], phi), 0., 2.*np.pi)[0] * abs(mu_t(xA_here, tein, gamma_grid[j]))

                muA_spline = splrep(r_grid, muA_integrand)
                muA_seeing_grid[k] = splint(0., r_grid[-1], muA_spline)

                def muB_func(r, phi):
                    return 1./(2.*np.pi)/psf_sigma**2 * np.exp(-0.5*((r*np.cos(phi) - xB_grid[k])**2 + r**2*np.sin(phi)**2)/psf_sigma**2) * abs(mu_r(xB_grid[k], tein, gamma_grid[j]))

                muB_integrand = np.zeros(nr)
                for l in range(nr):
                    muB_integrand[l] = r_grid[l] * quad(lambda phi: muB_func(r_grid[l], phi), 0., 2.*np.pi)[0] * abs(mu_t(xB_grid[k], tein, gamma_grid[j]))

                muB_spline = splrep(r_grid, muB_integrand)
                muB_seeing_grid[k] = splint(0., r_grid[-1], muB_spline)

        mutot_seeing_grid = muA_seeing_grid + muB_seeing_grid

        good = muB_grid > muB_min
        bad = np.logical_not(good)

        integrand_arr = 2.*np.pi*beta_grid
        integrand_arr[bad] = 0.

        integrand_spline = splrep(beta_grid, integrand_arr, k=1)

        good = (mutot_seeing_grid > 2.) & (muB_grid > muB_min)
        bad = np.logical_not(good)

        integrand_arr = 2.*np.pi*beta_grid
        integrand_arr[bad] = 0.

        integrand_spline = splrep(beta_grid, integrand_arr, k=1)
        mufibre2_cs_grid[j] = splint(beta_grid[0], beta_grid[-1], integrand_spline)

        good = (mutot_seeing_grid > 3.) & (muB_grid > muB_min)
        bad = np.logical_not(good)

        integrand_arr = 2.*np.pi*beta_grid
        integrand_arr[bad] = 0.

        integrand_spline = splrep(beta_grid, integrand_arr, k=1)
        mufibre3_cs_grid[j] = splint(beta_grid[0], beta_grid[-1], integrand_spline)

    group.create_dataset('m5_grid', data=m5_grid)
    group.create_dataset('dm5drein_grid', data=dm5drein_grid)
    group.create_dataset('mufibre2_cs_grid', data=mufibre2_cs_grid)
    group.create_dataset('mufibre3_cs_grid', data=mufibre3_cs_grid)

grid_file.close()

