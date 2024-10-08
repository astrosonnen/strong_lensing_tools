import numpy as np
import h5py
import os
import ndinterp
from sl_profiles import nfw, sersic, deVaucouleurs as deV
from scipy.interpolate import splrep, splev, splint
from scipy.misc import derivative
from scipy.optimize import brentq
from scipy.integrate import quad


# Adiabatically contracted dark matter profiles, with deVaucouleurs baryonic profile

thisdir = os.path.dirname(os.path.abspath(__file__))
m2d_gridname = thisdir+'/adcontr_deV_m2d_grid.hdf5'
sigma_gridname = thisdir+'/adcontr_deV_sigma_grid.hdf5'

Rgrid_min = 0.001
Rgrid_max = 100.
nR = 101
R_grid = np.logspace(np.log10(Rgrid_min), np.log10(Rgrid_max), nR)

r3dgrid_min = 0.1*Rgrid_min
r3dgrid_max = 10.*Rgrid_max
nr3d = 1001
r3d_grid = np.logspace(np.log10(r3dgrid_min), np.log10(r3dgrid_max), nr3d)

nu_min = -1.
nu_max = 1.
nnu = 11
nu_grid = np.linspace(nu_min, nu_max, nnu)

reff_min = 1e-2
reff_max = 1.
nreff = 10
reff_grid = np.logspace(np.log10(reff_min), np.log10(reff_max), nreff)

cvir_min = 1.
cvir_max = 10.
ncvir = 10
cvir_grid = np.logspace(np.log10(cvir_min), np.log10(cvir_max), ncvir)

fbar_min = 0.
fbar_max = 0.8
nfbar = 9
fbar_grid = np.linspace(fbar_min, fbar_max, nfbar)

def get_m3d_spline(mstar, reff, cvir, nu, ri_grid=r3d_grid):

    # defines grid of shells used to calculate 3d profile of fully adiabatically contracted halo
    # masses are in units of the virial mass 
    # lengths are in units of rs

    nr = len(ri_grid)

    mdmi = nfw.M3d(ri_grid, 1.)/nfw.M3d(cvir, 1.) # initial dark matter profile

    mstarf = lambda r: mstar * deV.fast_M3d(r/reff)

    rf_grid = 0.*ri_grid

    # calculates final position of each shell, in the maximum contraction case
    last_rf = 0.
    for k in range(nr):
        rffunc = lambda r: r*mstarf(r) + r*mdmi[k] - ri_grid[k]*(1. + mstar)*mdmi[k]
        rmax = ri_grid[k]*(1. + mstar)
        #rf_grid[k] = brentq(rffunc, last_rf, rmax)
        rf_grid[k] = brentq(rffunc, 0., rmax)
        last_rf = rf_grid[k]

    gam = rf_grid/ri_grid

    rf_here = ri_grid*gam**nu
    monotonic = rf_here[1:] <= rf_here[:-1]
    if monotonic.sum() > 0: # overlapping shells (can happen for large negative values of nu)
        monotonic_rf = [0.]
        monotonic_mdmi = [0.]
        for n in range(nr):
            if rf_here[n] > monotonic_rf[-1]:
                monotonic_rf.append(rf_here[n])
                monotonic_mdmi.append(mdmi[n])
        m3d_dm_spline = splrep(np.array(monotonic_rf), np.array(monotonic_mdmi))
    else:
        m3d_dm_spline = splrep(np.append(0., rf_here), np.append(0., mdmi))

    return m3d_dm_spline

def get_rhor2_spline(mstar, reff, cvir, nu, ri_grid=r3d_grid, m3d_spline=None):

    if m3d_spline is None:
        m3d_spline = get_m3d_spline(mstar, reff, cvir, nu, ri_grid=ri_grid)

    rhor2 = derivative(lambda r: splev(r, m3d_spline), ri_grid, dx=1e-8)/(4.*np.pi)

    r_here = np.append(0., ri_grid)
    rhor2_here = np.append(0., rhor2)
    rhor2_spline = splrep(r_here, rhor2_here)

    return rhor2_spline

def get_sigmar_spline(mstar, reff, mvir, rs, cvir, nu, r3dmin=0.001, r3dmax=1000., nr3d=1001, r2dmin=0.01, r2dmax=1000., nr2d=101, rhor2_spline=None, zmin=0.001, zmax=1000., nz=1001):

    if rhor2_spline is None:
        rhor2_spline = get_rhor2_spline(mstar, reff, mvir, rs, cvir, nu, rmin=r3dmin, rmax=r3dmax, nr=nr3d)

    R_grid = np.logspace(np.log10(r2dmin), np.log10(r2dmax), nr2d)

    sigma_grid_here = 0.*R_grid
    z_arr = np.logspace(np.log10(zmin), np.log10(zmax), nz)
    for k in range(nr2d):
        r_arr = (z_arr**2 + R_grid[k]**2)**0.5
        integrand_spline = splrep(z_arr, splev(r_arr, rhor2_spline)/r_arr**2)
        sigma_grid_here[k] = 2.*splint(0., zmax, integrand_spline)

    sigmar_spline = splrep(np.append(0., R_grid), np.append(0., R_grid*sigma_grid_here))
    return sigmar_spline

def get_m2d_spline(mstar, reff, nser, mvir, rs, cvir, nu, nr3d=1001, r2dmin=0.01, r2dmax=1000., nr2d=101, sigmar_spline=None):

    if sigmar_spline is None:
        sigmar_spline = get_sigmar_spline(mstar, reff, nser, mvir, rs, cvir, nu, r2dmin=r2dmin, r2dmax=r2dmax, nr2d=nr2d)

    R_grid = np.logspace(np.log10(r2dmin), np.log10(r2dmax), nr2d)
    m2d_grid = 0.*R_grid

    for i in range(nr2d):
        m2d_grid[i] = 2.*np.pi*splint(0., R_grid[i], sigmar_spline)

    m2d_spline = splrep(R_grid, m2d_grid)

    return m2d_spline

if not os.path.isfile(sigma_gridname):
    # makes a grid of surface mass density profile
    sigma_grid = np.zeros((nfbar, nreff, ncvir, nnu, nR))

    print('Computing grid of surface mass density...')
    for i in range(nfbar):
        print('%d/%d'%(i+1, nfbar))
        # stellar mass in units of the virial mass
        mstar = fbar_grid[i]/(1. - fbar_grid[i]) 

        for j in range(nreff):
            print('   %d/%d'%(j+1, nreff))
            for k in range(ncvir):
                print('      %d/%d'%(k+1, ncvir))
                for l in range(nnu):
                    rhor2_spline = get_rhor2_spline(mstar, reff_grid[j], cvir_grid[k], nu_grid[l], ri_grid=r3d_grid)
                    z_arr = np.logspace(np.log10(0.1*Rgrid_min), np.log10(10.*Rgrid_max), 1001)
                    for m in range(nR):
                        r_arr = (z_arr**2 + R_grid[m]**2)**0.5
                        integrand_spline = splrep(z_arr, splev(r_arr, rhor2_spline)/r_arr**2)
                        sigma_grid[i, j, k, l, m] = 2.*splint(0., z_arr[-1], integrand_spline)

    output_file = h5py.File(sigma_gridname, 'w')
    output_file.create_dataset('fbar_grid', data=fbar_grid)
    output_file.create_dataset('reff_grid', data=reff_grid)
    output_file.create_dataset('cvir_grid', data=cvir_grid)
    output_file.create_dataset('nu_grid', data=nu_grid)
    output_file.create_dataset('R_grid', data=R_grid)
    output_file.create_dataset('Sigma_grid', data=sigma_grid)
    output_file.close()

sigma_gridfile = h5py.File(sigma_gridname, 'r')
    
R_sigmagrid = sigma_gridfile['R_grid'][()]
fbar_sigmagrid = sigma_gridfile['fbar_grid'][()]
reff_sigmagrid = sigma_gridfile['reff_grid'][()]
cvir_sigmagrid = sigma_gridfile['cvir_grid'][()]
nu_sigmagrid = sigma_gridfile['nu_grid'][()]
sigma_grid = sigma_gridfile['Sigma_grid'][()]

axes = {0: splrep(fbar_sigmagrid, np.arange(len(fbar_sigmagrid))),\
        1: splrep(reff_sigmagrid, np.arange(len(reff_sigmagrid))),\
        2: splrep(cvir_sigmagrid, np.arange(len(cvir_sigmagrid))),\
        3: splrep(nu_sigmagrid, np.arange(len(nu_sigmagrid))),\
        4: splrep(R_sigmagrid, np.arange(len(R_sigmagrid)))}

sigma_interp = ndinterp.ndInterp(axes, sigma_grid)

if not os.path.isfile(m2d_gridname):

    m2d_grid = np.zeros((nfbar, nreff, ncvir, nnu, nR))

    print('Computing grid of enclosed mass...')
    for i in range(nfbar):
        print('%d/%d'%(i+1, nfbar))
        mstar = fbar_grid[i]/(1. - fbar_grid[i]) 

        for j in range(nreff):
            print('   %d/%d'%(j+1, nreff))
            for k in range(ncvir):
                print('      %d/%d'%(k+1, ncvir))
                for l in range(nnu):
                    rsigma_spline = splrep(np.append(0., R_grid), np.append(0., R_grid * sigma_grid[i, j, k, l]))
                    for m in range(nR):
                        m2d_grid[i, j, k, l, m] = 2.*np.pi*splint(0., R_grid[m], rsigma_spline)

    output_file = h5py.File(m2d_gridname, 'w')
    output_file.create_dataset('fbar_grid', data=fbar_grid)
    output_file.create_dataset('reff_grid', data=reff_grid)
    output_file.create_dataset('cvir_grid', data=cvir_grid)
    output_file.create_dataset('nu_grid', data=nu_grid)
    output_file.create_dataset('R_grid', data=R_grid)
    output_file.create_dataset('M2d_grid', data=m2d_grid)
    output_file.close()

m2d_gridfile = h5py.File(m2d_gridname, 'r')
    
R_m2dgrid = m2d_gridfile['R_grid'][()]
fbar_m2dgrid = m2d_gridfile['fbar_grid'][()]
reff_m2dgrid = m2d_gridfile['reff_grid'][()]
cvir_m2dgrid = m2d_gridfile['cvir_grid'][()]
nu_m2dgrid = m2d_gridfile['nu_grid'][()]
m2d_grid = m2d_gridfile['M2d_grid'][()]

axes = {0: splrep(fbar_m2dgrid, np.arange(len(fbar_m2dgrid))),\
        1: splrep(reff_m2dgrid, np.arange(len(reff_m2dgrid))),\
        2: splrep(cvir_m2dgrid, np.arange(len(cvir_m2dgrid))),\
        3: splrep(nu_m2dgrid, np.arange(len(nu_m2dgrid))),\
        4: splrep(R_m2dgrid, np.arange(len(R_m2dgrid)))}

m2d_interp = ndinterp.ndInterp(axes, m2d_grid)

def fast_Sigma(r, fbar, reff, rs, cvir, nu):

    r = np.atleast_1d(r)
    fbar = np.atleast_1d(fbar)
    reff = np.atleast_1d(reff)
    rs = np.atleast_1d(rs)
    cvir = np.atleast_1d(cvir)
    nu = np.atleast_1d(nu)

    l = max(len(r), len(fbar), len(reff), len(rs), len(cvir), len(nu))

    point = np.array([fbar*np.ones(l), reff/rs*np.ones(l), cvir*np.ones(l), nu*np.ones(l), r/rs*np.ones(l)]).T

    return sigma_interp.eval(point) / rs**2

def fast_M2d(r, fbar, reff, rs, cvir, nu):

    r = np.atleast_1d(r)
    fbar = np.atleast_1d(fbar)
    reff = np.atleast_1d(reff)
    rs = np.atleast_1d(rs)
    cvir = np.atleast_1d(cvir)
    nu = np.atleast_1d(nu)

    l = max(len(r), len(fbar), len(reff), len(rs), len(cvir), len(nu))

    point = np.array([fbar*np.ones(l), reff/rs*np.ones(l), cvir*np.ones(l), nu*np.ones(l), r/rs*np.ones(l)]).T

    return m2d_interp.eval(point)


