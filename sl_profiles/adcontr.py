import numpy as np
import pickle
import os
from wl_profiles import nfw, sersic
from scipy.interpolate import splrep, splev, splint
from scipy.misc import derivative
from scipy.optimize import brentq
from scipy.integrate import quad


grid_dir = os.environ.get('BHWLDIR') + '/wl_profiles/'
m2d_gridname = grid_dir+'/adcontr_m2d_coarsendinterp.dat'
sigma_gridname = grid_dir+'/adcontr_sigma_coarsendinterp.dat'

def get_light_m3d_spline(mstar, reff, nser, rmin=1e-4, rmax=1e4, nr=101):

    def rho(r): # stellar density
        deriv = lambda R: -sersic.b(nser)/nser*(R/(reff))**(1/nser)/R*sersic.Sigma(R, nser, reff)
        return -1./np.pi*quad(lambda R: deriv(R)/(R**2 - r**2)**0.5, r, np.inf)[0]

    # deprojects light profile and integrates it to obtain 3d stellar mass profile
    r_light = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    rhos = np.zeros(nr)
    for i in range(nr):
        rhos[i] = rho(r_light[i])
    rs0 = np.append(0., r_light)
    mp0 = np.append(0., 4.*np.pi*rhos*r_light**2)
    
    mprime_light = splrep(rs0, mp0)
    
    m3d_light = np.zeros(nr+1)
    for i in range(nr):
        m3d_light[i+1] = splint(0., r_light[i], mprime_light)
    
    m3d_light_spline = splrep(np.append(0., r_light), m3d_light)
    return m3d_light_spline

def get_m3d_spline(mstar, reff, nser, mvir, rs, cvir, nu, rmin=0.001, rmax=1000., nr=1001, nr_light=101, m3d_light_spline=None):

    if m3d_light_spline is None:
        m3d_light_spline = get_light_m3d_spline(mstar, reff, nser, nr=nr_light)

    # defines grid of shells used to calculate 3d profile of fully adiabatically contracted halo
    nr = 1001
    ri_grid = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    rvir = rs*cvir
    mdmi = mvir*nfw.M3d(ri_grid, rs)/nfw.M3d(rvir, rs) # initial dark matter profile

    mstarf = lambda r: mstar * splev(r, m3d_light_spline)

    rf_grid = 0.*ri_grid

    # calculates final position of each shell, in the maximum contraction case
    for k in range(nr):
        rffunc = lambda r: r*mstarf(r) + r*mdmi[k] - ri_grid[k]*(1. + mstar/mvir)*mdmi[k]
        rmax = ri_grid[k]*(1. + mstar/mvir)
        rf_grid[k] = brentq(rffunc, 0., rmax)

    gam = rf_grid/ri_grid

    rf_here = ri_grid*gam**nu
    monotonic = rf_here[1:] <= rf_here[:-1]
    if monotonic.sum() > 0: # overlapping shells (can happen for large negative values of nu)
        monotonic_rf = [0.]
        monotonic_mdmi = [0.]
        for n in range(nri):
            if rf_here[n] > monotonic_rf[-1]:
                monotonic_rf.append(rf_here[n])
                monotonic_mdmi.append(mdmi[n])
        m3d_dm_spline = splrep(np.array(monotonic_rf), np.array(monotonic_mdmi))
    else:
        m3d_dm_spline = splrep(np.append(0., rf_here), np.append(0., mdmi))

    return m3d_dm_spline

def get_rhor2_spline(mstar, reff, nser, mvir, rs, cvir, nu, rmin=0.001, rmax=1000., nr=1001, m3d_spline=None):

    if m3d_spline is None:
        m3d_spline = get_m3d_spline(mstar, reff, nser, mvir, rs, cvir, nu, rmin=rmin, rmax=rmax, nr=nr)

    ri_grid = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    rhor2 = derivative(lambda r: splev(r, m3d_spline), ri_grid, dx=1e-8)/(4.*np.pi)

    r_here = np.append(0., ri_grid)
    rhor2_here = np.append(0., rhor2)
    rhor2_spline = splrep(r_here, rhor2_here)

    return rhor2_spline

def get_sigmar_spline(mstar, reff, nser, mvir, rs, cvir, nu, r3dmin=0.001, r3dmax=1000., nr3d=1001, r2dmin=0.01, r2dmax=1000., nr2d=101, rhor2_spline=None, zmin=0.001, zmax=1000., nz=1001):

    if rhor2_spline is None:
        rhor2_spline = get_rhor2_spline(mstar, reff, nser, mvir, rs, cvir, nu, rmin=r3dmin, rmax=r3dmax, nr=nr3d)

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

if os.path.isfile(m2d_gridname) and os.path.isfile(sigma_gridname):
    f = open(m2d_gridname, 'r')
    adcontr_m2d_interp = pickle.load(f)
    f.close()
    
    f = open(sigma_gridname, 'r')
    adcontr_sigma_interp = pickle.load(f)
    f.close()

    def M2d_fast(r, fbar, reff, rs, c200, nu):
    
        r = np.atleast_1d(r)
        l = len(r)
    
        point = np.zeros((l, 5))
        point[:, 0] = np.log10(fbar) * np.ones(l)
        point[:, 1] = np.log10(reff/rs) * np.ones(l)
        point[:, 2] = np.log10(c200) * np.ones(l)
        point[:, 3] = nu * np.ones(l)
    
        point[:, 4] = r/rs
    
        return adcontr_m2d_interp.eval(point)
    
    def Sigma_fast(r, fbar, reff, rs, c200, nu):
    
        r = np.atleast_1d(r)
        l = len(r)
    
        point = np.zeros((l, 5))
        point[:, 0] = np.log10(fbar) * np.ones(l)
        point[:, 1] = np.log10(reff/rs) * np.ones(l)
        point[:, 2] = np.log10(c200) * np.ones(l)
        point[:, 3] = nu * np.ones(l)
    
        point[:, 4] = r/rs
    
        return adcontr_sigma_interp.eval(point)

