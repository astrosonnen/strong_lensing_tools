import numpy as np
from scipy.integrate import quad
import h5py
import os
from scipy.interpolate import splrep
import ndinterp

#calculates density profiles, projected mass densities, projected enclosed masses, 3d enclosed masses for generalized-NFW profiles.

grid_dir = os.environ.get('BHWLDIR') + '/wl_profiles/'

def rho(r, rs, beta):
    return 1./r**beta/(1. + r/rs)**(3.-beta) * rs**(beta-3.)

def Sigma(R, rs, beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    norm = 0.5*rs**2
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = (R/rs)**(1-beta)*quad(lambda theta: np.sin(theta)*(np.sin(theta) + R/rs)**(beta-3),0.,np.pi/2.)[0]
    return out/norm

def M2d(R, rs, beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = 2*np.pi*quad(lambda x: Sigma(x, rs, beta)*x, 0., R)[0]
    return out

def M3d(r, rs, beta):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0, len(r)):
        out[i] = 4*np.pi*quad(lambda x: rho(x, rs, beta)*x**2, 0., r[i])[0]
    return out

gridname = grid_dir+'/gNFW_grids.hdf5'

# grid interpolation parameters
bgrid_min = 0.2
bgrid_max = 2.8
Nb = 27

Rgrid_min = 0.001
Rgrid_max = 100.
Nr = 100

beta_grid = np.linspace(bgrid_min, bgrid_max, Nb)
R_grid = np.logspace(np.log10(Rgrid_min), np.log10(Rgrid_max), Nr)

axes = {0: splrep(beta_grid, np.arange(Nb)), 1: splrep(R_grid, np.arange(Nr))}

if not os.path.isfile(gridname):
    # calculates the quantity M2d(R, rs=1, beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.2 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print('calculating grid of projected masses...')
    R,B = np.meshgrid(R_grid, beta_grid)
    Sigma_grid = np.zeros((Nb, Nr))
    M2d_grid = np.zeros((Nb, Nr))
    M3d_grid = np.zeros((Nb, Nr))
    
    for i in range(Nb):
        print('inner slope %4.2f'%beta_grid[i])
        for j in range(Nr):
            Sigma_grid[i,j] = Sigma(R_grid[j], 1., beta_grid[i])
            M2d_grid[i,j] = M2d(R_grid[j], 1., beta_grid[i])
            M3d_grid[i,j] = M3d(R_grid[j], 1., beta_grid[i])

    grid_file = h5py.File(gridname, 'w')
    grid_file.create_dataset('R_grid', data=R_grid)
    grid_file.create_dataset('beta_grid', data=beta_grid)

    grid_file.create_dataset('Sigma_grid', data=Sigma_grid)
    grid_file.create_dataset('M2d_grid', data=M2d_grid)
    grid_file.create_dataset('M3d_grid', data=M3d_grid)

    grid_file.close()

else:
    grid_file = h5py.File(gridname, 'r')

    R_grid = grid_file['R_grid'][()]
    beta_grid = grid_file['beta_grid'][()]
    R,B = np.meshgrid(R_grid, beta_grid)

    Sigma_grid = grid_file['Sigma_grid'][()]
    M2d_grid = grid_file['M2d_grid'][()]
    M3d_grid = grid_file['M3d_grid'][()]

    grid_file.close()

Sigma_interp = ndinterp.ndInterp(axes, Sigma_grid*R, order=3)
M2d_interp = ndinterp.ndInterp(axes, M2d_grid*R**(B-3.), order=3)
M3d_interp = ndinterp.ndInterp(axes, M3d_grid*R**(B-3.), order=3)

def fast_M2d(R, rs, beta):
    R = np.atleast_1d(R)
    rs = np.atleast_1d(rs)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(R), len(rs))
    sample = np.array([beta*np.ones(length), R/rs*np.ones(length)]).reshape((2, length)).T
    M2d = M2d_interp.eval(sample)*(R/rs)**(3.-beta)
    return M2d

def fast_M3d(r, rs, beta):
    r = np.atleast_1d(r)
    rs = np.atleast_1d(rs)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(r), len(rs))
    sample = np.array([beta*np.ones(length), r/rs*np.ones(length)]).reshape((2, length)).T
    M3d = M3d_interp.eval(sample)*(r/rs)**(3.-beta)
    return M3d

def fast_Sigma(R, rs, beta):
    R = np.atleast_1d(R)
    rs = np.atleast_1d(rs)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(R), len(rs))
    sample = np.array([beta*np.ones(length), R/rs*np.ones(length)]).reshape((2, length)).T
    Sigma = Sigma_interp.eval(sample)/(R/rs)/rs**2
    return Sigma

