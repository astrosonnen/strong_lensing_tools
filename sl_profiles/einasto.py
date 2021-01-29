import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gfunc, gammainc
import pickle
import os
from scipy.interpolate import splrep

# calculates density profiles, projected mass densities, projected enclosed masses, 3d enclosed masses for the Einasto profile.

grid_dir = os.environ.get('BHWLDIR') + '/wl_profiles/'

def rho(r, r2, alpha):
    h = r2/(2./alpha)**(1/alpha)
    mtot = 4.*np.pi*h**3*np.exp(2./alpha)/alpha * gfunc(3./alpha)
    return np.exp(-2./alpha * ((r/r2)**alpha - 1.)) / mtot

def M3d(r, r2, alpha):
    h = r2/(2./alpha)**(1/alpha)
    #norm = 4.*np.pi*h**3*np.exp(2./alpha)/alpha
    #return (gfunc(3./alpha) - gfunc((r/h)**alpha))/norm
    return (gammainc(3./alpha, (r/h)**alpha))
    
def Sigma(R, r2, alpha):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    for i in range(len(Rs)):
        out[i] = 2.*quad(lambda z: rho((Rs[i]**2 + z**2)**0.5, r2, alpha), 0., np.inf)[0]
    return out

def M2d(R, r2, alpha):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    for i in range(len(Rs)):
        out[i] = 2*np.pi*quad(lambda x: Sigma(x, r2, alpha)*x, 0., Rs[i])[0]
    return out

def fast_M2d(R, r2, alpha):
    R = np.atleast_1d(R)
    alpha = np.atleast_1d(alpha)
    length = max(len(alpha), len(R))
    sample = np.array([alpha*np.ones(length), R/r2*np.ones(length)]).reshape((2, length)).T
    M2d = M2d_grid.eval(sample)
    return M2d

def fast_Sigma(R, r2, alpha):
    R = np.atleast_1d(R)
    alpha = np.atleast_1d(alpha)
    length = max(len(alpha), len(R))
    sample = np.array([alpha*np.ones(length), R/r2*np.ones(length)]).reshape((2, length)).T
    Sigma = Sigma_grid.eval(sample)/r2**2
    return Sigma

def make_M2d_grid(Nr=100, Nalpha=9, Rmin=0.01, Rmax=100., alpha_min=0.1, alpha_max=0.5):

    print('calculating grid of enclosed projected masses...')
    import ndinterp
    R_grid = np.logspace(np.log10(Rmin), np.log10(Rmax), Nr)
    R_spline = splrep(R_grid, np.arange(Nr))
    alpha_grid = np.linspace(alpha_min, alpha_max, Nalpha)
    alpha_spline = splrep(alpha_grid, np.arange(Nalpha))
    axes = {0:alpha_spline, 1:R_spline}

    R, A = np.meshgrid(R_grid, alpha_grid)
    M2d_grid = np.empty((Nalpha, Nr))
    
    for i in range(Nalpha):
        print('alpha %3.2f'%alpha_grid[i])
        for j in range(Nr):
            M2d_grid[i,j] = M2d(R_grid[j], 1., alpha_grid[i])
    thing = ndinterp.ndInterp(axes, M2d_grid, order=3)
    f = open(grid_dir+'/einasto_M2d_grid.dat','wb')
    pickle.dump(thing, f)
    f.close()

def make_Sigma_grid(Nr=100, Nalpha=9, Rmin=0.01, Rmax=100., alpha_min=0.1, alpha_max=0.5):

    print('calculating grid of surface mass density...')
    import ndinterp
    R_grid = np.logspace(np.log10(Rmin), np.log10(Rmax), Nr)
    R_spline = splrep(R_grid, np.arange(Nr))
    alpha_grid = np.linspace(alpha_min, alpha_max, Nalpha)
    alpha_spline = splrep(alpha_grid, np.arange(Nalpha))
    axes = {0: alpha_spline, 1: R_spline}

    R, A = np.meshgrid(R_grid, alpha_grid)
    Sigma_grid = np.empty((Nalpha, Nr))
    
    for i in range(Nalpha):
        print('alpha: %3.2f'%alpha_grid[i])
        for j in range(Nr):
            Sigma_grid[i,j] = Sigma(R_grid[j], 1., alpha_grid[i])
    thing = ndinterp.ndInterp(axes, Sigma_grid, order=3)
    f = open(grid_dir+'/einasto_Sigma_grid.dat','wb')
    pickle.dump(thing, f)
    f.close()

if not os.path.isfile(grid_dir+'/einasto_M2d_grid.dat'):
    make_M2d_grid()

if not os.path.isfile(grid_dir+'/einasto_Sigma_grid.dat'):
    make_Sigma_grid()

f = open(grid_dir+'/einasto_M2d_grid.dat','rb')
M2d_grid = pickle.load(f)
f.close()

f = open(grid_dir+'/einasto_Sigma_grid.dat','rb')
Sigma_grid = pickle.load(f)
f.close()

