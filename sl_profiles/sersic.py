import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc
from scipy.interpolate import splrep, splint
import ndinterp
import h5py


def b(nser):
    return 2*nser - 1./3. + 4/405./nser + 46/25515/nser**2

def L(nser, Re):
    return Re**2*2*np.pi*nser/b(nser)**(2*nser)*gfunc(2*nser)

def Sigma(R, nser, Re):
    return np.exp(-b(nser)*(R/Re)**(1./nser))/L(nser, Re)

def M2d(R, nser, Re):
    R = np.atleast_1d(R)
    out = 0.*R
    for i in range(0, len(R)):
        out[i] = 2*np.pi*quad(lambda r: r*Sigma(r, nser, Re), 0., R[i])[0]
    return out

# grids for fast computation
rgrid_min = 0.001
rgrid_max = 1000.
rgrid_n = 1000

ngrid_min = 0.5
ngrid_max = 20.
ngrid_n = 21

rr = np.logspace(np.log10(rgrid_min), np.log10(rgrid_max), rgrid_n)
nn = np.logspace(np.log10(ngrid_min), np.log10(ngrid_max), ngrid_n)

axes = {0: splrep(rr, np.arange(rgrid_n)), 1: splrep(nn, np.arange(ngrid_n))}

grid2dfilename = os.environ.get('BHWLDIR') + '/wl_profiles/sersic_m2d_grid.hdf5'

if not os.path.isfile(grid2dfilename):
    print('calculating grid of enclosed projected masses...')
    M2d_grid = np.zeros((rgrid_n, ngrid_n))

    for i in range(rgrid_n):
        for j in range(ngrid_n):
            M2d_grid[i, j] = M2d(rr[i], nn[j], 1.)

    grid_file = h5py.File(grid2dfilename, 'w')
    grid_file.create_dataset('grid', data=M2d_grid)
    grid_file.close()
else:
    grid_file = h5py.File(grid2dfilename, 'r')
    M2d_grid = grid_file['grid'][()].copy()
    grid_file.close()

M2d_interp = ndinterp.ndInterp(axes, M2d_grid, order=3)

def fast_M2d(x, nser):

    xarr = np.atleast_1d(x)
    narr = np.atleast_1d(nser)

    xlen = len(xarr)
    nlen = len(narr)

    if xlen == nlen:
        point = np.array((xarr, narr)).reshape((2, xlen)).T
    elif nlen == 1:
        point = np.array((xarr, nser*np.ones(xlen))).reshape((2, xlen)).T
    elif xlen == 1:
        xarr = x*np.ones(nlen)
        point = np.array((xarr, narr)).reshape((2, nlen)).T
    else:
        print('unable to match shapes of x and nser')
        df

    oob = xarr > rgrid_max
    oarr = M2d_interp.eval(point)
    oarr[oob] = 1.

    return oarr

def rho(r, nser, reff): # spherical deprojection
    rhere = np.atleast_1d(r)
    out = 0.*rhere
    deriv = lambda R: -b(nser)/nser*(R/(reff))**(1/nser)/R*Sigma(R, nser, reff)
    for i in range(len(rhere)):
        out[i] = -1./np.pi*quad(lambda R: deriv(R)/(R**2 - rhere[i]**2)**0.5, rhere[i], np.inf)[0]
    return out

grid3dfilename = os.environ.get('BHWLDIR') + '/wl_profiles/sersic_3dgrids.hdf5'

if not os.path.isfile(grid3dfilename):
    print('calculating grid of enclosed 3d masses...')
    M3d_grid = np.zeros((rgrid_n, ngrid_n))
    rho_grid = np.zeros((rgrid_n, ngrid_n))
    rr_ext = np.append(0., rr)

    for j in range(ngrid_n):
        rho_gridhere = np.zeros(rgrid_n+1)
        for i in range(rgrid_n):
            rho_gridhere[i+1] = rho(rr[i], nn[j], 1.)
        mprime_spline = splrep(rr_ext, 4.*np.pi*rho_gridhere*rr_ext**2)
        rho_grid[:, j] = rho_gridhere[1:]
        for i in range(rgrid_n):
            M3d_grid[i, j] = splint(0., rr[i], mprime_spline)

    grid_file = h5py.File(grid3dfilename, 'w')
    grid_file.create_dataset('m3d_grid', data=M3d_grid)
    grid_file.create_dataset('rho_grid', data=rho_grid)
    grid_file.create_dataset('r_grid', data=rr)
    grid_file.create_dataset('nser_grid', data=nn)
    grid_file.close()
else:
    grid_file = h5py.File(grid3dfilename, 'r')
    M3d_grid = grid_file['m3d_grid'][()].copy()
    rho_grid = grid_file['rho_grid'][()].copy()
    grid_file.close()

rho_interp = ndinterp.ndInterp(axes, rho_grid, order=3)
M3d_interp = ndinterp.ndInterp(axes, M3d_grid, order=3)

def fast_M3d(x, nser): # 3d mass enclosed within radius x=r/reff, normalized to unit total mass.

    xarr = np.atleast_1d(x)
    narr = np.atleast_1d(nser)

    xlen = len(xarr)
    nlen = len(narr)

    if xlen == nlen:
        point = np.array((xarr, narr)).reshape((2, xlen)).T
    elif nlen == 1:
        point = np.array((xarr, nser*np.ones(xlen))).reshape((2, xlen)).T
    elif xlen == 1:
        xarr = x*np.ones(nlen)
        point = np.array((xarr, narr)).reshape((2, nlen)).T
    else:
        print('unable to match shapes of x and nser')
        df

    oarr = M3d_interp.eval(point)

    oob_up = xarr > rgrid_max
    oob_dw = xarr < rgrid_min
    oarr[oob_up] = 1.
    oarr[oob_dw] = 0.

    return oarr

def fast_rho(x, nser):

    xarr = np.atleast_1d(x)
    narr = np.atleast_1d(nser)

    xlen = len(xarr)
    nlen = len(narr)

    if xlen == nlen:
        point = np.array((xarr, narr)).reshape((2, xlen)).T
    elif nlen == 1:
        point = np.array((xarr, nser*np.ones(xlen))).reshape((2, xlen)).T
    elif xlen == 1:
        xarr = x*np.ones(nlen)
        point = np.array((xarr, narr)).reshape((2, nlen)).T
    else:
        print('unable to match shapes of x and nser')
        df

    oarr = rho_interp.eval(point)

    oob_up = xarr > rgrid_max
    oob_dw = xarr < rgrid_min
    oarr[oob_up] = 0.
    oarr[oob_dw] = 0.

    return oarr

