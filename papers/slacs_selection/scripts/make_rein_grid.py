import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
from fitpars import gamma_min, gamma_max
from sl_profiles import powerlaw
import h5py


# Einstein radius of a power-law lens, as a function of s_cr, m5, gamma

# defines lensing-related functions
def alpha(x, s_cr, m5, gamma):
    return 10.**m5 / powerlaw.M2d(5., gamma) * powerlaw.M2d(x, gamma) / np.pi/x/s_cr

xmin = 0.01
xmax = 100000.

grid_file = h5py.File('rein_grid.hdf5', 'w')

m5_min = 10.
m5_max = 12.
nm5 = 101
m5_grid = np.linspace(9., 12., nm5)

ngamma = 17
gamma_grid = np.linspace(gamma_min, gamma_max, ngamma)

s_cr_min = 2.*1e9
s_cr_max = 1e10
nscr = 101
s_cr_grid = np.linspace(s_cr_min, s_cr_max, nscr)

grid_file.create_dataset('m5_grid', data=m5_grid)
grid_file.create_dataset('gamma_grid', data=gamma_grid)
grid_file.create_dataset('s_cr_grid', data=s_cr_grid)

rein_grid = np.zeros((nscr, nm5, ngamma))

for i in range(nscr):
    print(i)
    for j in range(nm5):
        for k in range(ngamma):

            def zerofunc(x):
                return x - alpha(x, s_cr_grid[i], m5_grid[j], gamma_grid[k])

            if zerofunc(xmin) > 0.:
                rein_grid[i, j, k] = 0.
            elif zerofunc(xmax) < 0.:
                rein_grid[i, j, k] = np.inf
                print(i, j, k)
            else:
                rein_grid[i, j, k] = brentq(zerofunc, xmin, xmax)

grid_file.create_dataset('rein_grid', data=rein_grid)

grid_file.close()

