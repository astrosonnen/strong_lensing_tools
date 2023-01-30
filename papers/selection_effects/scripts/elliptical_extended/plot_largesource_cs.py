import numpy as np
import pylab
import h5py
from labellines import labelLine, labelLines
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18

xlim = (0.09, 1.)
#ylim = (0.085, 1.5)

logscale = True

if logscale:
    leftm = 0.15
else:
    leftm = 0.08

cs_file = h5py.File('largesource_crosssect.hdf5', 'r')

ltein_grid = cs_file['ltein_grid'][()]
ntein = len(ltein_grid)

cs_grid = cs_file['cs_grid'][()]

caust_file = h5py.File('quick_caustic_area.hdf5', 'r')
caust_grid = caust_file['cs_grid'][()]

fig, ax = pylab.subplots(1, 1)#, figsize=(8, 4))
pylab.subplots_adjust(left=leftm, right=0.98, bottom=0.14, top=0.97, wspace=0.)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

ax.loglog(10.**ltein_grid, cs_grid[()]/np.pi, linewidth=2, label='Cross-section')
ax.loglog(10.**ltein_grid, caust_grid[()]/np.pi, linestyle='--', color='r', linewidth=2, label='Caustic area')
ax.axhline(1., linestyle=':', color='k', label='Source area')

ax.tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in')

ax.set_xlabel('$\\theta_{\mathrm{Ein}}/\\theta_{\mathrm{s, e}}$', fontsize=fsize)
ax.set_ylabel('$\sigma_{\mathrm{SL}}/(\pi\\theta_{\mathrm{s, e}}^2)$', fontsize=fsize)

ax.axvline(0.1, linestyle='-.', color='k', label='PSF FWHM')

#ax.set_xlim(xlim[0], xlim[1])
#ax.set_ylim(ylim[0], ylim[1])

#ax.loglog(re_max, cs_ps/np.pi, linestyle=':', color='k')

ax.legend(loc='lower right', fontsize=fsize)

pylab.savefig('../../figures/largesource_cs.eps')
pylab.show()


