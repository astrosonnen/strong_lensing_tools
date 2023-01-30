import numpy as np
import pylab
import h5py
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)


fsize = 20

sims = ['fiducial_1000sqdeg', 'highscatter_1000sqdeg', 'lowscatter_1000sqdeg', 'fiducial_1000sqdeg', 'fiducial_1000sqdeg']
sources = ['gal', 'gal', 'gal', 'qso', 'quads']
labels = ['Fiducial', 'High scatter', 'Low scatter', 'Quasars (all)', 'Quasars (quads)']

nsims = len(sims)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = pylab.subplots(1, 1)

pylab.subplots_adjust(left=0.15, right=1.00, bottom=0.14, top=1., wspace=0., hspace=0.)

tein_bins = np.linspace(0., 3., 31)

for n in range(nsims):

    if sources[n] == 'gal':
        lenspop = h5py.File('%s_lenses.hdf5'%sims[n], 'r')
        ax.hist(lenspop['tein_zs'][()], bins=tein_bins, color=colseq[n], histtype='step', label=labels[n], linewidth=2)
    elif sources[n] == 'qso':
        lenspop = h5py.File('%s_qsolenses.hdf5'%sims[n], 'r')
        ax.hist(lenspop['tein_zs'][()], bins=tein_bins, color=colseq[n], histtype='step', label=labels[n], linewidth=2, linestyle='dashed')
    elif sources[n] == 'quads':
        lenspop = h5py.File('%s_qsolenses.hdf5'%sims[n], 'r')
        ax.hist(lenspop['tein_zs'][lenspop['nimg'][()] > 3], bins=tein_bins, color=colseq[n], histtype='step', label=labels[n], linewidth=2, linestyle='dashed')

ax.set_xlabel('$\\theta_{\mathrm{Ein}}$', fontsize=fsize)
ax.set_ylabel('$N$', fontsize=fsize)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))

ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)
ax.legend(loc='upper right', fontsize=fsize)

pylab.savefig('../paper/tein_hist.eps')
pylab.show()


