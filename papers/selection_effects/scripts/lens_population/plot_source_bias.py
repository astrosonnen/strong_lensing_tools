import numpy as np
import pylab
import h5py
from scipy.optimize import leastsq
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)


fsize = 20

smag_min = 24.8
smag_max = 25.2

sims = ['fiducial_1000sqdeg', 'highscatter_1000sqdeg', 'lowscatter_1000sqdeg']
labels = ['Fiducial', 'High scatter', 'Low scatter']
nsims = len(sims)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = pylab.subplots(5, 1, figsize=(6, 14))

pylab.subplots_adjust(left=0.2, right=1.00, bottom=0.05, top=1., wspace=0., hspace=0.)

ntein = 21
tein_arr = np.linspace(0., 2., ntein)

detectable = h5py.File('detectable_sources.hdf5', 'r')

zmed_det = np.median(detectable['zs'][()])
smagmed_det = np.median(detectable['smag'][()])

smagbin_det = (detectable['smag'][()] > smag_min) & (detectable['smag'][()] < smag_max)

nsermed_det = np.median(detectable['nser'][smagbin_det])
sreffmed_det = np.median(detectable['sreff'][smagbin_det])

qmed_det = np.median(detectable['sq'][()])

for n in range(nsims):

    lenspop = h5py.File('%s_lenses.hdf5'%sims[n], 'r')

    zmed_arr = np.zeros(ntein)
    zerr_arr = np.zeros(ntein)

    smagmed_arr = np.zeros(ntein)
    smagerr_arr = np.zeros(ntein)

    sreffmed_arr = np.zeros(ntein)
    srefferr_arr = np.zeros(ntein)

    nsermed_arr = np.zeros(ntein)
    nsererr_arr = np.zeros(ntein)

    qmed_arr = np.zeros(ntein)
    qerr_arr = np.zeros(ntein)

    for i in range(ntein):
        teincut = (lenspop['tein_zs'][()] > tein_arr[i])
        nlens = teincut.sum()

        zmed_arr[i] = np.median(lenspop['zs'][teincut])
        zerr_arr[i] = np.std(lenspop['zs'][teincut])/float(nlens)**0.5

        smagmed_arr[i] = np.median(lenspop['smag'][teincut])
        smagerr_arr[i] = np.std(lenspop['smag'][teincut])/float(nlens)**0.5

        smagbin = teincut & (lenspop['smag'][()] > smag_min) & (lenspop['smag'][()] < smag_max)
        nbin = smagbin.sum()

        sreffmed_arr[i] = np.median(lenspop['sreff'][smagbin])
        srefferr_arr[i] = np.std(lenspop['sreff'][smagbin])/float(nbin)**0.5

        nsermed_arr[i] = np.median(lenspop['nser'][smagbin])
        nsererr_arr[i] = np.std(lenspop['nser'][smagbin])/float(nbin)**0.5

        qmed_arr[i] = np.median(lenspop['sq'][teincut])
        qerr_arr[i] = np.std(lenspop['sq'][teincut])/float(nlens)**0.5

    ax[0].errorbar(tein_arr, zmed_arr, yerr=zerr_arr, color=colseq[n], label=labels[n])
    ax[1].errorbar(tein_arr, smagmed_arr, yerr=smagerr_arr, color=colseq[n], label=labels[n])
    ax[2].errorbar(tein_arr, sreffmed_arr, yerr=srefferr_arr, color=colseq[n], label=labels[n])
    ax[3].errorbar(tein_arr, nsermed_arr, yerr=nsererr_arr, color=colseq[n], label=labels[n])
    ax[4].errorbar(tein_arr, qmed_arr, yerr=qerr_arr, color=colseq[n], label=labels[n])

# NEED TO REMOVE INDENTATION
ax[0].axhline(zmed_det, color='k', linestyle='--')#, label='Parent population')
ax[0].set_ylabel('Median $z$', fontsize=fsize)

ax[0].yaxis.set_major_locator(MultipleLocator(0.2))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))

ax[1].axhline(smagmed_det, color='k', linestyle='--', label='Detectable pop.')
ax[1].set_ylabel('Median $m_{\mathrm{s}}$', fontsize=fsize)

ax[1].yaxis.set_major_locator(MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))

ax[2].axhline(sreffmed_det, color='k', linestyle='--')
ax[2].set_ylabel("Median $\\theta_{\mathrm{s}}$\n at $m_{\mathrm{s}}=25\, ('')$", fontsize=fsize)

ax[2].yaxis.set_major_locator(MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(MultipleLocator(0.02))

ax[3].axhline(nsermed_det, color='k', linestyle='--', label='Detectable pop.')
ax[3].set_ylabel("Median $n$\n at $m_{\mathrm{s}}=25$", fontsize=fsize)

ax[3].yaxis.set_major_locator(MultipleLocator(0.2))
ax[3].yaxis.set_minor_locator(MultipleLocator(0.05))

ax[4].axhline(qmed_det, color='k', linestyle='--')
ax[4].set_ylabel("Median $q$", fontsize=fsize)
ax[4].set_xlabel('Minimum $\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax[4].yaxis.set_major_locator(MultipleLocator(0.02))
ax[4].yaxis.set_minor_locator(MultipleLocator(0.005))

ax[0].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
#ax[1].legend(loc='upper left', fontsize=fsize)
ax[3].legend(loc=(0.02, 0.8), fontsize=fsize, framealpha=1.)

ax[1].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[2].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[3].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[4].tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)

for j in range(5):
    ax[j].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[j].xaxis.set_minor_locator(MultipleLocator(0.1))

pylab.savefig('../paper/source_bias.eps')
pylab.show()


