import numpy as np
import pylab
import h5py
from simpars import *
from scipy.optimize import leastsq
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)


fsize = 20

sims = ['fiducial_1000sqdeg', 'highscatter_1000sqdeg', 'lowscatter_1000sqdeg']
labels = ['Fiducial', 'High scatter', 'Low scatter']
nsims = len(sims)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = pylab.subplots(4, 1, figsize=(6, 13))

pylab.subplots_adjust(left=0.2, right=1.00, bottom=0.05, top=1., wspace=0., hspace=0.)

ntein = 21
tein_arr = np.linspace(0., 2., ntein)

for n in range(nsims):

    galpop = h5py.File('%s_galaxies.hdf5'%sims[n], 'r')

    zmed_gal = np.median(galpop['z'][()])
    zmed_arr = np.zeros(ntein)
    zerr_arr = np.zeros(ntein)

    lmobsmed_gal = np.median(galpop['lmobs'][()])
    lmobsmed_arr = np.zeros(ntein)
    lmobserr_arr = np.zeros(ntein)

    mbin_gal = (galpop['lmobs'][()] > 11.38) & (galpop['lmobs'][()] < 11.42)
    lreffmed_gal = np.median(galpop['lreff'][mbin_gal])
    lreffmed_arr = np.zeros(ntein)
    lrefferr_arr = np.zeros(ntein)

    qmed_gal = np.median(galpop['q'][()])
    qmed_arr = np.zeros(ntein)
    qerr_arr = np.zeros(ntein)

    for i in range(ntein):
        lenscut = galpop['islens'][()] & (galpop['tein_zs'][()] > tein_arr[i])
        nlens = lenscut.sum()
        print('Theta_Ein > %2.1f. %d lenses'%(tein_arr[i], lenscut.sum()))
        zmed_arr[i] = np.median(galpop['z'][lenscut])
        zerr_arr[i] = np.std(galpop['z'][lenscut])/float(nlens)**0.5

        lmobsmed_arr[i] = np.median(galpop['lmobs'][lenscut])
        lmobserr_arr[i] = np.std(galpop['lmobs'][lenscut])/float(nlens)**0.5

        # fits for the mass-size relation

        lreff_here = galpop['lreff'][lenscut]
        lmobs_here = galpop['lmobs'][lenscut]

        def fitfunc(p):
            return p[0] + p[1] * (lmobs_here - lmobs_piv)

        def errfunc(p):
            return fitfunc(p) - lreff_here

        pfit = leastsq(errfunc, [1., 0.6])

        mu_R_here = pfit[0][0]
        logr_scat = np.std(fitfunc(pfit[0]) - lreff_here)

        lreffmed_arr[i] = mu_R_here
        lrefferr_arr[i] = logr_scat/float(nlens)**0.5

        qmed_arr[i] = np.median(galpop['q'][lenscut])
        qerr_arr[i] = np.std(galpop['q'][lenscut])/float(nlens)**0.5

    ax[0].errorbar(tein_arr, zmed_arr, yerr=zerr_arr, color=colseq[n], label=labels[n], linewidth=2)
    ax[1].errorbar(tein_arr, lmobsmed_arr, yerr=lmobserr_arr, color=colseq[n], label=labels[n], linewidth=2)
    ax[2].errorbar(tein_arr, lreffmed_arr, yerr=lrefferr_arr, color=colseq[n], linewidth=2, label=labels[n])
    ax[3].errorbar(tein_arr, qmed_arr, yerr=qerr_arr, color=colseq[n], linewidth=2)

# NEED TO REMOVE INDENTATION
ax[0].axhline(zmed_gal, color='k', linestyle='--')#, label='Parent population')
ax[0].set_ylabel('Median $z_{\mathrm{g}}$', fontsize=fsize)

ax[0].yaxis.set_major_locator(MultipleLocator(0.05))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.01))

ax[1].axhline(lmobsmed_gal, color='k', linestyle='--')
ax[1].set_ylabel('Median $\log{M_*^{(\mathrm{obs})}}$', fontsize=fsize)

ax[1].yaxis.set_major_locator(MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))

ax[2].axhline(lreffmed_gal, color='k', linestyle='--', label='Parent pop.')
#ax[2].set_ylabel('$\mu_{\mathrm{R},0}$', fontsize=fsize)
ax[2].set_ylabel('Median $\log{R_{\mathrm{e}}}$\n at $\log{M_*^{(\mathrm{obs})}}=11.4$', fontsize=fsize)

ax[2].yaxis.set_major_locator(MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(MultipleLocator(0.02))

ax[3].axhline(qmed_gal, color='k', linestyle='--')
ax[3].set_ylabel('Median $q_{\mathrm{g}}$', fontsize=fsize)
ax[3].set_xlabel('Minimum $\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax[3].yaxis.set_major_locator(MultipleLocator(0.02))
ax[3].yaxis.set_minor_locator(MultipleLocator(0.005))

ax[0].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[2].legend(loc='lower left', fontsize=fsize)

ax[1].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[2].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[3].tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)

for j in range(4):
    ax[j].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[j].xaxis.set_minor_locator(MultipleLocator(0.1))

pylab.savefig('../paper/lens_observable_bias.eps')
pylab.show()


