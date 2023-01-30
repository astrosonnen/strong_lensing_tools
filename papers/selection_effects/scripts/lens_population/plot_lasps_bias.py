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


fsize = 24

sims = ['fiducial_1000sqdeg', 'highscatter_1000sqdeg', 'lowscatter_1000sqdeg']

labels = ['Fiducial', 'High scatter', 'Low scatter']
nsims = len(sims)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = pylab.subplots(1, 1, figsize=(8, 6))

pylab.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99, wspace=0., hspace=0.)

ntein = 21
tein_arr = np.linspace(0., 2., ntein)

galpop = h5py.File('%s_galaxies.hdf5'%sims[0], 'r')

laspsmed_gal = np.median(galpop['lasps'][()])
laspsmed_arr = np.zeros(ntein)
laspserr_arr = np.zeros(ntein)

lmdm5med_arr = np.zeros(ntein)
lmdm5err_arr = np.zeros(ntein)

def mdm5fitfunc(p):
    return p[0] + p[1] * (galpop['lmobs'][()] - lmobs_piv) + p[2] * (galpop['lreff'][()] - lreff_piv)

def mdm5errfunc(p):
    return mdm5fitfunc(p) - galpop['lmdm5'][()]

pmdm5fit = leastsq(mdm5errfunc, [11., 0., 0.])

lmdm5med_gal = pmdm5fit[0][0]

ax.axhline(laspsmed_gal, color='k', linestyle='--', label='Parent pop.')

for n in range(nsims):

    galpop = h5py.File('%s_galaxies.hdf5'%sims[n], 'r')

    laspsmed_arr = np.zeros(ntein)
    laspserr_arr = np.zeros(ntein)

    for i in range(ntein):
        lenscut = galpop['islens'][()] & (galpop['tein_zs'][()] > tein_arr[i])

        nlens = lenscut.sum()
        print('Theta_Ein > %2.1f. %d lenses'%(tein_arr[i], lenscut.sum()))
        laspsmed_arr[i] = np.median(galpop['lasps'][lenscut])
        laspserr_arr[i] = np.std(galpop['lasps'][lenscut])/float(nlens)**0.5

    ax.errorbar(tein_arr, laspsmed_arr, yerr=laspserr_arr, color=colseq[n], label=labels[n], linewidth=2)

ax.set_ylabel('Median $\log{\\alpha_{\mathrm{SPS}}}$', fontsize=fsize)
ax.set_ylim(-0.07, 0.28)

ax.axhline(-0.04, linestyle=':', color='k')
ax.axhline(0.21, linestyle=':', color='k')
ax.text(0.2, -0.03, 'Chabrier IMF', fontsize=fsize)
ax.text(0.2, 0.22, 'Salpeter IMF', fontsize=fsize)

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

ax.tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)

ax.tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

ax.set_xlabel('Minimum $\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax.legend(loc='lower right', fontsize=fsize, framealpha=1.)

pylab.savefig('../paper/lasps_bias.png')
pylab.show()

