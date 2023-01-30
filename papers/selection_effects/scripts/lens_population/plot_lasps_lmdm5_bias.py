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

fig, ax = pylab.subplots(2, 1, figsize=(6, 8))

pylab.subplots_adjust(left=0.2, right=1.00, bottom=0.08, top=1., wspace=0., hspace=0.)

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

ax[0].axhline(laspsmed_gal, color='k', linestyle='--', label='Parent pop.')
ax[1].axhline(lmdm5med_gal, color='k', linestyle='--', label='Parent pop.')

for n in range(nsims):

    galpop = h5py.File('%s_galaxies.hdf5'%sims[n], 'r')

    laspsmed_arr = np.zeros(ntein)
    laspserr_arr = np.zeros(ntein)

    lmdm5med_arr = np.zeros(ntein)
    lmdm5err_arr = np.zeros(ntein)

    for i in range(ntein):
        lenscut = galpop['islens'][()] & (galpop['tein_zs'][()] > tein_arr[i])

        nlens = lenscut.sum()
        print('Theta_Ein > %2.1f. %d lenses'%(tein_arr[i], lenscut.sum()))
        laspsmed_arr[i] = np.median(galpop['lasps'][lenscut])
        laspserr_arr[i] = np.std(galpop['lasps'][lenscut])/float(nlens)**0.5

        # fits for the stellar-halo mass relation

        lmobs_here = galpop['lmobs'][lenscut]
        lreff_here = galpop['lreff'][lenscut]
        lmdm5_here = galpop['lmdm5'][lenscut]

        def mdm5fitfunc(p):
            return p[0] + p[1] * (lmobs_here - lmobs_piv) + p[2] * (lreff_here - lreff_piv)

        def mdm5errfunc(p):
            return mdm5fitfunc(p) - lmdm5_here

        pmdm5fit = leastsq(mdm5errfunc, [11., 0., 0.])

        mu_mdm5_here = pmdm5fit[0][0]
        lmdm5_scat = np.std(mdm5fitfunc(pmdm5fit[0]) - lmdm5_here)

        lmdm5med_arr[i] = mu_mdm5_here
        lmdm5err_arr[i] = lmdm5_scat/float(nlens)**0.5

    ax[0].errorbar(tein_arr, laspsmed_arr, yerr=laspserr_arr, color=colseq[n], label=labels[n], linewidth=2)
    ax[1].errorbar(tein_arr, lmdm5med_arr, yerr=lmdm5err_arr, color=colseq[n], linewidth=2, label=labels[n])

ax[0].set_ylabel('Median $\log{\\alpha_{\mathrm{SPS}}}$', fontsize=fsize)
ax[0].set_ylim(-0.07, 0.28)

ax[0].axhline(-0.04, linestyle=':', color='k')
ax[0].axhline(0.21, linestyle=':', color='k')
ax[0].text(0.2, -0.03, 'Chabrier IMF', fontsize=fsize)
ax[0].text(0.2, 0.22, 'Salpeter IMF', fontsize=fsize)

ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.02))

ax[0].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)

ax[1].set_ylabel('$\mu_{\mathrm{DM},0}$ (Mean $\log{M_{\mathrm{DM},5}}$ \n at fixed $M_*^{(\mathrm{obs})}$, $R_{\mathrm{e}}$)', fontsize=fsize)

ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.02))
ax[1].set_ylim(10.98, 11.28)

ax[1].tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)

for j in range(2):
    ax[j].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[j].xaxis.set_minor_locator(MultipleLocator(0.1))

ax[1].set_xlabel('Minimum $\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax[1].legend(loc='upper left', fontsize=fsize)

pylab.savefig('../paper/lasps_lmdm5_bias.png')
pylab.show()

