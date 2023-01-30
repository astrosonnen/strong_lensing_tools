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

sims = ['fiducial_1000sqdeg', 'fiducial_1000sqdeg', 'fiducial_1000sqdeg']
sources = ['gal', 'qso', 'quads']

labels = ['Extended sources', 'Quasars (all)', 'Quasars (quads)']
nsims = len(sims)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colseq[0], colseq[3], colseq[4]]

fig, ax = pylab.subplots(4, 1, figsize=(6, 13))

pylab.subplots_adjust(left=0.2, right=1.00, bottom=0.05, top=1., wspace=0., hspace=0.)

ntein = 21
tein_arr = np.linspace(0., 2., ntein)

galpop = h5py.File('fiducial_1000sqdeg_galaxies.hdf5', 'r')
laspsmed_gal = np.median(galpop['lasps'][()])
lm200med_gal = mu_h
qmed_gal = np.median(galpop['q'][()])

def mdm5fitfunc(p):
    return p[0] + p[1] * (galpop['lmstar'][()] - lmstar_piv) + p[2] * (galpop['lreff'][()] - lreff_piv)

def mdm5errfunc(p):
    return mdm5fitfunc(p) - galpop['lmdm5'][()]

pmdm5fit = leastsq(mdm5errfunc, [11., 0., 0.])

lmdm5med_gal = pmdm5fit[0][0]

for n in range(nsims):

    if sources[n] == 'gal':
        lenspop = h5py.File('%s_lenses.hdf5'%sims[n], 'r')
        linestyle = 'solid'

    elif sources[n] == 'qso':
        lenspop = h5py.File('%s_qsolenses.hdf5'%sims[n], 'r')
        linestyle = 'dashed'

    elif sources[n] == 'quads':
        lenspop = h5py.File('%s_qsolenses.hdf5'%sims[n], 'r')
        linestyle = 'dashed'

    laspsmed_arr = np.zeros(ntein)
    laspserr_arr = np.zeros(ntein)

    lm200med_arr = np.zeros(ntein)
    lm200err_arr = np.zeros(ntein)

    lmdm5med_arr = np.zeros(ntein)
    lmdm5err_arr = np.zeros(ntein)

    qmed_arr = np.zeros(ntein)
    qerr_arr = np.zeros(ntein)

    for i in range(ntein):
        if sources[n] == 'gal' or sources[n] == 'qso':
            lenscut = lenspop['tein_zs'][()] > tein_arr[i]
        elif sources[n] == 'quads':
            lenscut = (lenspop['tein_zs'][()] > tein_arr[i]) & (lenspop['nimg'][()] > 3)

        nlens = lenscut.sum()
        print('Theta_Ein > %2.1f. %d lenses'%(tein_arr[i], lenscut.sum()))
        laspsmed_arr[i] = np.median(lenspop['lasps'][lenscut])
        laspserr_arr[i] = np.std(lenspop['lasps'][lenscut])/float(nlens)**0.5

        qmed_arr[i] = np.median(lenspop['q'][lenscut])
        qerr_arr[i] = np.std(lenspop['q'][lenscut])/float(nlens)**0.5

        lmstar_here = lenspop['lmstar'][lenscut]
        lm200_here = lenspop['lm200'][lenscut]
        lmdm5_here = lenspop['lmdm5'][lenscut]
        lreff_here = lenspop['lreff'][lenscut]

        # fits for the stellar-halo mass relation

        def fitfunc(p):
            return p[0] + p[1] * (lmstar_here - lmstar_piv)

        def errfunc(p):
            return fitfunc(p) - lm200_here

        pfit = leastsq(errfunc, [13., 1.])

        mu_h_here = pfit[0][0]
        lm200_scat = np.std(fitfunc(pfit[0]) - lm200_here)

        lm200med_arr[i] = mu_h_here
        lm200err_arr[i] = lm200_scat/float(nlens)**0.5

        def mdm5fitfunc(p):
            return p[0] + p[1] * (lmstar_here - lmstar_piv) + p[2] * (lreff_here - lreff_piv)

        def mdm5errfunc(p):
            return mdm5fitfunc(p) - lmdm5_here

        pmdm5fit = leastsq(mdm5errfunc, [11., 0., 0.])

        mu_mdm5_here = pmdm5fit[0][0]
        lmdm5_scat = np.std(mdm5fitfunc(pmdm5fit[0]) - lmdm5_here)

        lmdm5med_arr[i] = mu_mdm5_here
        lmdm5err_arr[i] = lmdm5_scat/float(nlens)**0.5

    ax[0].errorbar(tein_arr, laspsmed_arr, yerr=laspserr_arr, color=colors[n], label=labels[n], linewidth=2, linestyle=linestyle)
    ax[1].errorbar(tein_arr, lm200med_arr, yerr=lm200err_arr, color=colors[n], label=labels[n], linewidth=2, linestyle=linestyle)
    ax[2].errorbar(tein_arr, lmdm5med_arr, yerr=lmdm5err_arr, color=colors[n], linewidth=2, label=labels[n], linestyle=linestyle)
    ax[3].errorbar(tein_arr, qmed_arr, yerr=qerr_arr, color=colors[n], linewidth=2, label=labels[n], linestyle=linestyle)

ax[0].set_ylabel('Median $\log{\\alpha_{\mathrm{SPS}}}$', fontsize=fsize)
ax[0].axhline(laspsmed_gal, color='k', linestyle='--')#, label='Parent population')
ax[0].set_ylim(-0.07, 0.28)

ax[0].axhline(-0.04, linestyle=':', color='k')
ax[0].axhline(0.21, linestyle=':', color='k')
ax[0].text(0.2, -0.03, 'Chabrier IMF', fontsize=fsize)
ax[0].text(0.2, 0.22, 'Salpeter IMF', fontsize=fsize)

ax[0].yaxis.set_major_locator(MultipleLocator(0.05))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.01))

ax[0].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)

ax[1].axhline(mu_h, color='k', linestyle='--')
ax[1].set_ylabel('$\mu_{\mathrm{h},0}$ (Mean $\log{M_{\mathrm{h}}}$ \n at fixed $M_*$)', fontsize=fsize)

ax[1].yaxis.set_major_locator(MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))

ax[2].axhline(lmdm5med_gal, color='k', linestyle='--', label='Parent pop.')
#ax[2].set_ylabel('$\mu_{\mathrm{DM},0}$', fontsize=fsize)
ax[2].set_ylabel('$\mu_{\mathrm{DM},0}$ (Mean $\log{M_{\mathrm{DM},5}}$ \n at fixed $M_*$, $R_{\mathrm{e}}$)', fontsize=fsize)

ax[2].yaxis.set_major_locator(MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(MultipleLocator(0.02))

ax[3].axhline(qmed_gal, color='k', linestyle='--')
ax[3].set_ylabel('Median $q$', fontsize=fsize)
ax[3].set_xlabel('Minimum $\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax[3].yaxis.set_major_locator(MultipleLocator(0.05))
ax[3].yaxis.set_minor_locator(MultipleLocator(0.01))

ax[1].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[2].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[3].tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)

for j in range(4):
    ax[j].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[j].xaxis.set_minor_locator(MultipleLocator(0.1))

ax[2].legend(loc='upper left', fontsize=fsize)

pylab.savefig('../paper/quasar_bias.eps')
pylab.show()


