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

pylab.subplots_adjust(left=0.23, right=1.00, bottom=0.05, top=1., wspace=0., hspace=0.)

ntein = 21
tein_arr = np.linspace(0., 2., ntein)

for n in range(nsims):

    galpop = h5py.File('%s_galaxies.hdf5'%sims[n], 'r')

    laspsmed_gal = np.median(galpop['lasps'][()])
    laspsmed_arr = np.zeros(ntein)
    laspserr_arr = np.zeros(ntein)

    lm200med_gal = mu_h
    lm200med_arr = np.zeros(ntein)
    lm200err_arr = np.zeros(ntein)

    lmdm5med_arr = np.zeros(ntein)
    lmdm5err_arr = np.zeros(ntein)

    def mdm5fitfunc(p):
        return p[0] + p[1] * (galpop['lmobs'][()] - lmobs_piv) + p[2] * (galpop['lreff'][()] - lreff_piv)

    def mdm5errfunc(p):
        return mdm5fitfunc(p) - galpop['lmdm5'][()]

    pmdm5fit = leastsq(mdm5errfunc, [11., 0., 0.])

    lmdm5med_gal = pmdm5fit[0][0]

    gammamed_arr = np.zeros(ntein)
    gammaerr_arr = np.zeros(ntein)

    def gammafitfunc(p):
        return p[0] + p[1] * (galpop['lmobs'][()] - lmobs_piv) + p[2] * (galpop['lreff'][()] - lreff_piv)

    def gammaerrfunc(p):
        return gammafitfunc(p) - galpop['gammadm'][()]

    pgammafit = leastsq(gammaerrfunc, [1.5, 0., 0.])

    gammamed_gal = pgammafit[0][0]

    for i in range(ntein):
        lenscut = galpop['islens'][()] & (galpop['tein_zs'][()] > tein_arr[i])

        nlens = lenscut.sum()
        print('Theta_Ein > %2.1f. %d lenses'%(tein_arr[i], lenscut.sum()))
        laspsmed_arr[i] = np.median(galpop['lasps'][lenscut])
        laspserr_arr[i] = np.std(galpop['lasps'][lenscut])/float(nlens)**0.5

        # fits for the stellar-halo mass relation

        lm200_here = galpop['lm200'][lenscut]
        lmobs_here = galpop['lmobs'][lenscut]
        lreff_here = galpop['lreff'][lenscut]
        lmdm5_here = galpop['lmdm5'][lenscut]
        gammadm_here = galpop['gammadm'][lenscut]

        def fitfunc(p):
            return p[0] + p[1] * (lmobs_here - lmobs_piv)

        def errfunc(p):
            return fitfunc(p) - lm200_here

        pfit = leastsq(errfunc, [13., 1.])

        mu_h_here = pfit[0][0]
        lm200_scat = np.std(fitfunc(pfit[0]) - lm200_here)

        lm200med_arr[i] = mu_h_here
        lm200err_arr[i] = lm200_scat/float(nlens)**0.5

        def mdm5fitfunc(p):
            return p[0] + p[1] * (lmobs_here - lmobs_piv) + p[2] * (lreff_here - lreff_piv)

        def mdm5errfunc(p):
            return mdm5fitfunc(p) - lmdm5_here

        pmdm5fit = leastsq(mdm5errfunc, [11., 0., 0.])
        #print('%s tein_min:%2.1f. log(mdm5) = %4.3f + %4.3f(log(Mstar) - 11.5) + %4.3f(log(Reff) - 1.2)'%(sims[n], tein_arr[i], pmdm5fit[0][0], pmdm5fit[0][1], pmdm5fit[0][2]))

        mu_mdm5_here = pmdm5fit[0][0]
        lmdm5_scat = np.std(mdm5fitfunc(pmdm5fit[0]) - lmdm5_here)

        lmdm5med_arr[i] = mu_mdm5_here
        lmdm5err_arr[i] = lmdm5_scat/float(nlens)**0.5

        def gammafitfunc(p):
            return p[0] + p[1] * (lmobs_here - lmobs_piv) + p[2] * (lreff_here - lreff_piv)

        def gammaerrfunc(p):
            return gammafitfunc(p) - gammadm_here

        pgammafit = leastsq(gammaerrfunc, [11., 0., 0.])

        mu_gamma_here = pgammafit[0][0]
        gamma_scat = np.std(gammafitfunc(pgammafit[0]) - gammadm_here)

        gammamed_arr[i] = mu_gamma_here
        gammaerr_arr[i] = gamma_scat/float(nlens)**0.5

    ax[0].errorbar(tein_arr, laspsmed_arr, yerr=laspserr_arr, color=colseq[n], label=labels[n], linewidth=2)
    ax[1].errorbar(tein_arr, lm200med_arr, yerr=lm200err_arr, color=colseq[n], label=labels[n], linewidth=2)
    ax[2].errorbar(tein_arr, lmdm5med_arr, yerr=lmdm5err_arr, color=colseq[n], linewidth=2)
    ax[2].axhline(lmdm5med_gal, color=colseq[n], linestyle='--')
    ax[3].errorbar(tein_arr, gammamed_arr, yerr=gammaerr_arr, color=colseq[n], linewidth=2, label=labels[n])
    ax[3].axhline(gammamed_gal, color=colseq[n], linestyle='--')

ax[0].set_ylabel('Median $\log{\\alpha_{\mathrm{SPS}}}$', fontsize=fsize)
ax[0].axhline(laspsmed_gal, color='k', linestyle='--', label='Parent pop.')
ax[0].set_ylim(-0.07, 0.28)

ax[0].axhline(-0.04, linestyle=':', color='k')
ax[0].axhline(0.21, linestyle=':', color='k')
ax[0].text(0.2, -0.03, 'Chabrier IMF', fontsize=fsize)
ax[0].text(0.2, 0.22, 'Salpeter IMF', fontsize=fsize)

ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.02))

ax[0].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)

ax[1].axhline(mu_h, color='k', linestyle='--', label='Parent pop.')
#ax[1].set_ylabel('$\mu_{\mathrm{h}}$', fontsize=fsize)
#ax[1].set_ylabel('Mean $\log{M_{\mathrm{h}}}$ at\n$\log{M_*}=11.5$', fontsize=fsize)
ax[1].set_ylabel('$\mu_{\mathrm{h},0}$ (Mean $\log{M_{\mathrm{h}}}$ \n at fixed $M_*^{(\mathrm{obs})}$)', fontsize=fsize)

ax[1].yaxis.set_major_locator(MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))

#ax[2].axhline(lreffmed_gal, color='k', linestyle='--')
#ax[2].set_ylabel('$\mu_{\mathrm{DM},0}$', fontsize=fsize)
#ax[2].set_ylabel('Mean $\log{M_{\mathrm{DM},5}}$ at\n$\log{M_*}=11.5,\log{R_{\mathrm{e}}}=1.2$', fontsize=fsize)
ax[2].set_ylabel('$\mu_{\mathrm{DM},0}$ (Mean $\log{M_{\mathrm{DM},5}}$ \n at fixed $M_*^{(\mathrm{obs})}$, $R_{\mathrm{e}}$)', fontsize=fsize)
#ax[2].text(0.05, 11.22, '$\log_{M_*}=11.5$, $\log_{R_\mathrm{e}}=1.2$', fontsize=fsize)

ax[2].yaxis.set_major_locator(MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(MultipleLocator(0.02))
ax[2].set_ylim(10.98, 11.28)

#ax[3].set_ylabel('$\mu_{\gamma,0}$', fontsize=fsize)
#ax[3].set_ylabel('Mean $\gamma_{\mathrm{DM},5}$ at\n fixed $M_*$, $R_{\mathrm{e}}$', fontsize=fsize)
ax[3].set_ylabel('$\mu_{\gamma,0}$ (Mean $\gamma_{\mathrm{DM},5}$\n at fixed $M_*^{(\mathrm{obs})}$, $R_{\mathrm{e}}$)', fontsize=fsize)
#ax[3].text(0.05, 1.4, '$\log_{M_*}=11.5$, $\log_{R_\mathrm{e}}=1.2$', fontsize=fsize)

ax[3].set_xlabel('Minimum $\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax[3].yaxis.set_major_locator(MultipleLocator(0.05))
ax[3].yaxis.set_minor_locator(MultipleLocator(0.01))

ax[1].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[2].tick_params(axis='both', which='both', direction='in', labelbottom=False, labelsize=fsize, right=True, top=True)
ax[3].tick_params(axis='both', which='both', direction='in', labelsize=fsize, right=True, top=True)

for j in range(4):
    ax[j].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[j].xaxis.set_minor_locator(MultipleLocator(0.1))

ax[1].legend(loc='upper left', fontsize=fsize)

pylab.savefig('../paper/lens_mass_bias.eps')
pylab.show()


