import pylab
import numpy as np
import h5py
from plotters import probcontour
import os
import sys
import h5py
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 10
#gal_color = (1., 0.2, 1.)
gal_color = (0.2, 0.2, 1.)
lens_color = 'g'
tein1_color = 'r'

modelname = 'fiducial_1000sqdeg'

galpop = h5py.File('%s_galaxies.hdf5'%modelname, 'r')

pars = ['z', 'lmobs', 'lasps', 'lreff', 'q', 'lm200', 'gammadm']

npars = len(pars)

lenses = galpop['islens'][()] & (galpop['tein_zs'][()] > 0.5)
tein1 = galpop['islens'][()] & (galpop['tein_zs'][()] > 1.)

ngal = galpop.attrs['nsamp']
nlens = lenses.sum()
ntein1 = tein1.sum()

gal_samp = {}
lens_samp = {}
tein1_samp = {}
for i in range(npars):
    gal_samp[pars[i]] = galpop['%s'%pars[i]][()].copy()
    lens_samp[pars[i]] = galpop['%s'%pars[i]][lenses].copy()
    tein1_samp[pars[i]] = galpop['%s'%pars[i]][tein1].copy()

nbins = 20

labels = ['$z_{\mathrm{g}}$', '$\log{M_*^{(\mathrm{obs})}}$', '$\log{\\alpha_{\mathrm{SPS}}}$', '$\log{R_{\mathrm{e}}}$', '$q$', '$\log{M_{\mathrm{h}}}$', '$\gamma_{\mathrm{DM}}$']

lims = [(0.1, 0.7), (11., 12.2), (-0.06, 0.28), (0.4, 1.8), (0.2, 1.), (12., 14.5), (1.1, 1.9)]

major_step = [0.2, 0.5, 0.1, 0.5, 0.5, 2, 0.2]
minor_step = [0.05, 0.1, 0.02, 0.1, 0.1, 0.5, 0.05]

fig = pylab.figure()
pylab.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.99, hspace=0.1, wspace=0.1)
#pylab.figtext(0.45, 0.95, 'Extended model', fontsize=fsize+3, backgroundcolor=(0., 1., 0.))

for i in range(npars):

    ax = fig.add_subplot(npars, npars, (npars+1)*i + 1)

    bins = np.linspace(lims[i][0], lims[i][1], nbins+1)

    gweights = np.ones(ngal)/float(ngal)/(bins[1] - bins[0])
    pylab.hist(gal_samp[pars[i]], bins=bins, color=gal_color, histtype='stepfilled', weights=gweights, linewidth=2, label='General population')

    lweights = np.ones(nlens)/float(nlens)/(bins[1] - bins[0])
    pylab.hist(lens_samp[pars[i]], bins=bins, color=lens_color, histtype='step', weights=lweights, linewidth=2, label="Lenses, $\\theta_{\mathrm{Ein}} > 0.5''$")

    tweights = np.ones(ntein1)/float(ntein1)/(bins[1] - bins[0])
    pylab.hist(tein1_samp[pars[i]], bins=bins, color=tein1_color, histtype='step', weights=tweights, linewidth=2, linestyle='--', label="Lenses, $\\theta_{\mathrm{Ein}} > 1.0''$")

    if i==0:
        ylim = pylab.ylim()
        pylab.ylim(ylim[0], ylim[1])

        box = ax.get_position()
        ax.legend(loc='upper right', bbox_to_anchor=(7., 1.), fontsize=fsize, scatterpoints=3)

    ax.set_xlim((lims[i][0], lims[i][1]))
    ax.tick_params(which='both', direction='in', labelrotation=45)
    ax.xaxis.set_major_locator(MultipleLocator(major_step[i]))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_step[i]))

    ax.set_yticks(())
    if i == npars-1:
        ax.set_xlabel(labels[i], fontsize=fsize)
    else:
        ax.tick_params(axis='x', labelbottom=False)

donelabel = False
for j in range(1, npars): # loops over rows
    if j == npars-1:
        xvisible = True
    else:
        xvisible = False

    for i in range(j): # loops over columns
        ax = pylab.subplot(npars, npars, npars*j+i+1)

        probcontour(gal_samp[pars[i]], gal_samp[pars[j]], color=gal_color, style='filled', linewidths=2)
        probcontour(lens_samp[pars[i]], lens_samp[pars[j]], color=lens_color, style='lines', linewidths=2, smooth=5)
        probcontour(tein1_samp[pars[i]], tein1_samp[pars[j]], color=tein1_color, style='lines', linewidths=2, linestyles='--', smooth=7)

        ax.set_xlim(lims[i])
        ax.set_ylim(lims[j])
        ax.tick_params(which='both', direction='in', labelsize=fsize, labelrotation=45)

        ax.xaxis.set_major_locator(MultipleLocator(major_step[i]))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_step[i]))

        ax.yaxis.set_major_locator(MultipleLocator(major_step[j]))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_step[j]))

        if i == 0:
            yvisible = True
            ax.set_ylabel(labels[j], fontsize=fsize)
            #if j == 1:
            #    box = ax.get_position()
            #    ax.legend(loc='upper right', bbox_to_anchor=(5., 2.), fontsize=fsize)

        else:
            ax.tick_params(axis='y', labelleft=False)

        if xvisible:
            ax.set_xlabel(labels[i], fontsize=fsize)
        else:
            ax.tick_params(axis='x', labelbottom=False)

pylab.savefig('../paper/lens_cornerplot.eps')
pylab.show()

