import pylab
import numpy as np
import h5py
from plotters import probcontour, rgb_to_hex
import os
from qsopars import *
from scipy.interpolate import splev
import sys
import h5py
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18
#parent_color = (1., 0.2, 1.)
parent_color = (0.2, 0.2, 1.)
detect_color = 'k'
tein05_color = 'g'
quad_color = 'r'

# first draws a sample from the parent population
nqso = 100000
zqso_detsamp = splev(np.random.rand(nqso), invcum_zqsodet_spline)
zqso_parsamp = splev(np.random.rand(nqso), invcum_zqso_spline)
t_samp = np.random.rand(nqso)
qsomag_detsamp = np.zeros(nqso)
qsomag_parsamp = np.zeros(nqso)
for i in range(nqso):
    ind = ztoind(zqso_detsamp[i])
    qsomag_detsamp[i] = splev(t_samp[i], invcum_detqso_splines[ind])
    qsomag_parsamp[i] = splev(t_samp[i], invcum_phiqso_splines[ind])

modelname = 'fiducial_1000sqdeg'

lenspop = h5py.File('%s_qsolenses.hdf5'%modelname, 'r')

detect_file = h5py.File('detectable_sources.hdf5', 'r')

tein05 = lenspop['tein_zs'][()] > 0.5
quad = (lenspop['tein_zs'][()] > 0.5) & (lenspop['nimg'][()] > 3)

pars = ['zqso', 'qsomag']
npars = len(pars)

nlens = tein05.sum()
nquad = quad.sum()

parent_samp = {'zqso': zqso_parsamp, 'qsomag': qsomag_parsamp}
detect_samp = {'zqso': zqso_detsamp, 'qsomag': qsomag_detsamp}

tein05_samp = {}
quad_samp = {}
for i in range(npars):
    tein05_samp[pars[i]] = lenspop['%s'%pars[i]][tein05].copy()
    quad_samp[pars[i]] = lenspop['%s'%pars[i]][quad].copy()

nbins = 20

labels = ['$z_{\mathrm{qso}}$', '$m_{\mathrm{qso}}$']

lims = [(0.8, 2.5), (19., 25.3)]

major_step = [1, 2]
minor_step = [0.2, 0.5]

fig = pylab.figure()
pylab.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.99, hspace=0.1, wspace=0.1)
#pylab.figtext(0.45, 0.95, 'Extended model', fontsize=fsize+3, backgroundcolor=(0., 1., 0.))

for i in range(npars):

    ax = fig.add_subplot(npars, npars, (npars+1)*i + 1)

    bins = np.linspace(lims[i][0], lims[i][1], nbins+1)

    sweights = np.ones(nqso)/float(nqso)/(bins[1] - bins[0])
    pylab.hist(parent_samp[pars[i]], bins=bins, color=parent_color, histtype='stepfilled', weights=sweights, linewidth=2, label='Parent pop.')

    dweights = np.ones(nqso)/float(nqso)/(bins[1] - bins[0])
    pylab.hist(detect_samp[pars[i]], bins=bins, color=detect_color, histtype='step', weights=dweights, linewidth=2, label="Detectable")

    lweights = np.ones(nlens)/float(nlens)/(bins[1] - bins[0])
    pylab.hist(tein05_samp[pars[i]], bins=bins, color=tein05_color, histtype='step', weights=lweights, linewidth=2, label="All, $\\theta_{\mathrm{Ein}} > 0.5''$")

    tweights = np.ones(nquad)/float(nquad)/(bins[1] - bins[0])
    pylab.hist(quad_samp[pars[i]], bins=bins, color=quad_color, histtype='step', weights=tweights, linewidth=2, label="Quads, $\\theta_{\mathrm{Ein}} > 1.0''$", linestyle='--')

    if i==0:
        ylim = pylab.ylim()
        pylab.ylim(ylim[0], ylim[1])

        box = ax.get_position()
        ax.legend(loc='upper right', bbox_to_anchor=(2.15, 1.), fontsize=fsize, scatterpoints=3)

    ax.set_xlim((lims[i][0], lims[i][1]))
    ax.tick_params(which='both', direction='in', labelrotation=45, labelsize=fsize)
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

        probcontour(parent_samp[pars[i]], parent_samp[pars[j]], color=parent_color, style='filled', linewidths=2, smooth=3)
        probcontour(detect_samp[pars[i]], detect_samp[pars[j]], color=detect_color, style='lines', linewidths=1, smooth=3)
        probcontour(tein05_samp[pars[i]], tein05_samp[pars[j]], color=tein05_color, style='lines', linewidths=2, smooth=5)
        ax.scatter(quad_samp[pars[i]], quad_samp[pars[j]], color=quad_color)

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

pylab.savefig('../paper/quasar_cornerplot.eps')
pylab.show()

