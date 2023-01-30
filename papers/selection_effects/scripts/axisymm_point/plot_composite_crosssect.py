import numpy as np
import pylab
import h5py
from labellines import labelLine, labelLines
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 22
lsize = 18

ylim = (0.01, 3.)

logscale = True

if logscale:
    leftm = 0.14
else:
    leftm = 0.08

cs_file = h5py.File('composite_rein1.0reff_grid.hdf5', 'r')

dms_grid = cs_file['dms_grid'][()]
ndms = len(dms_grid)-1

gammadm_grid = cs_file['gammadm_grid'][()]
ngamma = len(gammadm_grid)

fdm_grid = cs_file['fdm_grid'][()]
nfdm = len(fdm_grid)

cs_gammadm = cs_file['cs_vs_gammadm'][()]
cs_fdm = cs_file['cs_vs_fdm'][()]

fig, ax = pylab.subplots(1, 2, figsize=(8, 4))
pylab.subplots_adjust(left=leftm, right=1.00, bottom=0.16, top=0.99, wspace=0.)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(ndms):
    if i>0:
        ax[0].plot(gammadm_grid, cs_gammadm[:, i+1]/np.pi, label='$\Delta m_s=%2.1f$'%dms_grid[i+1], linewidth=2, color=colseq[i])
        ax[1].plot(fdm_grid, cs_fdm[:, i+1]/np.pi, label='$\Delta m_s=%2.1f$'%dms_grid[i+1], linewidth=2, color=colseq[i])
    else:
        ax[0].plot(gammadm_grid, cs_gammadm[:, i+1]/np.pi, linewidth=2, color=colseq[i])
        ax[1].plot(fdm_grid, cs_fdm[:, i+1]/np.pi, linewidth=2, color=colseq[i])

#lines0 = labelLines(ax[0].get_lines(), xvals = [1.75, 0.9, 1.6, 1.5, 1.4], fontsize=lsize, backgroundcolor='white')
#lines1 = labelLines(ax[1].get_lines(), xvals = [0.25, 0.23, 0.3, 0.4, 0.5], fontsize=lsize, backgroundcolor='white')
lines0 = labelLines(ax[0].get_lines()[1:], xvals = [0.9, 1.6, 1.5, 1.4], fontsize=lsize, backgroundcolor='white')
ax[0].text(1.4, 0.8, '$\Delta m_s = -2.0$', fontsize=lsize, color=colseq[0], rotation=20)
lines1 = labelLines(ax[1].get_lines()[1:], xvals = [0.23, 0.3, 0.4, 0.5], fontsize=lsize, backgroundcolor='white')
ax[1].text(0.1, 1., '$\Delta m_s = -2.0$', fontsize=lsize, color=colseq[0], rotation=-15)

ax[0].xaxis.set_major_locator(MultipleLocator(0.5))
ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))

ax[0].tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in')

ax[0].set_xlabel('$\gamma_{\mathrm{DM}}$', fontsize=fsize)
ax[0].set_ylabel('$\sigma_{\mathrm{SL}}/(\pi\\theta_{\mathrm{Ein}}^2)$', fontsize=fsize)

ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.05))

ax[1].tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in', labelleft=False)

ax[1].set_xlabel('$f_{\mathrm{DM}}$', fontsize=fsize)

ax[0].text(0.5, 1., '$f_{\mathrm{DM}}=0.5$', fontsize=fsize)
ax[1].text(0.55, 1., '$\gamma_{\mathrm{DM}}=1.5$', fontsize=fsize)

ax[0].set_yscale('log')
ax[1].set_yscale('log')

ax[0].set_ylim(ylim[0], ylim[1])
ax[1].set_ylim(ylim[0], ylim[1])

pylab.savefig('../../figures/axisymm_composite_crosssect.eps')
pylab.show()


