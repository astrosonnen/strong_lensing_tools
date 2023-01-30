import numpy as np
import pylab
import h5py
from labellines import labelLine, labelLines
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18

ylim = (0.03, 30.)

logscale = True

if logscale:
    leftm = 0.08
else:
    leftm = 0.06

cs_file = h5py.File('composite_physical_grid.hdf5', 'r')

dms_grid = cs_file['dms_grid'][1:]
ndms = len(dms_grid)

#lmstar_grid = cs_file['lmstar_grid'][()]
lmstar_grid = np.linspace(11., 12., 11)
#lreff_grid = cs_file['lreff_grid'][()]
lreff_grid = np.linspace(0.5, 1.3, 9)

gammadm_grid = cs_file['gammadm_grid'][()]
ngamma = len(gammadm_grid)

lmdm5_grid = cs_file['lmdm5_grid'][()]
nmdm5 = len(lmdm5_grid)

cs_gammadm = cs_file['cs_vs_gammadm'][:, 1:]
cs_lmdm5 = cs_file['cs_vs_lmdm5'][:, 1:]
cs_lmstar = cs_file['cs_vs_lmstar'][:, 1:]
cs_lreff = cs_file['cs_vs_lreff'][:, 1:]

rein_gammadm = cs_file['rein_vs_gammadm'][()]
rein_lmdm5 = cs_file['rein_vs_lmdm5'][()]
rein_lmstar = cs_file['rein_vs_lmstar'][()]
rein_lreff = cs_file['rein_vs_lreff'][()]

fig, ax = pylab.subplots(1, 4, figsize=(12, 4))
pylab.subplots_adjust(left=leftm, right=1.00, bottom=0.15, top=0.99, wspace=0.)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(ndms):
    ax[0].plot(lmstar_grid, cs_lmstar[:, i], label='$\Delta m_s=%2.1f$'%dms_grid[i], linewidth=2, color=colseq[i])
    ax[1].plot(lreff_grid, cs_lreff[:, i], label='$\Delta m_s=%2.1f$'%dms_grid[i], linewidth=2, color=colseq[i])
    ax[2].plot(gammadm_grid, cs_gammadm[:, i], label='$\Delta m_s=%2.1f$'%dms_grid[i], linewidth=2, color=colseq[i])
    ax[3].plot(lmdm5_grid, cs_lmdm5[:, i], label='$\Delta m_s=%2.1f$'%dms_grid[i], linewidth=2, color=colseq[i])

ax[0].plot(lmstar_grid, np.pi*rein_lmstar**2, linestyle='--', color='k')
ax[1].plot(lreff_grid, np.pi*rein_lreff**2, linestyle='--', color='k')
ax[2].plot(gammadm_grid, np.pi*rein_gammadm**2, linestyle='--', color='k')
ax[3].plot(lmdm5_grid, np.pi*rein_lmdm5**2, linestyle='--', color='k')

ax[0].text(11.2, 4., '$\pi\\theta_{\mathrm{Ein}}^2$', fontsize=fsize, rotation=25)

#lines = labelLines(ax[1].get_lines()[1:], xvals = [0.3,0.3,0.45,0.5,0.55], fontsize=fsize, backgroundcolor='white')

ax[0].xaxis.set_major_locator(MultipleLocator(0.2))
ax[0].xaxis.set_minor_locator(MultipleLocator(0.05))

ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.05))

ax[2].xaxis.set_major_locator(MultipleLocator(0.5))
ax[2].xaxis.set_minor_locator(MultipleLocator(0.1))

ax[3].xaxis.set_major_locator(MultipleLocator(0.2))
ax[3].xaxis.set_minor_locator(MultipleLocator(0.05))

ax[0].tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in')

ax[0].set_xlabel('$\log{M_*}$', fontsize=fsize)
ax[0].set_ylabel('$\sigma_{\mathrm{SL}}$ (arcsec$^2$)', fontsize=fsize)

ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.05))

ax[1].tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in', labelleft=False)
ax[2].tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in', labelleft=False)
ax[3].tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in', labelleft=False)

ax[1].set_xlabel('$\log{R}_e$', fontsize=fsize)
ax[2].set_xlabel('$\gamma_{\mathrm{DM}}$', fontsize=fsize)
ax[3].set_xlabel('$\log{M}_{\mathrm{DM},5}$', fontsize=fsize)

ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')
ax[3].set_yscale('log')

ax[0].set_ylim(ylim[0], ylim[1])
ax[1].set_ylim(ylim[0], ylim[1])
ax[2].set_ylim(ylim[0], ylim[1])
ax[3].set_ylim(ylim[0], ylim[1])

pylab.savefig('../../figures/axisymm_composite_physical_crosssect.eps')
pylab.show()


