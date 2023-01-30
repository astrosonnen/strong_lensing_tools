import numpy as np
import pylab
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
rc('text', usetex=True)


fsize = 18

e_list = [0., 0.1, 0.2, 0.3, 0.4]
ne = len(e_list)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = pylab.subplots(figsize=(5.5, 5.5))

pylab.subplots_adjust(left=0.17, right=1., bottom=0.17, top=1.)

xlim = (-0.95, 0.95)
ylim = (-0.95, 0.95)

"""
# draws a grid
xgrid = np.arange(xlim[0], xlim[1], 0.05)
ygrid = np.arange(ylim[0], ylim[1], 0.05)

for i in range(len(xgrid)):
    ax.axvline(xgrid[i], color='grey', linewidth=0.5)

for i in range(len(ygrid)):
    ax.axhline(ygrid[i], color='grey', linewidth=0.5)
"""

for i in range(ne):

    e_here = e_list[i]
    q_here = 1. - e_here

    f = open('composite_e%2.1f_crit.dat'%e_here)
    table = np.loadtxt(f)
    f.close()

    xs1 = table[:, 2]
    ys1 = table[:, 3]
    xs2 = table[:, 6]
    ys2 = table[:, 7]

    nseg = len(xs1)

    for j in range(nseg):
        if j==0:
            pylab.plot([xs1[j], xs2[j]], [ys1[j], ys2[j]], color=colseq[i], label='$q=%2.1f$'%q_here)
        else:
            pylab.plot([xs1[j], xs2[j]], [ys1[j], ys2[j]], color=colseq[i])

#ax.set_aspect(1.)

ax.tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in')#, width=1)
ax.set_xlabel('$\\beta_x/\\theta_{\mathrm{Ein}}$', fontsize=fsize)
ax.set_ylabel('$\\beta_y/\\theta_{\mathrm{Ein}}$', fontsize=fsize)

ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))

ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])

pylab.arrow(-0.75, 0.78, 0.2, 0, length_includes_head=False, head_width=0.05, color='k')
pylab.text(-0.8, 0.82, 'Major axis', fontsize=fsize)

pylab.legend(loc='lower right', framealpha=1., fontsize=fsize)

pylab.savefig('../../figures/composite_caustics.png')
pylab.show()

