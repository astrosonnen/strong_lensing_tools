import numpy as np
import pylab
import h5py
from labellines import labelLine, labelLines
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc('text', usetex=True)


fsize = 18

ylim = (0.01, 3.)

logscale = True

if logscale:
    leftm = 0.14
else:
    leftm = 0.08

nq = 6
q_arr = np.linspace(0.5, 1., nq)
e_arr = 1. - q_arr

ndms = 5
dms_arr = np.linspace(-2., 2., ndms)

cs_arr = np.zeros((nq, ndms))
cs_quad = np.zeros((nq, ndms))

fig, ax = pylab.subplots(1, 1)#, figsize=(8, 4))
pylab.subplots_adjust(left=leftm, right=0.98, bottom=0.14, top=0.97, wspace=0.)

colseq = pylab.rcParams['axes.prop_cycle'].by_key()['color']

nmocks = 100000

for i in range(nq):

    nmulti = np.zeros(ndms)
    nquads = np.zeros(ndms)

    f = open('composite_e%2.1f_mock.dat'%e_arr[i], 'r')
    lines = f.readlines()[1:]
    f.close()
    
    nimg = 0

    ndet_list = []

    sline = True
    for line in lines:
        if sline:
            nimg = int(line[0])
            nhere = 0
            ndet = np.zeros(ndms)
            if nimg > 0:
                sline = False
            else:
                sline = True
        elif nimg > 1:
            line = line.split()
            mu = float(line[2])
            for j in range(ndms):
                if abs(mu) > 10.**(dms_arr[j]/2.5):
                    ndet[j] += 1
            nhere += 1
            if nhere == nimg:
                sline = True
                ndet_list.append(ndet)
        else:
            sline = True

    ndet_arr = np.array(ndet_list)

    for j in range(ndms):
        nmulti[j] = (ndet_arr[:, j].flatten() > 1).sum()
        nquads[j] = (ndet_arr[:, j].flatten() > 3).sum()

    cs_arr[i, :] = nmulti * 4. / nmocks
    cs_quad[i, :] = nquads * 4. / nmocks

    print(i, nmulti, nquads)

for i in range(ndms):
    if i>1:
        ax.plot(q_arr, cs_arr[:, i]/np.pi, label='$\Delta m_s=%2.1f$'%dms_arr[i], linewidth=2, color=colseq[i])
    else:
        ax.plot(q_arr, cs_arr[:, i]/np.pi, linewidth=2, color=colseq[i])

lines = labelLines(ax.get_lines()[2:], xvals = [0.65, 0.9, 0.58], fontsize=fsize, backgroundcolor='white')
ax.text(0.73, 0.40, '$\Delta m_s=-1.0$', fontsize=fsize, color=colseq[1])
ax.text(0.51, 0.67, '$\Delta m_s=-2.0$', fontsize=fsize, color=colseq[0])

for i in range(ndms):
    ax.plot(q_arr, cs_quad[:, i]/np.pi, label='$\Delta m_s=%2.1f$'%dms_arr[i], linewidth=2, color=colseq[i], linestyle='--')

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

ax.tick_params(axis='both', which='both', top=True, right=True, labelsize=fsize, direction='in')

ax.set_xlabel('$q$', fontsize=fsize)
ax.set_ylabel('$\sigma_{\mathrm{SL}}/(\pi\\theta_{\mathrm{Ein}}^2)$', fontsize=fsize)

ax.set_yscale('log')

ax.set_ylim(0.001, 1.)
ax.set_xlim(0.5, 1.)

ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))

pylab.savefig('../../figures/ell_pnt_cs.eps')
pylab.show()


