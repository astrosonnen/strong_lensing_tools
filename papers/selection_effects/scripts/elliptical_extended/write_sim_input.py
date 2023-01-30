import numpy as np
from sl_profiles import sersic
from scipy.special import gamma as gfunc
import os


np.random.seed(0)

# prepares a Glafic input file

nre = 11
logre_grid = np.linspace(-1., 0., nre)

ftot = 200.
pix = 0.05
nser = 1.

f = open('preamble.input', 'r')
prelines = f.readlines()
f.close()

nsource = 10000
    
rmax = 1.5

# draws the same source positions for each value of re, so that
# different simulations can be compared more easily

# generates N sources within a circle of radius 1''
r = np.random.rand(nsource)**0.5 * rmax
phi = 2.*np.pi*np.random.rand(nsource)
x = r * np.cos(phi)
y = r * np.sin(phi)

for i in range(nre):

    re = 10.**logre_grid[i]
    
    prefix = 'ftot%d_logre%2.1f'%(ftot, logre_grid[i])
    
    I0 = ftot/(2.*np.pi*(re/pix)**2*nser/sersic.b(nser)**(2*nser)*gfunc(2.*nser))
    
    lines = prelines.copy()
    
    lines.append('\n')
    lines.append('start_command\n')
    lines.append('\n')
    lines.append('reset_extend 1 2 %f\n'%I0)
    lines.append('reset_extend 1 7 %f\n'%10.**logre_grid[i])
    lines.append('\n')
   
    for n in range(nsource):
        lines.append('reset_par prefix %s/%s_%04d\n'%(prefix, prefix, n))
        lines.append('reset_extend 1 3 %f\n'%(x[n]))
        lines.append('reset_extend 1 4 %f\n'%(y[n]))
        lines.append('writeimage\n')
        #lines.append('reset_par prefix %s_%04d_wnoise\n'%(prefix, n))
        #lines.append('writeimage 0. 1.\n')
        lines.append('\n')
    
    lines.append('quit\n')

    dirname = 'mockdir/%s/'%prefix
    
    if not os.path.isdir(dirname):
        os.system('mkdir %s'%dirname)
    
    f = open('mockdir/%s.input'%prefix, 'w')
    f.writelines(lines)
    f.close()

