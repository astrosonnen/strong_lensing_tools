import numpy as np
import os
#import glafic
import h5py
from sl_profiles import gnfw, deVaucouleurs as deV
from sl_cosmology import Mpc, c, G, M_Sun
import sl_cosmology
from scipy.optimize import brentq


# sets physical parameters of the gnfw + deV model to get theta_Ein=1''

# primary parameters
omegaM = 0.3
omegaL = 0.7
weos = -1.
hubble = 0.7
prefix = 'composite'
xmin = -3.
ymin = -3.
xmax = 3.
ymax = 3.
pix_ext = 0.2
pix_poi = 0.1
maxlev = 5

#glafic.init(omegaM, omegaL, weos, hubble, prefix, xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb = 0)#, nfw_users=1, flag_hodensity=2, hodensity=200.)

#glafic.reset_par('nfw_users', 1)

# defines the lens parameters
zd = 0.3
zs = 1.5

dd = sl_cosmology.Dang(zd)
ds = sl_cosmology.Dang(zs)
dds = sl_cosmology.Dang(zs, zd)

kpc = Mpc/1000.
arcsec2rad = np.deg2rad(1./3600.)
arcsec2kpc = arcsec2rad * dd * 1000.

s_cr = c**2/(4.*np.pi*G)*ds/dds/dd/Mpc/M_Sun*kpc**2 # critical surface mass density, in M_Sun/kpc**2

# I want a lens with a 1" Einstein radius
rein_arcsec = 1.
rein_phys = rein_arcsec * arcsec2kpc
mein_phys = np.pi * rein_phys**2 * s_cr

# and with Reff = rein
reff_arcsec = rein_arcsec
reff_phys = reff_arcsec * arcsec2kpc

# and with f_dm = 0.5
f_dm = 0.5
mstar_phys = (1. - f_dm) * mein_phys / deV.M2d(reff_phys, reff_phys)

# and with dark matter slope of 1.5
gammadm = 1.5

# and with r_s = 20 * reff
rs_phys = 10. * reff_phys

# Finds the virial mass of the halo
rs_arcsec = rs_phys / arcsec2kpc

mein_dm = f_dm * mein_phys
gnfw_norm = mein_dm / gnfw.M2d(rein_phys, rs_phys, gammadm)

rhoc = sl_cosmology.rhoc(zd)

def r200_zerofunc(r200):
    m3d_here = gnfw_norm * gnfw.M3d(r200, rs_phys, gammadm)
    volume = 4./3.*np.pi * (r200/1000.)**3
    avg_rho = m3d_here / volume
    return avg_rho - 200.*rhoc

r200_phys = brentq(r200_zerofunc, 10., 1000.)
m200_phys = gnfw_norm * gnfw.M3d(r200_phys, rs_phys, gammadm)

m200_glafic = m200_phys * hubble
mstar_glafic = mstar_phys * hubble

print('GNFW component parameters. Mvir=%4.3e, rs=%4.3f arcsec'%(m200_glafic, rs_arcsec))
print('Sersic component parameters. Mtot=%4.3e, reff=%4.3f arcsec'%(mstar_glafic, reff_arcsec))

print('Parameters in physical units. logMstar=%4.3f, reff=%4.3f kpc'%(np.log10(mstar_phys), reff_phys))
def kappa(x, gnfw_norm, rs, gammadm, mstar, reff, s_cr): 
    # dimensionless surface mass density
    return (mstar * deV.Sigma(abs(x), reff) + gnfw_norm * gnfw.fast_Sigma(abs(x), rs, gammadm))/s_cr

kappa_ein = kappa(rein_phys, gnfw_norm, rs_phys, gammadm, mstar_phys, reff_phys, s_cr)
print('kappa at the Einstein radius: %4.3f'%kappa_ein)
 
"""
glafic.startup_setnum(2, 0, 0)

e_arr = [0., 0.1, 0.2, 0.3, 0.4]

glafic.set_lens(1, 'sers', zd, mstar_glafic, 0.0, 0.0, 0., 0.0, reff_arcsec, 4.)
glafic.set_lens(2, 'gnfw', zd, m200_glafic, 0.0, 0.0, 0., 0.0, rs_arcsec, gammadm)

for e in e_arr:
    glafic.reset_par('prefix', 'composite_%2.1f'%e)

    glafic.reset_lens(1, 5, e)
    glafic.reset_lens(2, 5, e)

    glafic.writecrit(1.5)

    # model_init needs to be done again whenever model parameters are changed
    #glafic.model_init(verb = 0)

glafic.quit()
"""


