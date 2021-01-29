# scripts to calculate lensing-related quantities for a spherical NFW profile centrally contracted due to the baryonic dstribution
# The profile give the projected surface density and enclosed mass

import os
import numpy as np
from wl_profiles import nfw, sersic
import wl_cosmology
from wl_cosmology import G, f_bar as baryon_fraction, Mpc, M_Sun, default_cosmo
from scipy.integrate import quad
from scipy import interpolate
import h5py


kpc = Mpc/1000.
DM_fraction = 1. - baryon_fraction

# the functions used for the integration
def g_NFW(x):
    """ Returns ln(1+x) - x/(1+x)."""
    return np.log(1+x) - x/(1.+x)

def mass_NFW( r, M200, R200, c200 ):
    """Computes the enclosed mass within r for an NFW profile given the halo mass, M200, the halo radius, R200, and the halo concentration, c200."""
    r2 = np.atleast_1d( r ).copy()
    r2[r2>R200] = R200
    c_r_R = c200 * r / R200
    return M200 * g_NFW(c_r_R) / g_NFW(c200)

def density_NFW( r, M200, c200, Rs):
    """Computes the NFW density at distance r from the halo centre."""
    return M200 / (4.*np.pi * g_NFW(c200) * r * (r+Rs)*(r+Rs) )

def radius_NFW( M200, redshift=0.):  # R200 radius in kpc
    mean_density = 200. * wl_cosmology.rhoc(redshift, cosmo=default_cosmo)
    return ( 3./4./np.pi * M200 / mean_density )**(1./3.) * 1000.

def contract_enclosed_mass( mass_DM, mass_bar, f_bar=0.157 ):
    """ Returns the contracted DM profile given the 'uncontracted' profile and that of the baryonic distribution.
   
   Args:
      mass_DM       : enclosed mass in the DM component in the absence of baryons. 
                          It corresponds to '(1-baryon_fraction) * enclosed mass in
                          DMO (dark matter only) simulations'.
      mass_bar      : enclosed baryonic mass for which to calculate the DM profile.
      f_bar         : optional cosmic baryonic fraction.
   Returns:
      Array of 'contracted' enclosed masses.
   """
    eta_bar = mass_bar / mass_DM * (1.-f_bar) / f_bar  # the last two terms account for transforming the DM mass into the corresponding baryonic mass in DMO simulations
    increase_factor = 0.45 + 0.38 * (eta_bar + 1.16)**0.53
    return mass_DM * increase_factor


def contract_density( density_DM, density_bar, mass_DM, mass_bar, f_bar=0.157 ):
    """ Returns the contracted DM density profile given the 'uncontracted' density and that of the baryonic distribution.
    It uses the differential (d/dr) form of Eq. (11) from Cautun et al (2020).
   
   Args:
      density_DM    : array of DM densities. 
                          It corresponds to '(1-baryon_fraction) * density in
                          DMO (dark matter only) simulations'.
      density_bar   : array of baryonic densities.
      mass_DM       : enclosed mass in the DM component in the absence of baryons. 
                          It corresponds to '(1-baryon_fraction) * enclosed mass in
                          DMO (dark matter only) simulations'.
      mass_bar      : enclosed baryonic mass for which to calculate the DM profile.
      f_bar         : optional cosmic baryonic fraction.
   Returns:
      Array of 'contracted' DM densities.
   """
        
    eta_bar = mass_bar / mass_DM * (1.-f_bar) / f_bar  # the last two terms account for transforming the DM mass into the corresponding baryonic mass in DMO simulations
    first_factor = 0.45 + 0.38 * (eta_bar + 1.16)**0.53
    temp         = density_bar - eta_bar * density_DM * f_bar / (1.-f_bar)
    const_term   = 0.38 * 0.53 * (eta_bar + 1.16)**(0.53-1.) * (1.-f_bar) / f_bar * temp
    
    return density_DM * first_factor + const_term



def calculate_density( r_log_bins, mass_profile ):
    out_shell_mass  = mass_profile[1:] - mass_profile[:-1]
    out_radial_bins = np.sqrt( r_log_bins[1:] * r_log_bins[:-1] )
    out_radial_dr = r_log_bins[1:] - r_log_bins[:-1]
    return out_shell_mass / (4.*np.pi * out_radial_bins**2 * out_radial_dr ), out_radial_bins



def projected_density_integrate( R, func_dens, R200, numBins=1000 ):
    los_dis = np.logspace( -5., 0, numBins, endpoint=False ) * R200
    delta_los = los_dis.copy()
    delta_los[1:] = los_dis[1:] - los_dis[:-1]
    dis  = np.sqrt( R[:,None]**2 + los_dis[None,:]**2 )
    dens = func_dens( dis )
    dens[dis>R200] = 0.
    return 2.*(dens * delta_los[None,:]).sum( axis=1 )

def projected_density_NFW( R, M200, c200, Rs ):
    r = np.atleast_1d( R )
    nfw_norm = M200/nfw.M3d(c200*Rs, Rs)
    return nfw_norm*nfw.Sigma(r, Rs)

def projected_density_NFW_contracted( R, M200, c200, Rs, density_increase ):
    r = np.atleast_1d( R )
    proj_den = r.copy()
    for i in range( 0, len(r) ):
        lim = np.pi / 2. - 1.e-6
        proj_den[i] = quad( lambda theta: density_increase(r[i]/np.cos(theta)) * np.cos(theta)*(np.cos(theta) + r[i]/Rs)**-2, 0., lim, epsrel=1.e-4 )[0]
    return proj_den * M200 / (2.*np.pi * Rs**2 * g_NFW(c200))

"""

# calculate an interpolation table for the contraction
grid_Mstar_M200 = np.logspace( -2.7, -0.2, 26 )
grid_Reff_R200  = np.logspace( -2.6, -0.8, 19 )
grid_r_R200 = np.logspace( -5, 0, 101 )

Sigma_factor_grid = np.zeros( ( len(grid_Mstar_M200), len(grid_Reff_R200), len(grid_r_R200) ), np.float64 )
M2d_factor_grid   = np.zeros( ( len(grid_Mstar_M200), len(grid_Reff_R200), len(grid_r_R200) ), np.float64 )

grid_dir = os.environ.get('BHWLDIR') + '/wl_profiles/'
gridname = grid_dir+'/halo_contraction_grids.hdf5'

if not os.path.isfile(gridname):
    
    # calculate the contraction grids
    ref_Mstar = 10**11.5
    ref_Reff  = 10**0.9
    ref_conc  = 5

    import time
    start = time.time()
    r_log_bins_R200 = np.logspace( -4, 0, 121 )

    for i in range( len(grid_Mstar_M200) ):
        print( i, "out of", len(grid_Mstar_M200), ": ", time.time()-start, "s" )
        Mstar = ref_Mstar
        M200  = Mstar / grid_Mstar_M200[i]
        R200  = radius_NFW( M200 )
        conc  = ref_conc
        Rs    = R200 / conc

        r_log_bins = r_log_bins_R200 * R200
        r_log_bins = r_log_bins[ r_log_bins>.1 ]  # keep only bins at 0.1 kpc or further
        print( "\t", r_log_bins[ [0,-1] ], "kpc" )

        grid_r = grid_r_R200 * R200
        print( "\t", grid_r[ [0,-1] ], "kpc" )

        Mstar_fraction = Mstar / M200 / baryon_fraction
        Mgas_fraction  = 1. - Mstar_fraction
        if Mgas_fraction < 0.: Mgas_fraction = 0.

        for j in range( len(grid_Reff_R200) ):
            Reff = grid_Reff_R200[j] * R200

            # get the enclosed mass in the various components
            temp = mass_NFW( r_log_bins, M200, R200, conc )
            enclosed_mass_DM  = temp * (1. - baryon_fraction)
            enclosed_mass_gas = temp * baryon_fraction * Mgas_fraction
            enclosed_mass_star = sersic.fast_M3d( r_log_bins/Reff, 4) * Mstar

            # calculate the increase in density due to the baryonic contraction
            density_DM_original, radial_bins = calculate_density( r_log_bins, enclosed_mass_DM )
            density_stars, radial_bins = calculate_density( r_log_bins, enclosed_mass_star )
            density_bars, radial_bins  = calculate_density( r_log_bins, enclosed_mass_star+enclosed_mass_gas )

            density_ratio = contract_density( density_DM_original, density_bars, enclosed_mass_DM[1:], \
                                             (enclosed_mass_star+enclosed_mass_gas)[1:] ) \
                            / density_DM_original
            density_ratio[ density_ratio<1 ] = 1.
            density_increase = interpolate.interp1d( radial_bins, density_ratio, bounds_error=False, \
                                                    fill_value=(density_ratio[0], density_ratio[-1]) )

            # calculate the projected densities
            proj_den_orig = projected_density_NFW( grid_r, M200, conc, Rs )
            proj_den_cont = projected_density_NFW_contracted( grid_r, M200, conc, Rs, density_increase )

            Sigma_factor_grid[i,j,:] = proj_den_cont / proj_den_orig
            M2d_factor_grid[i,j,:] = (proj_den_cont*grid_r**2).cumsum() / (proj_den_orig*grid_r**2).cumsum()

    print( "Total computation time:", time.time()-start, " s" )
    
    # save the results to a file
    print("Writing the interpolation grid to:  ", gridname)
    grid_file = h5py.File(gridname, 'w')
    grid_file.create_dataset('grid_Mstar_M200', data=grid_Mstar_M200)
    grid_file.create_dataset('grid_Reff_R200', data=grid_Reff_R200)
    grid_file.create_dataset('grid_r_R200', data=grid_r_R200)

    grid_file.create_dataset('Sigma_factor_grid', data=Sigma_factor_grid)
    grid_file.create_dataset('M2d_factor_grid', data=M2d_factor_grid)

    grid_file.close()

else:
    print("Reading the interpolation grid from:  ", gridname)
    grid_file = h5py.File(gridname, 'r')

    grid_Mstar_M200 = grid_file['grid_Mstar_M200'][()]
    grid_Reff_R200 = grid_file['grid_Reff_R200'][()]
    grid_r_R200 = grid_file['grid_r_R200'][()]

    Sigma_factor_grid = grid_file['Sigma_factor_grid'][()]
    M2d_factor_grid = grid_file['M2d_factor_grid'][()]

    grid_file.close()



import ndinterp
from scipy.interpolate import splrep

axes = {0: splrep( grid_Mstar_M200, np.arange( len(grid_Mstar_M200) ) ), 
        1: splrep( grid_Reff_R200, np.arange( len(grid_Reff_R200) ) ),
        2: splrep( grid_r_R200, np.arange( len(grid_r_R200) ) )
       }

Sigma_factor_interp = ndinterp.ndInterp( axes, Sigma_factor_grid, order=3 )
M2d_factor_interp   = ndinterp.ndInterp( axes, M2d_factor_grid, order=3 )

def fast_M2d_factor( R, Mstar, Mhalo, Reff, R200 ):
    R_R200 = np.atleast_1d( R/R200 )
    Mstar_M200 = np.atleast_1d( Mstar/Mhalo )
    Reff_R200 = np.atleast_1d( Reff/R200 )
    length = len(R_R200)
    sample = np.array( [Mstar_M200*np.ones(length), Reff_R200*np.ones(length), R_R200]).reshape((3, -1)).T
    M2d_factor = M2d_factor_interp.eval( sample )
    return M2d_factor

def fast_Sigma_factor( R, Mstar, Mhalo, Reff, R200 ):
    R_R200 = np.atleast_1d( R/R200 )
    Mstar_M200 = np.atleast_1d( Mstar/Mhalo )
    Reff_R200 = np.atleast_1d( Reff/R200 )
    length = len(R_R200)
    sample = np.array( [Mstar_M200*np.ones(length), Reff_R200*np.ones(length), R_R200]).reshape((3, -1)).T
    Sigma_factor = Sigma_factor_interp.eval( sample )
    return Sigma_factor

def fast_M2d( R, Mstar, Mhalo, Reff, R200, Rs ):
    nfw_norm = Mhalo / gnfw.M3d( R200, Rs, 1.0 )
    nfw_contribution = nfw_norm * gnfw.fast_M2d( R, Rs, 1.0 )
    return nfw_contribution * fast_M2d_factor( R, Mstar, Mhalo, Reff, R200 )

def fast_Sigma( R, Mstar, Mhalo, Reff, R200, Rs ):
    nfw_norm = Mhalo / gnfw.M3d( R200, Rs, 1.0 )
    nfw_contribution = nfw_norm * gnfw.fast_Sigma( R, Rs, 1.0 )
    return nfw_contribution * fast_Sigma_factor( R, Mstar, Mhalo, Reff, R200 )

"""
