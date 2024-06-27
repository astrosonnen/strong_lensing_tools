# Scripts

### Contents

`fit_mz_distribution.py`: fits for the distribution in redshift-stellar mass space of the parent sample.

`mz_distribution.py`: reads the output of fit_mz_distribution.py. Contains the function `draw_mz`, which returns a sample of pair of values (z, mstar) drawn from the redshift-stellar mass distribution.

`make_slacs_lensing_grids.py`: on a grid of values of the total density slope (gamma), it computes, for each SLACS lens:

- The value of the mass enclosed within 5kpc (m5) that matches the Einstein radius
- The Jacobian of the variable change from m5 to the Einstein radius
- The lensing cross-section for the fiducial reference source
- The lensing cross-section for a reference source with intrinsic emission line flux equal to half of the detection limit.

`make_slacs_jeans_grids.py`: on a grid of values of the total density slope (gamma), it integrates the spherical Jeans equation to obtain the model-predicted velocity dispersion within the SDSS spectroscopic aperture. It requires [this](https://github.com/astrosonnen/spherical_jeans) spherical Jeans code to run.

`fit_slonly.py`: fits a model for the distribution in m5 and gamma of the population of **SLACS lenses** to the SLACS data. Does **not** account for selection effects.

`make_rein_grid.py`: computes the Einstein radius of a power-law lens, on a grid of values of the projected mass within 5kpc, the density slope gamma, and the critical surface mass density Sigma_cr.

`make_jeans_grid.py`: computes the seeing-convolved surface brightness-weighted line-of-sight stellar velocity dispersion within the SDSS aperture, on a grid of values of the redshift, density slope and half-light radius, for a power-law lens with unit mass enclosed within 5kpc.

`make_crosssect_grid.py`: computes the lensing cross-section of a powerlaw-lens with a reference source, on a grid of values of the Einstein radius and power-law slope. The reference source has total broadband flux equal to the photometric detection limit, and total emission line flux equal to a third of the spectroscopic detection limit.

`get_slacs_flatprior_grids.py`: computes the posterior probability of lens model parameters of individual SLACS lenses, assuming flat priors on gamma and m5.

`fit_full.py`: fits the full model to the data.

