# Scripts

## Overview

The sets of scripts collected in this folder can be used to carry out the following parts of the analysis:

1. Fit a model for the distribution of density profile parameters to the SLACS and parent sample data, while accounting for selection effects.
2. Draw samples of galaxies and strong lenses from the model, given its posterior probability.
3. Fit a model to the SLACS lenses, without correcting for selection effects.

## Usage

**Requirements**: add the present repository and [this](https://github.com/astrosonnen/spherical_jeans) spherical Jeans code to run to the `PYTHONPATH`; `emcee`; `h5py`.

### Fitting the full model

1. Run `fit_mz_distribution.py` to infer the distribution in stellar mass and redshift of the parent sample galaxies.
2. Run `make_slacs_lensing_grids.py` to fit a power-law model to the Einstein radius of the SLACS lenses, and to obtain the lensing cross-section as a function of model parameters.
3. Run `make_slacs_jeans_grids.py` to obtain the velocity dispersion of SLACS lenses as a function of model parameters.
3. Run `get_slacs_flatprior_grids.py` to fit power-law models to individual lenses, assuming a flat prior on the model parameters.
4. Run `make_rein_grid.py` and `make_jeans_grid.py` to compute Einstein radius and velocity dispersion of galaxies on a grid of model parameters.
5. Run `make_crosssect_grid.py` to get the lensing cross-section on a grid of model parameters.
6. Run `fit_full.py` to fit the model, with an MCMC.

The final output will be a chain of samples from the posterior, stored in a file named `full_inference.hdf5`. For each model parameter, the chain is stored in a dataset. It has shape (100, `Nstep`), where the number of rows matches the number of walkers used for the MCMC, and `Nstep` the number of steps in the MCMC (set to 2000 by default).


## Contents

`parent_sample_pars.py`: defines parameters to be used for the analysis of the parent sample.

`fitpars.py`: defines parameters to be used for the strong lensing analysis.

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

`draw_pp_samples.py`: draws posterior predicted samples from the model.

`draw_nopfind_pp_samples.py`: draws posterior predicted samples from the model, ignoring the lens finding probability term.

