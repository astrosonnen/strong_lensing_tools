# Scripts

## Overview

The sets of scripts collected in this folder can be used to carry out the following parts of the analysis:

1. Fit the model to the data.
2. Fit a (wrong) model to the SLACS lenses, without correcting for selection effects.
3. Draw samples of galaxies and strong lenses from the model, given its posterior probability.

## Usage

**Requirements**: 
- Strong lensing tools (i.e. this repository): add to the `PYTHONPATH`
- [this](https://github.com/astrosonnen/spherical_jeans) spherical Jeans code: add to the `PYTHONPATH`.
- `emcee`.
- `h5py`.

### Fitting the full model

1. Run `get_slacs_teincs_grids.py` to predict the Einstein radius and the lensing cross-section as a function of halo mass, stellar mass, and halo contraction efficiency of the SLACS lenses.
2. Run `get_pop_teincs_grids.py` to do the same thing for a large sample of galaxies drawn from the model. These are needed to compute the normalisation of the prior.
3. Run `do_fit.py` to fit the model, with an MCMC.

The final output will be a chain of samples from the posterior, stored in a file named `inference.hdf5`. For each model parameter, the chain is stored in a dataset. It has shape (100, `Nstep`), where the number of rows matches the number of walkers used for the MCMC, and `Nstep` the number of steps in the MCMC (set to 2000 by default).


## Contents

`adcontr_funcs.py`: contains functions used to compute the properties of contracted dark matter halos.

`do_fit.py`: fits the model to the SLACS data.

`do_noseleff_fit.py`: fits the model with no selection effects treatment.

`draw_pp_samples.py`: draws posterior predicted samples from the model.

`fitpars.py`: defines parameters to be used for the strong lensing analysis.

`get_pop_teincs_grids.py`: generates a mock population of galaxies drawn from the model. For each galaxy, computes Einstein radius and lensing cross-section on a grid of halo mass, stellar mass and halo contraction efficiency.

`get_slacs_teincs_grids.py`: computes Einstein radius and lensing cross-section of the SLACS lenses on a grid of halo mass, stellar mass and halo contraction efficiency.

`gnfw_lensingfuncs.py`: functions used to compute lensing properties of deVaucouleurs + gNFW lens models.

`halo_pars.py`: parameters and functions describing the halo mass distribution

`masssize.py`: parameters and functions describing the mass-size relation

`parent_sample_pars.py`: defines parameters describing the parent sample.

`pop_funcs.py`: contains functions used to draw mock galaxy populations from the model distribution.

`read_slacs.py`: reads the table with SLACS observations.


