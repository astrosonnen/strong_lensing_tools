# The SLACS strong lens sample, debiased

## Author

Alessandro Sonnenfeld

## Contents

`SLACS_table.cat`: table with data of the SLACS lenses.
`parent_sample.fits`: .fits file with data of the parent sample.
`full_inference.hdf5`: chain of samples from the posterior probability distribution of the full model.
`slonly_inference.hdf5`: chain of samples from the posterior probability distribution of the lens-only model, with **no** selection function correction.
`full_pp_samples.hdf5`: posterior predicted samples from the full model.
`full_nopfind_pp_samples.hdf5`: posterior predicted samples from the full model, with no lens finding probability term. This is used to predict the population of detectable lenses.

`scripts/`: The Python scripts used for the experiments carried out in the paper.

`figures/`: The figures in the paper


### Chain files

File `full_inference.hdf5` contains the chain with samples from the posterior probability. Samples in each of the parameters are stored in separate datasets. The shape of each dataset is (100, 2000), with the first dimension being the number of walkers in the MCMC. The datasets are the following:

| Name | Description |
| ---- | ----------- |
| mu_m5 | Mean m5 at log-stellar mass 11.3 and average size |
| mu_gamma | Mean gamma at log-stellar mass 11.3 and average size |
| beta_m5 | Dependence of m5 on stellar mass |
| xi_m5 | Dependence of m5 on excess size | 
| beta_gamma | Dependence of gamma on stellar mass |
| xi_gamma | Dependence of gamma on excess size | 
| sigma_m5 | Intrinsic scatter in m5 around the mean |
| sigma_gamma | Intrinsic scatter in gamma around the mean |
| t_find | Parameter `theta_0` of the lens finding probability |
| la_find | Log-10 of parameter `a` of the lens finding probability |
| fpfit | Parameters of the fundamental plane fit |
| logp | log of posterior probability |

The file `slonly_inference.hdf5` is similarly structured.

### Posterior predicted samples

The file `full_pp.hdf5` contains posterior predicted samples of properties of parent sample galaxies and strong lenses. It contains 1000 draws from the posterior. The data are organised in groups, as follows:

- `hyperpars`: values of the model parameters.
- `subset`: properties of a random sample of 100 galaxies (different seed for each sample).
- `lenses`: properties of a sample of 59 lenses.
- `pop_sigma_bin`: average properties of parent sample galaxies, in bins of velocity dispersion. Bin edges are defined in `sigma_bins`.
- `pop_ms_bin`: average properties of parent sample galaxies, in bins of log-stellar mass. Bin edges are defined in `ms_bins`.
- `lens_sigma_bin`: average properties of parent sample galaxies, in bins of velocity dispersion.
- `lens_ms_bin`: average properties of parent sample galaxies, in bins of log-stellar mass.

The file `full_nopfind_pp.hdf5` is similarly structured.
