# The SLACS strong lens sample, debiased

## Authors

Alessandro Sonnenfeld, 

## Contents

`SLACS_table.cat`: table with data of the SLACS lenses.
`parent_sample.fits`: .fits file with data of the parent sample.
`full_inference.hdf5`: chain of samples from the posterior probability distribution of the full model.
`slonly_inference.hdf5`: chain of samples from the posterior probability distribution of the lens-only model, with **no** selection function correction.
`full_pp_samples.hdf5`: posterior predicted samples from the full model.
`full_nopfind_pp_samples.hdf5`: posterior predicted samples from the full model, with no lens finding probability term.

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


