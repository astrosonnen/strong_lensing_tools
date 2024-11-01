# SLACS debiased, II. Lensing-only constraints on the stellar IMF and dark matter contraction in early-type galaxies

## Author

Alessandro Sonnenfeld

## Contents

`SLACS_table.cat`: table with data of the SLACS lenses.
`parent_sample.fits`: .fits file with data of the parent sample.
`inference.hdf5`: chain of samples from the posterior probability distribution of the full model.
`slonly_inference.hdf5`: chain of samples from the posterior probability distribution of the lens-only model, with **no** selection function correction.
`pp_samples.hdf5`: posterior predicted samples from the full model.

`scripts/`: The Python scripts used for the experiments carried out in the paper.

`figures/`: The figures in the paper


### Chain files

File `inference.hdf5` contains the chain with samples from the posterior probability. Samples in each of the parameters are stored in separate datasets. The shape of each dataset is (100, 2000), with the first dimension being the number of walkers in the MCMC. The datasets are the following:

| Name | Description |
| ---- | ----------- |
| lasps | Log-stellar population synthesis mismatch parameter |
| eps | Dark matter contraction efficiency parameter |
| mu_mh | Mean log(halo mass) at log(stellar mass)=11.3 and average size |
| beta_mh | Dependence of log(halo mass) on stellar mass |
| sigma_mh | Intrinsic scatter in log(halo mass) around the mean |
| mu_sigma | Mean log(sigma_ap) at log(stellar mass)=11.3 and average size |
| beta_sigma | Dependence of log(sigma_ap) on stellar mass |
| xi_sigma | Dependence of log(sigma_ap) on excess size | 
| nu_sigma | Dependence of log(sigma_ap) on excess halo mass | 
| sigma_sigma | Intrinsic scatter in log(sigma_ap) around the mean |
| t_find | Parameter `theta_0` of the lens finding probability |
| la_find | Log-10 of parameter `a` of the lens finding probability |
| psl_norm | Predicted number density of lenses (arbitrary units) |
| logp | log of posterior probability |

The file `slonly_inference.hdf5` is similarly structured.

### Posterior predicted samples

The file `pp_samples.hdf5` contains posterior predicted samples of properties of parent sample galaxies and strong lenses. It contains 10000 draws from the posterior. The data are organised in groups, as follows:

- `hyperpars`: values of the model parameters.
- `subset`: properties of a random sample of 100 galaxies (different seed for each sample).
- `lenses`: properties of a sample of 59 lenses.
- `lens_hyperplane`: coefficients of the fundamental hyper-plane, measured on the lenses
- `lens_sapobsplane`: coefficients of the fundamental hyper-plane, measured on the lenses using noisy velocity dispersion measurements

The `hyperpars` group contains datasets with the same name as the MCMC file described above.
For the `lenses` group, the following datasets are available. The shape of each dataset is (10000, 59).

| Name | Description |
| ---- | ----------- |
| tein | Einstein radius (arcsec) |
| tein_est | Estimated Einstein radius, a' la Bolton et al. (2006) (arcsec) |
| rein | Einstein radius (kpc) |
| ms | Log(sps stellar mass) |
| ms_obs | Observed Log(sps stellar mass) |
| lmstar | Log(true stellar mass) |
| re | Log(Reff/kpc) |
| zd | Lens redshift |
| zs | Source redshift |
| mh | Log(M200) |
| r200 | r200 (kpc) ||
| rs | Scale radius of the gNFW dark matter halo (kpc) |
| gamma | Inner slope of gNFW dark matter halo |
| sigma | Aperture velocity dispersion (km/s) |
| sigma_obs | Observed aperture velocity dispersion (km/s) |
| gammapl | Lensing-only power-law slope |
| lmst | Mass-sheet transformation parameter lambda_mst |

