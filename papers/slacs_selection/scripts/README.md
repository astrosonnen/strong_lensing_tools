# Scripts

### Contents

`make_slacs_lensing_grids.py`: on a grid of values of the total density slope (gamma), it computes, for each SLACS lens:

- The value of the mass enclosed within 5kpc (m5) that matches the Einstein radius
- The Jacobian of the variable change from m5 to the Einstein radius
- The lensing cross-section for the fiducial reference source
- The lensing cross-section for a reference source with intrinsic emission line flux equal to half of the detection limit.

`make_slacs_jeans_grids.py`: on a grid of values of the total density slope (gamma), it integrates the spherical Jeans equation to obtain the model-predicted velocity dispersion within the SDSS spectroscopic aperture. It requires [this](https://github.com/astrosonnen/spherical_jeans) spherical Jeans code to run.

