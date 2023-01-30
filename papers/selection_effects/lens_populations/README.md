## Lens population simulations

### Simulation description 

We simulated three different populations of galaxies, with different values of the intrinsic scatter in the stellar and dark matter mass. We labeled these simulations as 'fiducial', 'low scatter' and 'high scatter'.
For each of these galaxy populations we simulated strong lensing events with a population of background extended sources.
In the case of the fiducial scatter simulation we also produced a simulation of galaxy-quasar lenses.

### Contents

`XXX_galaxies.hdf5`: file containing information on the properties of the galaxies (lenses and non-lenses). Below is a list of the datasets included in the .hdf5 file

| Dataset | Description                    |
| ------- | ------------------------------ |
| z       | Redshift                       |
| lmstar  | Log true stellar mass          |
| lmobs   | Log observed stellar mass      |
| lreff   | Log half-light radius (in kpc) |
| lm200   | Log halo mass                  |
| lasps   | Log stellar population synthesis mismatch parameter |
| gammadm | Inner slope of dark matter profile |
| lmdm5   | Log projected dark matter mass enclosed within 5kpc |
| q       | Axis ratio (minor-to-major) |
| r200    | Halo virial radius (in kpc) |
| rs      | Halo scale radius (in kpc) |
| tcaust  | Size of the radial caustic at a source redshift of 2.5 (arcsec) |
| tein    | Einstein radius at a source redshift of 2.5 (arcsec) |
| islens  | Lenses an extended source (bool) |
| qsolens | Lenses a quasar (bool, only for `fiducial` sim) |
| tein_zs | Einstein radius at the source redshift (arcsec, only for lenses) |
| tein_zqso | Einstein radius at the qso redshift (arcsec, only for quasar lenses) |



