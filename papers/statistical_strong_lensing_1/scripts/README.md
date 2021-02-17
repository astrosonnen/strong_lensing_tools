# Contents

In order to reproduce the analysis of the paper, there are three steps to follow:

- step 1: Create the mock sample by running the script `make_mock.py`

- step 2: Calculate lensing grids for each lens in the sample (computing, for each value of the dark matter mass and slope, the stellar mass and source position necessary to reproduce the observed image positions). This requires running the script `get_lensmodel_grids.py`. It will take several hours to run and will take approximately 1GB of disk space.

- step 3: Run an MCMC to sample the posterior probability distribution of the model parameters given the data. The script `run_base_inference.py` fits for the base model, while the script `run_extended_inference.py` fits for the extended model. These scripts take a long time to run: about 12 hours when using 50 cores.

