import numpy as np
from sl_profiles import sersic
from lensdet import detect_lens
import h5py


ntein = 11
ltein_grid = np.linspace(-1., 0., ntein)

nsim = 10000

nser = 1.

sky_rms = 1.
sb_min = 2. * sky_rms

cs_grid = np.zeros(ntein)

islens_grid = np.zeros((ntein, nsim), dtype=bool)
x_grid = np.zeros((ntein, nsim))
y_grid = np.zeros((ntein, nsim))

for m in range(ntein):

    print(m)
    ltein = ltein_grid[m]

    nlens = 0
    islens_sim = np.zeros(nsim, dtype=bool)
    nimages_sim = np.zeros(nsim, dtype=int)

    img_file = h5py.File('mockdir/ltein%2.1f_images.hdf5'%ltein, 'r')

    rmax = img_file.attrs['rmax']
    source_area = np.pi*rmax**2

    x_grid[m, :] = img_file['x'][()]
    y_grid[m, :] = img_file['y'][()]

    for i in range(nsim):

        img = img_file['lens_%04d_wseeing'%i][()]

        res = detect_lens(img, sky_rms, npix_min=1)
        islens_sim[i] = res[0]
        islens_grid[m, i] = res[0]

    cs_grid[m] = islens_sim.sum()/float(nsim) * source_area

output = h5py.File('largesource_crosssect.hdf5', 'w')

output.create_dataset('ltein_grid', data=ltein_grid)
output.create_dataset('cs_grid', data=cs_grid)
output.create_dataset('islens_grid', data=islens_grid)
output.create_dataset('x_grid', data=x_grid)
output.create_dataset('y_grid', data=y_grid)


