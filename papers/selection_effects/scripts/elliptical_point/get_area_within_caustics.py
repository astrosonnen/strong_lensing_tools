import numpy as np
import pylab
import h5py
from scipy.optimize import brentq


nq = 6
q_arr = np.linspace(0.5, 1., nq)
e_arr = 1. - q_arr

cs_arr = np.zeros(nq)
cs_quad = np.zeros(nq)

nmocks = 100000

output_file = h5py.File('caustic_areas.hdf5', 'w')

output_file.create_dataset('e_grid', data=e_arr)

for i in range(nq):

    nmulti = 0
    nquads = 0

    print(i)

    f = open('composite_e%2.1f_mock.dat'%e_arr[i], 'r')
    lines = f.readlines()[1:]
    f.close()
    
    nimg = 0

    ndet_list = []

    sline = True
    for line in lines:
        if sline:
            nimg = int(line[0])
            nhere = 0
            if nimg > 0:
                sline = False
                if nimg > 1:
                    nmulti += 1
                    if nimg > 3:
                        nquads += 1
            else:
                sline = True
        elif nimg > 1:
            nhere += 1
            if nhere == nimg:
                sline = True
        else:
            sline = True

    cs_arr[i] = nmulti * 4. / nmocks
    cs_quad[i] = nquads * 4. / nmocks

    print(i, nmulti, nquads)

output_file.create_dataset('full_cs', data=cs_arr)
output_file.create_dataset('quad_cs', data=cs_quad)

