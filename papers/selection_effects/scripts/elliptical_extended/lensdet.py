import numpy as np
from skimage import measure


def detect_lens(img, sky_rms, nsigma_pixdet=2., npix_min=10):

    sb_min = nsigma_pixdet * sky_rms

    ny, nx = img.shape
    x0 = nx/2. - 0.5
    y0 = ny/2. - 0.5

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # counts the number of images with the minimum threshold (2 sigma)
    footprint = img > sb_min

    img_detected = img.copy()

    labels = measure.label(footprint)
    nreg = labels.max()
    euler = measure.euler_number(footprint)
    nimg_std = 0
    nholes_std = nreg - euler
    for n in range(nreg):
        npix_tmp = (labels==n+1).sum()
        signal = img[labels==n+1].sum()
        noise = npix_tmp**0.5 * sky_rms
        img_sn = signal/noise
        if img_sn >= 10. and npix_tmp >= npix_min:
            nimg_std += 1
        else:
            img_detected[labels==n+1] = 0.

    std_footprint = img_detected > sb_min

    sorted_img = img.flatten().copy()
    sorted_img.sort()
    sb_det = sorted_img[sorted_img > sb_min]
    nup = len(sb_det)

    i = 0
    nimg_max = nimg_std
    nimg_tmp = nimg_std
    sb_maxlim = sb_min

    nholes_max = 0

    best_footprint = std_footprint.copy()

    # if there's at least one detected image, increases the threshold to see
    # if more can be detected

    while nimg_std > 0 and nimg_tmp >= nimg_max and i < nup:
        sb_lim = sb_det[i]

        img_detected = img.copy()
        footprint_tmp = img > sb_lim

        labels = measure.label(footprint_tmp)
        nreg = labels.max()

        nimg_tmp = 0
        nholes_tmp = 0
        for n in range(nreg):
            npix_tmp = (labels==n+1).sum()
            signal = img[labels==n+1].sum()
            noise = npix_tmp**0.5 * sky_rms
            img_sn = signal/noise
            if img_sn >= 10. and npix_tmp >= npix_min:
                nimg_tmp += 1
                # checks if the image has a hole
                euler = measure.euler_number(labels==n+1)
                nholes_tmp += 1 - euler
            else:
                img_detected[labels==n+1] = 0.

        if nimg_tmp > nimg_max:
            nimg_max = nimg_tmp
            sb_maxlim = sb_lim
            best_footprint = img_detected > sb_lim

        if nholes_tmp > nholes_max:
            nholes_max = nholes_tmp

        i += 1

    if nimg_max > 1:
        islens = True
    else:
        islens = False

    return islens, nimg_std, nimg_max, nholes_std, nholes_max, std_footprint, best_footprint, sb_maxlim

