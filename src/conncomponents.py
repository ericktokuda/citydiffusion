#!/usr/bin/env python3
"""Function related to the connected components of the image """

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys, PIL
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
from skimage import measure, filters, morphology

##########################################################
def get_connected_components(imgorig, outdir):
    """Short description     """
    info(inspect.stack()[0][3] + '()')

    img = imgorig[:, :, 0]
    blobs = measure.label(img, background=0)
    areathresh = 9
    blobs =  morphology.area_opening(blobs, area_threshold=areathresh)
    blobs =  morphology.area_closing(blobs, area_threshold=areathresh)

    cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(blobs, cmap=cmap)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'out.png'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img', required=True, help='Input image in BW')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    np.random.seed(0)
    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    img = np.array(PIL.Image.open(args.img))
    get_connected_components(img, args.outdir)
    info('For Aiur!')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
