#!/usr/bin/env python3
"""Dilate images up to the area of a reference image and randomly remove the elements of the last step """

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import skimage
from skimage.morphology import dilation
from PIL import Image

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refimage', required=True, help='Reference image')
    parser.add_argument('--tgtimage', required=True, help='Image to dilate')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    structel = np.ones((3,3))
    tgtarea = len(np.where(np.asarray(Image.open(args.refimage)) != 0)[0])
    info('trgetarea:{}'.format(tgtarea))
    tgtimg = np.asarray(Image.open(args.tgtimage))
    maxval = np.max(tgtimg)
    info('initiarea:{}'.format(int(np.sum(tgtimg / maxval))))

    sz = tgtimg.shape
    diag = np.sqrt(sz[0]**2 + sz[1]**2) # Max number of dilations

    diff = np.ones(tgtimg.shape)
    for i in range(int(diag)):
        info('i:{}'.format(i))
        curarea = np.sum(tgtimg / maxval)
        if curarea >= tgtarea: break
        newimg = skimage.morphology.dilation(tgtimg, selem=structel)
        diff = newimg - tgtimg
        tgtimg = newimg

    assert np.all(newimg >= 0)
    dt = int(curarea - tgtarea)
    m = int(np.sum(diff != 0))
    indsdel = np.random.choice(m, size=dt, replace=False)
    xs = np.where(diff)[0][indsdel]
    ys = np.where(diff)[1][indsdel]
    tgtimg[xs, ys] = 0
    suff, ext = os.path.splitext(os.path.basename(args.tgtimage))
    outpath = pjoin(args.outdir, '{}_dilated{}'.format(suff, ext))
    Image.fromarray(tgtimg).save(outpath)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
