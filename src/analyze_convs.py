#!/usr/bin/env python3
"""Analyze convolutions
"""

import argparse
import time
import os, sys
from os.path import join as pjoin
import inspect

import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import h5py
from scipy import ndimage
from myutils import info, create_readme

##########################################################
def hdf2numpy(hdfpath):
    """Convert hdf5 file to numpy """
    with h5py.File(hdfpath, 'r') as f:
        data = np.array(f['data'])
    return data

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdfdir', default='/tmp/out/', help='Hdf5 directory')
    parser.add_argument('--minpix', type=float, default=-1, help='Min pixel value')
    parser.add_argument('--kersize', type=float, required=True, help='Kernel size that was used in the convolution')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    files = sorted(os.listdir(args.hdfdir))
    stds = []
    for f in files:
        if not f.endswith('.hdf5'): continue
        stds.append(int(f.replace('.hdf5', '')))

    kersize = 501

    mask0 = hdf2numpy(pjoin(args.hdfdir, '00.hdf5'))
    h, w = mask0.shape
    # margin = kersize
    # cropsize = [h - 2*kersize, w - 2*kersize]
    # mask0 = mask0[margin:-margin, margin:-margin]

    figsize = (8, 8)
    plt.figure(figsize=figsize)
    plt.imshow(mask0.astype(int), cmap='gray') # store cropped mask
    plt.savefig(pjoin(args.outdir, 'cropped.png')); plt.close()

    plt.figure(figsize=figsize)
    distransf = ndimage.distance_transform_edt(mask0.astype(int))
    plt.imshow(distransf) # distance transform
    plt.colorbar()
    plt.savefig(pjoin(args.outdir, 'distransform.png')); plt.close()

    if args.minpix < 0:
        minpix = np.min(hdf2numpy(pjoin(args.hdfdir, '{:02d}.hdf5'.format(sorted(stds)[-1]))))
    else:
        minpix = args.minpix

    info('minpix:{}'.format(minpix))

    # steps = -10*np.ones((cropsize[0], cropsize[1]))
    steps = -10*np.ones((h, w), dtype=float)
    
    nstds = len(stds)
    for i, std in enumerate(sorted(stds, reverse=True)):
        k = nstds - i
        info('std:{}'.format(std))
        mask = hdf2numpy(pjoin(args.hdfdir, '{:02d}.hdf5'.format(std)))
        # mask = mask[margin:-margin, margin:-margin]
        inds = np.where(mask >= minpix)
 
        steps[inds] = k # we'are gonna keep the lowest k among iterations

    # from matplotlib import cm
    # from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    # mycmap = cm.get_cmap('jet', 12)
    plt.figure(figsize=figsize)
    plt.imshow(steps,
            # cmap=mycmap,
            # norm=matplotlib.colors.LogNorm(vmin=steps.min(), vmax=steps.max()),
            )
    plt.title('Initial mean:{:.02f}, threshold:{:.02f}'.format(np.mean(mask0),
        minpix))
    plt.colorbar()
    plt.savefig(pjoin(args.outdir, 'diffusion_{:03d}'.format(int(minpix*100))))

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
