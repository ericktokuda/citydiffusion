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
def get_connected_components(maskorig, areathresh, outdir):
    """Get the connected components considering an 8-connectivity"""
    info(inspect.stack()[0][3] + '()')

    if maskorig.ndim > 2: mask = maskorig[:, :, 0]
    else: mask = maskorig

    comps, n = measure.label(mask, background=0, return_num=True, connectivity=2)
    comps =  morphology.area_opening(comps, area_threshold=areathresh)
    comps =  morphology.area_closing(comps, area_threshold=areathresh)
    m = len(np.unique(comps))
    info('Number of components:{}, {}'.format(n, m))
    return comps

##########################################################
def plot_distribution(comps, outdir):
    """Plot the histogram, removing the background"""
    info(inspect.stack()[0][3] + '()')
    vals, counts = np.unique(comps, return_counts=True)
    counts = counts[1:] # remove background
    inds = np.where(counts > 0)
    plt.hist(counts[inds], bins=30)
    plt.xlabel('Area of the component'); plt.ylabel('Number of components')
    plt.savefig(pjoin(outdir, 'hist_zoom0.png'))
    plt.close()

    inds = np.where( (counts > 0) & (counts < 25000))
    plt.hist(counts[inds], bins=30)
    plt.xlabel('Area of the component'); plt.ylabel('Number of components')
    plt.savefig(pjoin(outdir, 'hist_zoom1.png'))
    plt.close()

    inds = np.where( (counts > 0) & (counts < 5000))
    plt.hist(counts[inds], bins=30)
    plt.xlabel('Area of the component'); plt.ylabel('Number of components')
    plt.savefig(pjoin(outdir, 'hist_zoom2.png'))
    plt.close()

##########################################################
def plot_connected_comps(mask, comps, outdir):
    """Plot the connected components """
    info(inspect.stack()[0][3] + '()')

    cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))

    nrows = 1;  ncols = 2; figscale = 2
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))

    axs[0, 0].imshow(mask, cmap='gray')
    axs[0, 1].imshow(comps, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'plot.png'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', required=True, help='Input image in BW')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    np.random.seed(0)
    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    areathresh = 9

    mask = np.array(PIL.Image.open(args.mask))
    comps = get_connected_components(mask, areathresh, args.outdir)
    plot_distribution(comps, args.outdir)
    plot_connected_comps(mask, comps, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
