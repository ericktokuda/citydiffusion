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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import pandas as pd
import h5py
from scipy import ndimage
from myutils import info, create_readme
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 360000000

##########################################################
def hdf2numpy(hdfpath):
    """Convert hdf5 file to numpy """
    with h5py.File(hdfpath, 'r') as f:
        data = np.array(f['data'])
    return data

##########################################################
def plot_disttransform(figsize, mask0, outpath):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    distransf = ndimage.distance_transform_edt(mask0.astype(int))
    im = ax.imshow(distransf)
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Distance', labelpad=15, rotation=-90)
    plt.tight_layout()
    plt.savefig(outpath)

##########################################################
def list_hdffiles_and_stds(hdfdir):
    """Get list of standard deviations from filenames in the directory"""
    info(inspect.stack()[0][3] + '()')

    hdfpaths = []
    stds = []

    for f in sorted(os.listdir(hdfdir)):
        if not f.endswith('.hdf5'): continue
        hdfpaths.append(pjoin(hdfdir, f))
        stds.append(int(f.replace('.hdf5', '')))

    return hdfpaths, stds

##########################################################
def parse_urban_mask(maskpath, maskshape):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    import cv2
    from skimage.transform import resize
    if maskpath:
        mask = np.asarray(Image.open(maskpath))[:, :, 0]
        mask = (mask > 128)
        mask = resize(mask, maskshape)
    else:
        mask = np.ones(maskshape, dtype=bool)


    borderpts, aux = cv2.findContours(mask.astype(np.uint8),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return borderpts, mask

##########################################################
def plot_contour(maskpath):
    """Plot the contour"""
    info(inspect.stack()[0][3] + '()')

    import cv2
    im = cv2.imread(maskpath)

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,
            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

##########################################################
def plot_threshold(minpixarg, hdfdir, mask0, urbanmaskarg, figsize, outdir):
    """Plot the required time of the pixels of a map to achieve a minimum value"""
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'diffusion_{:03d}.pdf'. \
            format(int(minpixarg*100)))

    stepsat = 18 # satutation step .6:18, .75:25
    hdfpaths, stds = list_hdffiles_and_stds(hdfdir)

    if minpixarg < 0:
        # By setting this value, I guarantee *all* pixels will achieve the
        # minimum desired values

        largeststdpath = pjoin(hdfdir, '{:02d}.hdf5'.format(sorted(stds)[-1]))
        minpix = np.min(hdf2numpy(largeststdpath))

        # print(outpath, meanpix); return # For finding the min over all cities
    else:
        minpix = minpixarg

    info('minpix:{}'.format(minpix))

    urbanborder, urbanmask = parse_urban_mask(urbanmaskarg, mask0.shape)

    steps = - np.ones(mask0.shape, dtype=int) # Num steps to achieve minpix

    info('Traversing backwards...')
    nstds = len(stds)

    for i, std in enumerate(sorted(stds, reverse=True)): # Traverse backwards
        k = nstds - i - 1
        info('diff step:{}'.format(std))
        mask = hdf2numpy(pjoin(hdfdir, '{:02d}.hdf5'.format(std)))
        inds = np.where(mask >= minpix)
        steps[inds] = k # Keep the minimum time to achieve minpix

    # from matplotlib import cm
    # from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    # mycmap = cm.get_cmap('jet', 12)

    info('Saturating pixels by {}'.format(stepsat))
    steps[steps > stepsat] = stepsat

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    steps[~urbanmask.astype(bool)] = 0 # Crop urban area
    im = ax.imshow(steps, vmin=0, vmax=stepsat,
            # cmap=mycmap,
            # norm=matplotlib.colors.LogNorm(vmin=steps.min(), vmax=steps.max()),
            )

    ax.set_axis_off()
    # ax.set_title('Initial mean:{:.02f}, threshold:{:.02f}'. \
            # format(np.mean(mask0), minpix))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cmap = plt.cm.viridis  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

    bounds = np.arange(0, stepsat + .1, 1)
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cbar = plt.colorbar(im, cax=cax,
                            spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    # cbar = plt.colorbar(im, cax=cax)

    for aux in urbanborder: ax.plot(aux[:, 0, 0], aux[:, 0, 1], c='gray')

    cbar.set_label('Time (steps)', labelpad=15, rotation=-90)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    distr = np.zeros(stepsat + 1, dtype=int)

    # Store the steps
    vals, counts = np.unique(steps[urbanmask.astype(bool)], return_counts=True)
    info(vals, counts)
    for v, c in zip(vals, counts): distr[int(v)] = c
    np.savetxt(pjoin(outdir, 'hist.csv'), distr, fmt='%d', delimiter=',')

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.hist(range(stepsat + 1), weights=distr/np.sum(distr))
    ax.set_xlabel('Time (steps)')
    ax.set_ylim(0, .5)
    plt.savefig(pjoin(outdir, 'hist.pdf'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdfdir', default='/tmp/out/', help='Hdf5 directory')
    parser.add_argument('--urbanmask', help='Hdf5 directory')
    parser.add_argument('--minpix', type=float, default=-1, help='Min pixel value')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    W = 600; H = 800; figsize=(W*.01, H*.01)

    mask0 = hdf2numpy(pjoin(args.hdfdir, '00.hdf5'))

    distpath = pjoin(args.outdir, 'distransform.pdf')
    plot_disttransform(figsize, mask0, distpath)

    plot_threshold(args.minpix, args.hdfdir, mask0, args.urbanmask,
            figsize, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
