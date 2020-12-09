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
import matplotlib as mpl; mpl.use('Agg')
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

RURAL = -2
##########################################################
def hdf2numpy(hdfpath):
    """Convert hdf5 file to numpy """
    with h5py.File(hdfpath, 'r') as f:
        data = np.array(f['data'])
    return data

##########################################################
def plot_disttransform(figsize, mask0path, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'distransform.pdf')
    mask0 = hdf2numpy(mask0path)
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

    return borderpts, mask.astype(bool)

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
def get_min_time(minpixarg, hdfpaths):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    masklast = hdf2numpy(hdfpaths[-1])
    minpix = np.min(masklast) if minpixarg < 0 else minpixarg

    steps = - np.ones(masklast.shape, dtype=int) # Num steps to achieve minpix

    nstds = len(hdfpaths)

    for i in range(nstds - 1, -1, -1): # Traverse backwards
        mask = hdf2numpy(hdfpaths[i])
        steps[np.where(mask >= minpix)] = i # Keep the minimum time to achieve minpix

    return steps

##########################################################
def fill_non_urban_area(steps, urbanmaskarg, fillvalue=0):
    """The area not covered by @urbanmaskarg is filled with @fillvalue"""
    info(inspect.stack()[0][3] + '()')
    
    urbanborder, urbanmask = parse_urban_mask(urbanmaskarg, steps.shape)
    
    if np.all(urbanmask): info('No urban area provided!')
    steps[~urbanmask] = fillvalue
    return steps

##########################################################
def plot_threshold(stepsorig, minpixarg, stepsat, urbanmaskarg, figsize, outdir):
    """Plot values in @steps
    If @stepsat == -1, it finds the reachable values and ignore @stepsat.
    """
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'heatmap_{:.02f}.pdf'.format(minpixarg))

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    steps = stepsorig.copy()
    steps[np.where(steps == RURAL)] = 0

    if stepsat > 0:
        info('Saturating pixels by {}'.format(stepsat))
        im = ax.imshow(steps, vmin=0, vmax=stepsat)
        bounds = np.arange(0, stepsat + 1, 1)
    else:
        im = ax.imshow(steps)
        bounds = np.arange(0, np.max(steps) + 1, 1)
                # cmap=mycmap,
                # norm=matplotlib.colors.LogNorm(vmin=steps.min(), vmax=steps.max()),

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cmap = plt.cm.viridis  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('x', cmaplist, cmap.N)
    
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar = plt.colorbar(im, cax=cax, spacing='proportional',
            ticks=bounds, boundaries=bounds, format='%1i')

    urbanborder, _ = parse_urban_mask(urbanmaskarg, steps.shape)
    for aux in urbanborder: ax.plot(aux[:, 0, 0], aux[:, 0, 1], c='gray')

    cbar.set_label('Time (steps)', labelpad=15, rotation=-90)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_histogram(steps, minpixarg, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    # Store the steps
    vals, counts = np.unique(steps, return_counts=True)
    
    distr = np.zeros(np.max(vals) + 1, dtype=int)
    for v, c in zip(vals, counts):
        if v < 0: continue # non urban areas
        distr[int(v)] = c

    countpath = pjoin(outdir, 'count_{:.02f}.txt'.format(minpixarg))
    np.savetxt(countpath, distr, fmt='%d')

##########################################################
def print_mean_and_min(hdfpaths):
    """Get the mean of @mask0path and the min pixel value of @masklastpath """
    mean0 = np.mean(hdf2numpy(hdfpaths[0])) # Get the minpix of last iter
    minlast = np.min(hdf2numpy(hdfpaths[-1])) # Get the minpix of last iter
    info('mean0:{:.02f}, minpix:{:.02f}'.format(mean0, minlast))

##########################################################
def store_steps(steps, minpix, outdir):
    """Store @steps as hdf5"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'steps_{:.02f}.hdf5'.format(minpix))
    with h5py.File(outpath, "w") as f:
        dset = f.create_dataset("data", data=steps, dtype='f')

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

    if args.minpix == -1: stepsat = -1
    else: stepsat = 18 # Adjust here, if args.urbanmask is set

    hdfpaths, stds = list_hdffiles_and_stds(args.hdfdir)
    print_mean_and_min(hdfpaths)
    plot_disttransform(figsize, hdfpaths[0], args.outdir)
    steps = get_min_time(args.minpix, hdfpaths)
    steps = fill_non_urban_area(steps, args.urbanmask, RURAL)
    store_steps(steps, args.minpix, args.outdir)
    plot_threshold(steps, args.minpix, stepsat, args.urbanmask, figsize, args.outdir)
    plot_histogram(steps, args.minpix, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
