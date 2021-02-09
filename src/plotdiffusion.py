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
import matplotlib as mpl#; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import pandas as pd
import h5py
from scipy import ndimage
from myutils import info, create_readme
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 360000000

RURAL = -2
FIGSCALE = 8
##########################################################
def hdf2numpy(hdfpath):
    """Convert hdf5 file to numpy """
    with h5py.File(hdfpath, 'r') as f:
        data = np.array(f['data'])
    return data

##########################################################
def plot_disttransform(mask0path, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'distransform.pdf')
    mask0 = hdf2numpy(mask0path)
    figsize = (FIGSCALE, int(FIGSCALE * (mask0.shape[0] / mask0.shape[1])))
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
    plt.close()

##########################################################
def list_hdffiles_and_stds(hdfdir):
    """Get list of standard deviations from filenames in the directory"""
    info(inspect.stack()[0][3] + '()')

    hdfpaths = []
    stds = []

    for f in sorted(os.listdir(hdfdir)):
        if not f.endswith('.hdf5') or 'steps' in f: continue
        hdfpaths.append(pjoin(hdfdir, f))
        stds.append(int(f.replace('.hdf5', '')))

    return hdfpaths, stds

##########################################################
def parse_urban_mask(maskpath, maskshape):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    import cv2
    from skimage.transform import resize
    if os.path.exists(maskpath):
        mask = np.asarray(Image.open(maskpath))[:, :, 0]
        mask = (mask > 128)
        mask = resize(mask, maskshape)
    else:
        info('NOT using any urban mask')
        mask = np.ones(maskshape, dtype=bool)

    borderpts, aux = cv2.findContours(mask.astype(np.uint8),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return borderpts, mask.astype(bool)

##########################################################
def get_min_time(minpixarg, hdfpaths):
    """Get minimum time for masks in @hdfpaths to achieve @minpixarg"""
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
        bounds = np.arange(np.min(steps), np.max(steps))
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
def plot_histograms_2d(hdfpaths, urbmaskpath, nbins, period, outdir):
    """Plot every @period histogram in t in 2d"""
    info(inspect.stack()[0][3] + '()')

    nonsource = ~hdf2numpy(hdfpaths[0]).astype(bool)
    borderpts, urbmask = parse_urban_mask(urbmaskpath, nonsource.shape)
    validids = np.logical_and(urbmask, nonsource)

    binw = 1 / nbins
    bins = np.arange(0, 1.00001, binw)
    hs = []

    fix, ax = plt.subplots()

    hdfpaths = hdfpaths
    for i, f in enumerate(hdfpaths):
        if i == 0: continue # skip first iteration
        if i % period != 0: continue # skip non-divisible by period

        mask = hdf2numpy(f)
        h, aux = np.histogram(mask[validids].flatten(), bins=bins)
        ax.plot(range(len(h)), h, label=f)

    # plt.legend()
    plt.savefig(pjoin(outdir, 'hist2d.png'))
    plt.close()

##########################################################
def plot_histograms_3d(hdfpaths, urbmaskpath, nbins, period, clipval,
                       maxsteps, outdir):
    """Plot every @period histogram in t in 3d"""
    info(inspect.stack()[0][3] + '()')

    nonsource = ~hdf2numpy(hdfpaths[0]).astype(bool)
    borderpts, urbmask = parse_urban_mask(urbmaskpath, nonsource.shape)
    validids = np.logical_and(urbmask, nonsource)

    binw = 1 / nbins
    bins = np.arange(0, 1 + .000001, binw)
    hs = []

    fix, ax = plt.subplots()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    n = len(hdfpaths) # due to the visualization issues

    for i, f in enumerate(hdfpaths):
        if i == 0 or i % period != 0: continue # skip first and non-divisiable steps

        mask = hdf2numpy(f)
        h, aux = np.histogram(mask[validids].flatten(), bins=bins)
        h = h / np.sum(h)
        mids = [(aux[j] + aux[j+1]) / 2 for j in range(nbins)]

        if clipval > 0: h[h > clipval] = clipval

        ax.plot(mids, [i] * nbins, h, label=f, zorder=(maxsteps - i))

    # One of the simulation stop criteria the pixels do not change
    # (within epsilon) anymore its values

    for j in range(i, maxsteps):
        if j % period != 0: continue
        ax.plot(mids, [j] * nbins, h, label=f, zorder=(maxsteps - j))

    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Time')
    ax.set_zlabel('Density')
    ax.set_zlim3d(0, 0.14)

    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'hist3d.png'))
    plt.close()

##########################################################
def plot_contour(stepsorig, minpixarg, stepsat, urbanmaskarg, figsize, outdir):
    """Plot values in @steps
    If @stepsat == -1, it finds the reachable values and ignore @stepsat.
    """
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'contour_{:.02f}.pdf'.format(minpixarg))

    steps = stepsorig.copy()
    # steps[np.where(steps == RURAL)] = 0

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.contour(steps)
    # cbar = fig.colorbar(im)
    # cbar.set_label('Time (steps)', labelpad=15, rotation=-90)
    plt.gca().invert_yaxis()
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

##########################################################
def print_mean_and_min(hdfpaths):
    """Get the mean of @mask0path and the min pixel value of @masklastpath """
    mean0 = np.mean(hdf2numpy(hdfpaths[0])) # Get the minpix of last iter
    minlast = np.min(hdf2numpy(hdfpaths[-1])) # Get the minpix of last iter
    info('mean0:{:.02f}, minpix:{:.02f}'.format(mean0, minlast))

##########################################################
def plot_lastiter_distrib(hdfpaths, outdir):
    mask = hdf2numpy(hdfpaths[-1])
    fig, ax = plt.subplots()
    ax.hist(mask.flatten(), density=True)
    plt.savefig(pjoin(outdir, 'lastdistrib.png'))
    plt.close()

##########################################################
def plot_signatures(hdfpaths, outdir):
    points = [[200, 50], [200, 150], [200, 200], [200, 250]]
    n = len(points)
    vals = [[], [], [], []]

    fig, ax = plt.subplots()

    for i, hdfpath in enumerate(hdfpaths):
        mask = hdf2numpy(hdfpath)
        for j, pt in enumerate(points):
            vals[j].append(mask[pt[0], pt[1]])

    xs = range(len(vals[0]))
    for j, val in enumerate(vals):
        ax.plot(xs, val, alpha=.7, label='pt{}'.format(j))

    ax.set_xlabel('Time')
    ax.set_ylabel('Pixel value')
    plt.legend()
    plt.savefig(pjoin(outdir, 'signature.png'))
    plt.close()


##########################################################
def get_step_distrib(stepspath):
    """Get distribution of steps """
    info(inspect.stack()[0][3] + '()')

    rural = invalid = 0

    with h5py.File(stepspath, 'r') as f:
        steps = np.array(f['data']).astype(int)

    vals, counts = np.unique(steps, return_counts=True)
    ruralidx = np.where(vals == RURAL)[0]
    invalididx = np.where(vals == -1)[0]

    N = np.sum(counts) # Whole image

    if len(ruralidx) > 0: rural = counts[ruralidx[0]]
    if len(invalididx) > 0: invalid = counts[invalididx[0]]

    firstvalid = np.where(vals == 0)[0][0] # Index of the first valid
    countsall = counts[firstvalid:]

    return rural, invalid, countsall, N
