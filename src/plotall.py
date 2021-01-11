#!/usr/bin/env python3
"""Plot all given a directory with subdirectories
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import h5py
from scipy.optimize import curve_fit
from myutils import info, create_readme
from plotdiffusion import *

##########################################################
RURAL = -2

##########################################################
def plot_fits(y, outpath):
    """Plot fits of the data"""
    info(inspect.stack()[0][3] + '()')

    x = np.array(list(range(1, len(y) + 1)))

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='gray')

    def lognormal(x, mu, s):
        return  np.exp(-(np.log(x) - mu)**2 / (2 * s**2)) / (x * s * np.sqrt(2*np.pi))
    def fdistrib(x, d1, d2):
        return  np.sqrt( ((d1*x)**d1) * (d2**d2) / ((d1*x + d2) ** (d1 + d2))) / \
            (x * scipy.special.beta(d1 / 2, d2 / 2))
    def gammak4(x, t):
        return  ( x**(4-1) * np.exp(-x / t) ) / np.math.factorial(4 - 1)
    def gammak5(x, t):
        return  ( x**(5-1) * np.exp(-x / t) ) / np.math.factorial(5 - 1)
    def gammak6(x, t):
        return  ( x**(6-1) * np.exp(-x / t) ) / np.math.factorial(6 - 1)
    def rayleigh(x, s):
        return  ( x / (s*s) ) * np.exp(- x*x / (2 * s*s))

    funcs = {}
    funcs['lognormal'] = lognormal
    funcs['f'] = fdistrib
    funcs['gammak4'] = gammak4
    funcs['gammak5'] = gammak5
    funcs['gammak6'] = gammak6
    funcs['rayleigh'] = rayleigh

    for label, func in funcs.items():
        try:
            params, _ = curve_fit(func, x, y, maxfev=10000)
            ax.plot(x, func(x, *params), label=label, alpha=.6)
            print(params)
        except Exception as e:
            info('Could not fit {}'.format(label))
            ax.plot(x, [0] * len(x), label=label, alpha=.6)
    ax.set_xlabel('Time (steps)')
    plt.legend(loc='upper right')
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_histograms(countsall, invalidall, outdir):
    """Plot the histogram of all cities in @citiesdir and write to @outdir"""
    info(inspect.stack()[0][3] + '()')

    maxlen = 0 # Get max len
    for relcount in countsall.values():
        if len(relcount) > maxlen: maxlen = len(relcount)

    fig, ax = plt.subplots()
    x = range(-1, maxlen)
    for city, c in countsall.items():
        N = np.sum(c) + invalidall[city]
        cityrel = (c / N ).tolist()
        n = len(cityrel)
        invalidrel = [invalidall[city] / N]
        counts = invalidrel + cityrel + [0]*(maxlen - n)
        ax.plot(x, counts, label=city)

    ax.set_xlabel('Time (steps)')
    plt.legend(loc='upper right')
    plt.savefig(pjoin(outdir, 'hists.png'))
    plt.close()

##########################################################
def parse_urban_mask(maskpath, maskshape):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    import cv2
    from skimage.transform import resize
    from PIL import Image
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
def plot_lastiter_distribs(citiesdir, urbmaskdir, outdir):
    fig, ax = plt.subplots()
    for city in os.listdir(citiesdir):
        laststeppath = pjoin(citiesdir, city, '100.hdf5')
        print(laststeppath)
        if not os.path.exists(laststeppath): continue
        with h5py.File(laststeppath, 'r') as f:
            mask = np.array(f['data'])
        urbpath = pjoin(urbmaskdir, city + '.png')
        urbanborder, urbanmask = parse_urban_mask(urbpath, mask.shape)
        k = mask[urbanmask].flatten()
        h = np.histogram(k, bins=100, density=True)[0]
        # h, counts = np.unique(mask[urbanmask], return_counts=True)
        ax.plot(h, label=city)
    plt.legend()
    plt.savefig(pjoin(outdir, 'lastdistrib.png'))
    plt.close()

##########################################################
def plot_fits_all(countsall, outdir):
    """Plot the histogram of all cities in @citiesdir and write to @outdir"""
    info(inspect.stack()[0][3] + '()')

    for city, v in countsall.items():
        vals = v[1:] # Exclude -1
        plot_fits(vals, pjoin(outdir, 'fits_{}.png').format(city))

##########################################################
def get_distribs(citiesdir, minpix, outdir):
    """Plot the histogram of all cities in @citiesdir and write to @outdir"""
    info(inspect.stack()[0][3] + '()')

    rural = {}
    invalid = {}
    countsall = {}

    for city in os.listdir(citiesdir):
        stepspath = pjoin(citiesdir, city, 'steps_{:.02f}.hdf5'.format(minpix))

        if not os.path.exists(stepspath): continue

        rural[city] = invalid[city] = 0

        with h5py.File(stepspath, 'r') as f:
            steps = np.array(f['data']).astype(int)

        vals, counts = np.unique(steps, return_counts=True)
        ruralidx = np.where(vals == RURAL)[0]
        invalididx = np.where(vals == -1)[0]

        N = np.sum(counts) # Whole image

        if len(ruralidx) > 0: rural[city] = counts[ruralidx[0]]
        if len(invalid) > 0: invalid[city] = counts[invalididx[0]]

        firstvalid = np.where(vals == 0)[0][0] # Index of the first valid
        countsall[city] = counts[firstvalid:]

    return rural, invalid, countsall, N

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--citiesdir', required=True, help='Cities directory')
    parser.add_argument('--urbmaskdir', default='', help='Urban mask directory')
    parser.add_argument('--minpix', required=True, type=float,
                        help='Minimum pixel, suffix of the count filename')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    stepsat = -1

    for city in sorted(os.listdir(args.citiesdir)):
        citydir = pjoin(args.citiesdir, city)
        if city.startswith('.') or not os.path.isdir(citydir): continue
        outdir = pjoin(args.outdir, city)
        os.makedirs(outdir, exist_ok=True)

        urbpath = pjoin(args.urbmaskdir, city + '.png')
        if os.path.exists(urbpath.replace('.png', '.jpg')):
            urbpath = urbpath.replace('.png', '.jpg')
        hdfpaths, stds = list_hdffiles_and_stds(citydir)
        print_mean_and_min(hdfpaths)

        plot_disttransform(hdfpaths[0], outdir)
        steps = get_min_time(args.minpix, hdfpaths)
        steps = fill_non_urban_area(steps, urbpath, RURAL)

        nbins = 100
        period = 2
        plot_histograms_2d(hdfpaths, urbpath, nbins, period, outdir)

        clipval = .15
        maxsteps = 100
        plot_histograms_3d(hdfpaths, urbpath, nbins, period, clipval,
                           maxsteps, outdir)

        stepspath = pjoin(outdir, 'steps_{:.02f}.hdf5'.format(args.minpix))
        with h5py.File(stepspath, "w") as f:
            f.create_dataset("data", data=steps, dtype='f')

        # plot_lastiter_distrib(hdfpaths, outdir)
        # plot_signatures(hdfpaths, outdir)

        figsize = (FIGSCALE, int(FIGSCALE * (steps.shape[0] / steps.shape[1])))

        plot_threshold(steps, args.minpix, stepsat, urbpath, figsize, outdir)
        plot_contour(steps, args.minpix, stepsat, urbpath, figsize, outdir)

    rural, invalid, counts, N = get_distribs(args.outdir, args.minpix, args.outdir)

    plot_histograms(counts, invalid, args.outdir)
    plot_fits_all(counts, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()

