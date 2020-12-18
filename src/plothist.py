#!/usr/bin/env python3
"""Plot histograms of different simulations. It must be run after
simudiffusion.py and plotdiffusion.py. It expect a directory structure in which each folder contains a "count_MINPIX.txt" file
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
        params, _ = curve_fit(func, x, y, maxfev=10000)
        print(params)
        ax.plot(x, func(x, *params), label=label, alpha=.6)
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
    parser.add_argument('--minpix', required=True, type=float,
                        help='Minimum pixel, suffix of the count filename')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    rural, invalid, counts, N = get_distribs(args.citiesdir, args.minpix, args.outdir)

    plot_histograms(counts, invalid, args.outdir)
    plot_fits_all(counts, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()

