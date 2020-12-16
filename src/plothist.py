#!/usr/bin/env python3
"""Plot histograms
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
from scipy.optimize import curve_fit
from myutils import info, create_readme

##########################################################
def plot_fits(y, outpath):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    x = np.array(list(range(1, len(y) + 1)))

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='gray')

    def lognormal(x, mu, s):
        return  np.exp(-(np.log(x) - mu)**2 / (2 * s**2)) / (x * s * np.sqrt(2*np.pi))
    def fdistrib(x, d1, d2):
        return  np.sqrt( ((d1*x)**d1) * (d2**d2) / ((d1*x + d2) ** (d1 + d2))) / \
            (x * scipy.special.beta(d1 / 2, d2 / 2))
    def gammak3(x, t):
        return  ( x**(3-1) * np.exp(-x / t) ) / np.math.factorial(3 - 1)
    def gammak4(x, t):
        return  ( x**(4-1) * np.exp(-x / t) ) / np.math.factorial(4 - 1)
    def gammak5(x, t):
        return  ( x**(5-1) * np.exp(-x / t) ) / np.math.factorial(5 - 1)
    def gammak6(x, t):
        return  ( x**(6-1) * np.exp(-x / t) ) / np.math.factorial(6 - 1)

    funcs = {}
    funcs['lognormal'] = lognormal
    funcs['f'] = fdistrib
    funcs['gammak3'] = gammak3
    funcs['gammak4'] = gammak4
    funcs['gammak5'] = gammak5
    funcs['gammak6'] = gammak6

    for label, func in funcs.items():
        params, _ = curve_fit(func, x, y, maxfev=10000)
        print(params)
        ax.plot(x, func(x, *params), label=label, alpha=.6)
    ax.set_xlabel('Time (steps)')
    plt.legend(loc='upper right')
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_histograms(countsall, outdir):
    """Plot the histogram of all cities in @citiesdir and write to @outdir"""
    info(inspect.stack()[0][3] + '()')

    maxlen = 0
    for count in countsall.values():
        if len(count) > maxlen: maxlen = len(count)

    fig, ax = plt.subplots()
    x = range(1, maxlen + 1) # We treat separately step=0 (started green)
    for city, citycount in countsall.items():
        n = len(citycount)
        countrel = citycount.tolist() + [0]*(maxlen - n)
        ax.plot(x, countrel, label=city)

    ax.set_xlabel('Time (steps)')
    plt.legend(loc='upper right')
    plt.savefig(pjoin(outdir, 'hists.png'))
    plt.close()

##########################################################
def plot_fits_all(countsall, outdir):
    """Plot the histogram of all cities in @citiesdir and write to @outdir"""
    info(inspect.stack()[0][3] + '()')

    for city, vals in countsall.items():
        plot_fits(vals, pjoin(outdir, 'fits_{}.png').format(city))

##########################################################
def get_distribs(citiesdir, outdir):
    """Plot the histogram of all cities in @citiesdir and write to @outdir"""
    info(inspect.stack()[0][3] + '()')

    countsall = {}
    green0 = {}
    for d in os.listdir(citiesdir):
        # if len(d) > 2: continue
        citydir = pjoin(citiesdir, d)
        if not os.path.isdir(citydir): continue
        histpath = os.path.join(citydir, 'count_-1.00.txt')
        count = np.loadtxt(histpath)
        countsall[d] = count[1:] / np.sum(count[1:])
        green0[d] = count[0] / np.sum(count)

    return countsall, green0

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--citiesdir', required=True, help='Cities directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    countsall, green0 = get_distribs(args.citiesdir, args.outdir)
    plot_histograms(countsall, args.outdir)
    plot_fits_all(countsall, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()

