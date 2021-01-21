#!/usr/bin/env python3
"""The main function if match_hist, which matches the histogram of srcdir
to the histogram of tpldir"""

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
import imageio
from pathlib import Path

##########################################################
def count_unique_across_files(imgpaths):
    """Count ocurrences of 0-255 values"""
    info(inspect.stack()[0][3] + '()')

    vals_unique = np.arange(256)
    counts_unique = np.zeros(256)
    
    for p in imgpaths:
        im = imageio.imread(p)
        vals, counts = np.unique(im, return_counts=True)
        counts_unique[vals] += counts

    return counts_unique

##########################################################
def get_histmatch_func(srcpaths, tplpaths, outdir='/tmp'):
    src_counts = count_unique_across_files(srcpaths)
    src_cdf = np.cumsum(src_counts) / np.sum(src_counts)

    bufpath = pjoin(outdir, 'tpl_counts.txt')
    if os.path.exists(bufpath):
        info('Loading existing template counts {}'.format(bufpath))
        tpl_counts = np.genfromtxt(bufpath)
    else:
        tpl_counts = count_unique_across_files(tplpaths)
        np.savetxt(bufpath, tpl_counts)

    tpl_cdf = np.cumsum(tpl_counts) / np.sum(tpl_counts)

    return np.interp(src_cdf, tpl_cdf, np.arange(256))

##########################################################
def match_hist(srcpaths, tplpaths, outdir):
    """Matches images in histogram of the concatenated corresponding to @srcpaths
    to the histogram of concatenated images from @tplpaths. It recreates the
    parent folder of every file in @srcpaths in @outdir"""
    info(inspect.stack()[0][3] + '()')

    info('Num source images: {}'.format(len(srcpaths)))
    info('Num template images: {}'.format(len(tplpaths)))

    f = get_histmatch_func(srcpaths, tplpaths, outdir)

    for p in srcpaths:
        d, filename = os.path.split(p)
        d, d1 = os.path.split(d)
        odir = pjoin(outdir, d1)
        os.makedirs(odir, exist_ok=True)
        img = imageio.imread(p)
        imgout = f[img.ravel()].reshape(img.shape)
        imageio.imwrite(pjoin(odir, filename), imgout.astype(np.uint8))
        
##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--srcdir', required=True, help='Source images root dir')
    parser.add_argument('--tpldir', required=True, help='Template images root dir')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    readmepath = create_readme(sys.argv, args.outdir)
    np.random.seed(0)

    srcpaths = list(Path(args.srcdir).rglob('*.png'))
    tplpaths = list(Path(args.tpldir).rglob('*.png'))

    match_hist(srcpaths, tplpaths, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
