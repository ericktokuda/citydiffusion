#!/usr/bin/env python3
"""Plot article figures
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
from myutils import info, create_readme
import scipy; from scipy.signal import convolve2d
import imageio
from myutils import plot

##########################################################
def get_kernel(kerdiam, kerstd, outdir):
    """Get 2d kernel """
    info(inspect.stack()[0][3] + '()')
    ker = (scipy.signal.gaussian(kerdiam, kerstd)).reshape(kerdiam, 1)
    ker2d = np.dot(ker, ker.T)
    ker2d = ker2d / np.sum(ker2d)
    imageio.imwrite(pjoin(outdir, 'kernel.png'), ker2d.astype(np.uint8))
    return ker2d

##########################################################
def diffuse_with_source(imorig, ker2d, refpoints, nsteps, label, palette, outdir):
    """Profile value of @refpoints with diffusion with source"""
    info(inspect.stack()[0][3] + '()')
    im = imorig.copy()
    profile = np.zeros((nsteps, len(refpoints)), dtype=float)
    colours = plot.hex2rgb(palette)
    r = 3

    for i in range(nsteps):
        outpath = pjoin(outdir, '{}_{:02d}.png'.format(label, i))
        # imageio.imwrite(outpath, im)
        colpath = pjoin(outdir, '{}_{:02d}_col.png'.format(label, i))
        colored = np.ones((imorig.shape[0], imorig.shape[1], 3), dtype=np.uint8)
        for j in range(3): colored[:, :, j] = im * 255
        for j, p in enumerate(refpoints):
            colored[p[0]-r:p[0]+r, p[1]-r:p[1]+r, :] = colours[j]
        imageio.imwrite(colpath, colored)

        im = convolve2d(im, ker2d, mode='same')
        im[np.where(imorig == 1)] = 1 # Source

        for j, p in enumerate(refpoints):
            x, y = p
            profile[i, j] = im[x, y]
    return profile

##########################################################
def plot_signatures(outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    n = 301
    nquart = int(n / 4)
    m = int(1.5 * nquart)
    nsteps = 30
    l = 4
    palette = plot.palettes['pastel']

    refpoints = (np.array([
                [1/4, 1/2],
                [3/8, 1/2],
                [1/2, 1/2],
                ]) * n).astype(int)

    ker2d =  get_kernel(kerdiam=int(n/5), kerstd=int(n/5), outdir=outdir)

    im = np.ones((n, n), dtype=np.uint8); im[nquart:-nquart, nquart:-nquart] = 0
    prof = diffuse_with_source(im, ker2d, refpoints, nsteps, 'A', palette, outdir)
    fig, axs = plt.subplots(1, 1, figsize=(l, l))
    for j in range(prof.shape[1]):
        axs.plot(range(nsteps), prof[:, j], c=palette[j], label='Point {}'.format(j))
        axs.set_xlabel('Time')
        axs.set_ylabel('Green index')
    plt.tight_layout()
    axs.legend(); plt.savefig(pjoin(outdir, 'A.pdf'))

    im = np.ones((n, n), dtype=np.uint8); im[nquart:-nquart, m:-m] = 0
    prof = diffuse_with_source(im, ker2d, refpoints, nsteps, 'B', palette, outdir)
    fig, axs = plt.subplots(1, 1, figsize=(l, l))
    for j in range(prof.shape[1]):
        axs.plot(range(nsteps), prof[:, j], c=palette[j], label='Point {}'.format(j))
        axs.set_xlabel('Time')
        axs.set_ylabel('Green index')
    plt.tight_layout()
    axs.legend(); plt.savefig(pjoin(outdir, 'B.pdf'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    readmepath = create_readme(sys.argv, args.outdir)

    plot_signatures(args.outdir)
    info('For Aiur!')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
