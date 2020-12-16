#!/usr/bin/env python3
"""Convolve circle wth mask
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import PIL
from PIL import Image
from scipy.spatial import distance
from scipy.signal import correlate2d, convolve2d
from multiprocessing import Pool
from scipy.stats import moment
import scipy
import torch
from torch.nn import functional as F
from scipy import signal, stats
import h5py
import pandas as pd
from myutils import info, create_readme, append_to_file

CUDA = torch.cuda.is_available()
#CUDA=False

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def get_circle_coords(r, c):
    """Get coordinates of points inside circle of radius @r and center @c"""
    ndim = len(c)
    mincoords = (c - r).astype(int)
    maxcoords = (c + r).astype(int)
    ranges = [list(range(mincoords[i], maxcoords[i]+1)) for i in range(ndim)]
    from itertools import product
    coords = np.array(list(product(*ranges)))
    inside = distance.cdist([c], coords) <= r
    return coords[inside[0]]

##########################################################
def get_circular_kernel(r, ndims):
    """Get circular kernel with radius @r """
    c = np.ones(ndims)*r
    d = 2*r+1
    ker = np.zeros([d]*ndims)
    inside = get_circle_coords(r, c)
    ker[tuple(inside.T)] = 1
    return ker

##########################################################
def colours_to_labels_from_first_coord(map_):
    """Transform colours to labels"""
    info(inspect.stack()[0][3] + '()')
    M = np.max(map_)
    if len(map_.shape) == 2:
        labels = map_ == M
    else:
        labels = map_[:, :, 0] == M
    return labels.astype(int)

##########################################################
def conv_gpu(ker, map_):
    """Correlate in gpu """
    # info(inspect.stack()[0][3] + '()')
    imgSize = map_.shape[0]

    filt = torch.from_numpy(ker).type(torch.float32)
    filt = filt.unsqueeze(0).unsqueeze(0)
    img = torch.from_numpy(map_).type(torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)

    if CUDA:
        filt = filt.cuda()
        img = img.cuda()

    # m = torch.nn.Upsample(scale_factor=2, mode='nearest')
    # img = m(img)
    # m = torch.nn.AvgPool2d((2, 2), stride=(2, 2))
    # imgout = m(imgout)
    # model = torch.nn.Sequential(
            # torch.nn.Upsample(scale_factor=2, mode='nearest'),
            # torch.nn.ReflectionPad2d(1),
            # torch.nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
            # F.conv2d(filt, bias=None, padding=0, stride=(1, 1))
            # )
    # imgout = model(img)

    imgout = F.conv2d(img, filt, bias=None, padding=ker.shape[0]//2, stride=(1, 1))
    return imgout.cpu().numpy()[0][0]

##########################################################
def store_data(zold, i, kerrad, outdir):
    """Store intermediate or just the final data"""
    outpath = pjoin(outdir, '{:02d}.png'.format(i))
    r = kerrad
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(zold[r:-r,r:-r], cmap='gray');
    plt.savefig(outpath)
    plt.close()

    outpath = pjoin(outdir, '{:02d}.hdf5'.format(i))
    with h5py.File(outpath, "w") as f:
        dset = f.create_dataset("data", data=zold[r:-r,r:-r], dtype='f')

#############################################################
def run_experiment(labels, diam, std, eps, maxiter, outfmt, outdir):
    kerrad = int(diam/2)

    h, w = labels.shape

    ker = (scipy.signal.gaussian(diam, std)).reshape(diam, 1)
    ker2d = np.dot(ker, ker.T) # separability of the kernel
    ker2d = ker2d / np.sum(ker2d) # normalization
    ker = (scipy.signal.gaussian(diam, std)).reshape(diam, 1)
    ker2d = np.dot(ker, ker.T) # separability of the kernel
    ker2d = ker2d / np.sum(ker2d) # normalization
    plt.imshow(ker2d)
    plt.savefig(pjoin(outdir, 'kernel.png'))
    plt.close()

    zold = labels.copy()
    for i in range(maxiter+1):
        info('i:{}'.format(i))
        store_data(zold, i, kerrad, outdir)

        if CUDA: z =  conv_gpu(ker2d, zold)
        else: z = convolve2d(zold, ker2d, mode='same')
        z[np.where(labels == 1)] = 1
        if i > 0 and np.linalg.norm(z - zold) < eps: break
        else: zold = z

    return i

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', default='data/toy02.png', help='Mask path')
    parser.add_argument('--kersize', default=15, type=int, help='Kernel size (odd integer value)')
    parser.add_argument('--kerstd', default=5, type=int, help='Standard deviation (int)')
    parser.add_argument('--samplesz', default=-1, type=int, help='sample size')
    parser.add_argument('--outfmt', default='both', help='Output format (png,hdf5,both)')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    if (args.kersize %2) != 1: info('Please provide an ODD diameter'); return
    if CUDA: info('Using cuda:{}')
    else: info('NOT using cuda:{}')

    PIL.Image.MAX_IMAGE_PIXELS = 360000000
    map_ = np.asarray(Image.open(args.mask))
    labels0 = colours_to_labels_from_first_coord(map_)

    h, w = labels0.shape
    samplesz = args.samplesz
    eps = 10
    maxiter = 100

    info('labels.shape:{}'.format(labels0.shape))
    info('samplesz:{}'.format(samplesz))
    info('kerstd:{}'.format(args.kerstd))

    if samplesz > 0 and samplesz < h and samplesz < w:
        margin = [(h - samplesz) // 2, (w - samplesz) // 2]
        dx = 0; dy = 0 # dy = 500
        labels0  = labels0[margin[0]+dx:margin[0]+dx+samplesz,
                margin[1]+dy:margin[1]+dy+samplesz]

    pad = args.kersize // 2
    labels = np.ones((labels0.shape[0]+2*pad, labels0.shape[1]+2*pad), dtype=int)
    labels[pad:-pad, pad:-pad] = labels0

    outpath = pjoin(args.outdir, 'README.csv')
    del labels0

    lastiter = run_experiment(labels, args.kersize, args.kerstd, eps,
                              maxiter, args.outfmt, args.outdir)
    info('lastiter:{}'.format(lastiter))
    append_to_file(readmepath, 'lastiter:{}'.format(lastiter))

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
