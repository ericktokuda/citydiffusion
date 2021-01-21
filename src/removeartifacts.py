#!/usr/bin/env python3
"""Remove border lines
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

import numpy as np
import cv2

##########################################################
def get_kernels(k):
    """Get kernels"""
    # info(inspect.stack()[0][3] + '()')
    halfk =  int(k//2)
    kernels = {}

    kernels['l'] = np.zeros((k, k), dtype=np.uint8); kernels['l'][ :, 1:] = 1
    kernels['r'] = np.zeros((k, k), dtype=np.uint8); kernels['r'][ :,  :-1] = 1
    kernels['t'] = np.zeros((k, k), dtype=np.uint8); kernels['t'][1:,  :] = 1
    kernels['b'] = np.zeros((k, k), dtype=np.uint8); kernels['b'][ :-1,  :] = 1
    return kernels

##########################################################
def apply_kernels(img, kernels):
    """Apply kernels to the borders """
    # info(inspect.stack()[0][3] + '()')
    k = kernels['l'].shape[0]
    img[:, :k] = cv2.morphologyEx(img[:, :k], cv2.MORPH_CLOSE, kernels['l'])
    img[:, -k:] = cv2.morphologyEx(img[:, -k:], cv2.MORPH_CLOSE, kernels['r'])
    img[:k, :] = cv2.morphologyEx(img[:k, :], cv2.MORPH_CLOSE, kernels['t'])
    img[-k:, :] = cv2.morphologyEx(img[-k:, :], cv2.MORPH_CLOSE, kernels['b'])
    return img


##########################################################
def remove_border_artifact(imgpath, kernels, outpath):
    """Remove border artifact from @imgpath using the @kernels"""
    # info(inspect.stack()[0][3] + '()')
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = apply_kernels(img, kernels)
    cv2.imwrite(outpath, img)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zoomdir', required=True, help='Images directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    kernels = get_kernels(k=6)
    for x in sorted(os.listdir(args.zoomdir)):
        d = pjoin(args.zoomdir, x)
        outdir = pjoin(args.outdir, x)
        os.makedirs(outdir, exist_ok=True)
        if not os.path.isdir(d): continue
        info('dir:{}'.format(x))
        for y in sorted(os.listdir(d)):
            if not y.endswith('.png'): continue
            impath = pjoin(d, y)
            outpath = pjoin(outdir, y)
            remove_border_artifact(impath, kernels, outpath)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
