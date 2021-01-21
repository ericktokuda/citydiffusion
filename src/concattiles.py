#!/usr/bin/env python3
"""Assemble tiles into a grid """

import argparse
import time
import os, shutil, glob, math
from os.path import join as pjoin
import inspect
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
# from skimage.io import imread, imsave
import cv2


##########################################################
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

##########################################################
def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

##########################################################
def get_tile_indices(imdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    xs = []
    for x in sorted(os.listdir(imdir)):
        if os.path.isdir(pjoin(imdir, x)): xs.append(int(x))

    ys = []
    for y in sorted(os.listdir(pjoin(imdir, str(xs[0])))):
        if y.endswith('.png'): ys.append(int(y.replace('.png', '')))

    return xs, ys

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imdir', required=True, help='Output directory')
    parser.add_argument('--zoom', default=18, type=int, help='Zoom of the images')
    parser.add_argument('--tilesize', default=32, type=int, help='Tile tgt sizes')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    t = args.tilesize

    xs, ys = get_tile_indices(args.imdir)

    x0 = np.min(xs); xsmax = np.max(xs)
    y0 = np.min(ys); ysmax = np.max(ys)

    ncols = xsmax - x0 + 1; nrows = ysmax - y0 + 1
    H = t * nrows; W = t * ncols
    info('We are gonna generate an image {}x{}'.format(W, H))

    img_dummy = cv2.imread(os.path.join(args.imdir, str(x0), str(y0)+'.png'))
    if len(img_dummy.shape) == 3: coloured = True
    else: coloured = False

    if coloured: img = np.zeros((nrows*t, ncols*t, 3), dtype=np.uint8)
    else: img = np.zeros((nrows*t, ncols*t), dtype=np.uint8)

    for i in range(ncols):
        xtile = x0 + i
        info('xtile:{} ({}/{})'.format(xtile, i, ncols))
        for j in range(nrows):
            ytile = y0 + j
            k = '{},{}'.format(xtile, ytile)
            img_path = os.path.join(args.imdir, str(xtile), str(ytile)+'.png')
            if not os.path.exists(img_path): continue
            z = cv2.imread(img_path)
            z = cv2.resize(z, (t, t))
            if coloured: img[j*t:(j+1)*t, i*t:(i+1)*t, :] = z
            else: img[j*t:(j+1)*t, i*t:(i+1)*t] = z

    cv2.imwrite(pjoin(args.outdir, 'pred_{}.png'.format(args.tilesize)), img)
    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()


