#!/usr/bin/env python3
"""Keep intact the region within the mask provided and fill the outside with the desired value

for C in FS LO RP SJ SO UB; do  python src/fillnonurban.py --map ~/results/citysegm/20201217-6cities_filtered_concat32/${C}.png --urbanmask ~/results/citysegm/20201114-masks/${C}/binaria_${C}.jpg; done
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
import cv2
from skimage.transform import resize
import PIL; from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 360000000
from myutils import info, create_readme

##########################################################
def parse_urban_mask(maskpath, maskshape):
    """Parse @maskpath in grey level, with three colour channels"""
    info(inspect.stack()[0][3] + '()')

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
def fill_non_urban_area(map_, urbanmaskarg, fillvalue=0):
    """The area not covered by @urbanmaskarg is filled with @fillvalue"""
    info(inspect.stack()[0][3] + '()')

    urbanborder, urbanmask = parse_urban_mask(urbanmaskarg, map_.shape)

    if np.all(urbanmask): info('No urban area provided!')
    map_[~urbanmask] = fillvalue
    return map_

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, help='Map path')
    parser.add_argument('--urbanmask', required=True, help='Urban mask in grey level')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    map_ = np.array(Image.open(args.map))[:, :, 0]
    map_[np.where(map_ < 128)] = 0; map_[np.where(map_ >= 128)] = 255
    map_ = fill_non_urban_area(map_, args.urbanmask, fillvalue=0)
    result = Image.fromarray(map_.astype(np.uint8))
    filename = os.path.basename(args.map)
    result.save(pjoin(args.outdir, filename))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
