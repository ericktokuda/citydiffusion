#!/usr/bin/env python3
"""Validate segmentation results
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
from myutils import info, create_readme, append_to_file
from PIL import Image

##########################################################
def parse_first_dim(imgpath):
    """Open and convert to boolean """
    # info(inspect.stack()[0][3] + '()')
    img = np.array(Image.open(imgpath))
    if img.ndim == 3: img = img[:, :, 0]
    return img.astype(bool).astype(int)

##########################################################
def overlay_images(gt, pred, outpath):
    h, w = gt.shape
    overlaid = np.ones((h, w, 4), dtype=np.uint8)
    overlaid[:, :, 2] = gt
    overlaid[:, :, 0] = pred
    overlaid[:, :, 1] = 0
    overlaid *= 255
    Image.fromarray(overlaid).save(outpath)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gtdir', required=True, help='GT directory')
    parser.add_argument('--preddir', required=True, help='Predicted directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    files = sorted(os.listdir(args.gtdir))

    res = 'file,precision,recall,f1\n'

    for f in files:
        if not f.endswith('.png'): continue
        info('f: {}'.format(f))

        gt = parse_first_dim(pjoin(args.gtdir, f))
        pred = parse_first_dim(pjoin(args.preddir, f))
        overlay_images(gt, pred, pjoin(args.outdir, f))
        diff = pred - gt
        vals, counts = np.unique(diff, return_counts=True)

        if -1 not in vals: fn = 0
        else: fn = counts[np.where(vals == -1)][0]

        if 1 not in vals: fp = 0
        else: fp = counts[np.where(vals == 1)][0]

        if 0 not in vals: tptn = 0
        else: tptn = counts[np.where(vals == 0)][0]

        totalpred = np.sum(pred)
        totalgt = np.sum(gt)
        tp = totalpred - fp
        tn = tptn - tp
        prec = tp / totalpred
        rec = tp / totalgt
        f1 = (2 * prec * rec) / (prec + rec)
        # info('prec:{:.02f}, recall:{:.02f}'.format(prec, rec))
        res += '{},{:.02f},{:.02f},{:.02f}\n'.format(f, prec, rec, f1)

    append_to_file(pjoin(args.outdir, 'results.csv'), res)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
