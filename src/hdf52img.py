#!/usr/bin/env python3
"""Convert hdf5 to png
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
import h5py

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdfdir', help='Hdf5 dir')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    files = sorted(os.listdir(args.hdfdir))
    for f in files:
        if not f.endswith('.hdf5'): continue
        h5path = pjoin(args.hdfdir, f)
        with h5py.File(h5path, "r") as fh:
            data = np.array(fh.get('data'))
        info('f:{}, shape:{}'.format(f, data.shape))
        plt.imshow(data, cmap='gray')
        plt.savefig(pjoin(args.outdir, f.replace('.hdf5', '.png')))

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
