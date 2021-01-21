#!/usr/bin/env python3
"""one-line docstring
"""

import os
from os.path import join as pjoin
import numpy as np
import imageio
from pathlib import Path
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

from pytest import approx
from src.histmatch import *

def test_histmatch():
    outdir = '/tmp/out/'
    tpldir = pjoin(outdir, 'tpl')
    srcdir = pjoin(outdir, 'src')
    matchedir = pjoin(outdir, 'matched')

    os.makedirs(tpldir, exist_ok=True)
    os.makedirs(tpldir, exist_ok=True)
    os.makedirs(matchedir, exist_ok=True)

    tpl = data.coffee()
    src = data.chelsea()

    
    for kstr, k in zip(['tpl', 'src'], [tpl, src]):
        i = 0
        sz = k.shape[0]
        indir = pjoin(outdir, kstr, 'A{}'.format(i))
        os.makedirs(indir, exist_ok=True)
        imageio.imwrite(pjoin(indir, '{}.png'.format(i)),
                k[:sz // 2, :, :].astype(np.uint8))

        i = 1
        indir = pjoin(outdir, kstr, 'A{}'.format(i))
        os.makedirs(indir, exist_ok=True)
        imageio.imwrite(pjoin(indir, '{}.png'.format(i)),
                k[sz // 2:, :, :].astype(np.uint8))


    srcpaths = list(Path(srcdir).rglob('*.png'))
    tplpaths = list(Path(tpldir).rglob('*.png'))
    match_hist(srcpaths, tplpaths, matchedir)

    t = imageio.imread(pjoin(matchedir, 'A0/B0/0.png'))
    b = imageio.imread(pjoin(matchedir, 'A1/B1/1.png'))

    k1 = np.concatenate([t, b])
    k2 = match_histograms(src, tpl).astype(np.uint8)
    assert approx(k1) == k2
