#!/usr/bin/env python3
"""Function related to the connected components of the image """

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys, PIL
import pickle as pkl
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme, append_to_file
from skimage import measure, filters, morphology

##########################################################
def get_connected_components(maskorig, areathresh, outdir):
    """Get the connected components considering an 8-connectivity"""
    info(inspect.stack()[0][3] + '()')

    outpath = pjoin(outdir, 'components.pkl')
    if os.path.exists(outpath):
        info('Loading cached components file')
        return pkl.load(open(outpath, 'rb'))

    if maskorig.ndim > 2: mask = maskorig[:, :, 0]
    else: mask = maskorig

    comps, n = measure.label(mask, background=0, return_num=True, connectivity=2)
    comps =  morphology.area_opening(comps, area_threshold=areathresh)
    comps =  morphology.area_closing(comps, area_threshold=areathresh)
    m = len(np.unique(comps))
    pkl.dump(comps, open(outpath, 'wb'))
    info('Number of components:{}, {}'.format(n, m))
    return comps

##########################################################
def plot_distribution(comps, outdir):
    """Plot the histogram, removing the background"""
    info(inspect.stack()[0][3] + '()')
    vals, counts = np.unique(comps, return_counts=True)
    counts = counts[1:] # remove background
    inds = np.where(counts > 0)
    plt.hist(counts[inds], bins=30)
    plt.xlabel('Area of the component'); plt.ylabel('Number of components')
    plt.savefig(pjoin(outdir, 'hist_zoom0.png'))
    plt.close()

    inds = np.where( (counts > 0) & (counts < 5000))
    plt.hist(counts[inds], bins=30)
    plt.xlabel('Area of the component'); plt.ylabel('Number of components')
    plt.savefig(pjoin(outdir, 'hist_zoom1.png'))
    plt.close()

    inds = np.where( (counts > 0) & (counts < 100))
    plt.hist(counts[inds], bins=30)
    plt.xlabel('Area of the component'); plt.ylabel('Number of components')
    plt.savefig(pjoin(outdir, 'hist_zoom2.png'))
    plt.close()

##########################################################
def plot_connected_comps(mask, comps, minarea1, minarea2, outdir):
    """Plot the connected components """
    info(inspect.stack()[0][3] + '()')
    aux = np.random.rand(256,3)
    aux[0, :] = [0, 0, 0] # black background
    cmap = matplotlib.colors.ListedColormap(aux)

    nrows = 1;  ncols = 4; figscale = 16
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))

    axs[0, 0].imshow(mask, cmap='gray')
    axs[0, 1].imshow(comps, cmap=cmap)

    filtered = filter_by_area(comps, minarea1, keep='gt')
    axs[0, 2].imshow(filtered, cmap=cmap)
    filtered[np.where(filtered > 0)] = 255
    im = PIL.Image.fromarray(filtered.astype(np.uint8))
    im.save(pjoin(outdir, 'filtered_{}.png'.format(minarea1)))
    info('np.sum(filtered):{}'.format(np.sum(filtered)))

    filtered = filter_by_area(comps, minarea2, keep='gt')
    axs[0, 3].imshow(filtered, cmap=cmap)
    filtered[np.where(filtered > 0)] = 255
    im = PIL.Image.fromarray(filtered.astype(np.uint8))
    im.save(pjoin(outdir, 'filtered_{}.png'.format(minarea2)))
    info('np.sum(filtered):{}'.format(np.sum(filtered)))

    filtered = filter_by_area(comps, minarea2, keep='gt')
    axs[0, 3].imshow(filtered, cmap=cmap)
    filtered[np.where(filtered > 0)] = 255
    im = PIL.Image.fromarray(filtered.astype(np.uint8))
    im.save(pjoin(outdir, 'filtered_{}.png'.format(minarea2)))
    info('np.sum(filtered):{}'.format(np.sum(filtered)))

    filtered = filter_by_area(comps, minarea1, keep='lt')
    axs[0, 3].imshow(filtered, cmap=cmap)
    filtered[np.where(filtered > 0)] = 255
    im = PIL.Image.fromarray(filtered.astype(np.uint8))
    im.save(pjoin(outdir, 'filtered_{}_lt.png'.format(minarea1)))
    info('np.sum(filtered):{}'.format(np.sum(filtered)))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'plot.png'))

##########################################################
def get_quantiles(comps, delta, outdir):
    """Get percentiles"""
    ticks = np.arange(0, 1.00001, delta)
    vals, counts = np.unique(comps, return_counts=True)
    quantiles = np.quantile(counts[1:], ticks)
    info('vals:{}, quantiles:{}'.format(vals, quantiles))
    txt = '{}\n{}'.format(ticks, quantiles)
    append_to_file(pjoin(outdir, 'percentiles.csv'), txt)
    return quantiles

##########################################################
def filter_by_area(compsorig, refarea, keep='gt'):
    """Filter by area=@refarea"""
    info(inspect.stack()[0][3] + '()')
    comps = compsorig.copy()
    vals, areas = np.unique(comps, return_counts=True)

    if keep == 'gt':
        delinds = np.where(areas < refarea)
    elif keep == 'lt':
        delinds = np.where(areas > refarea)

    delmask = np.isin(comps, vals[delinds])
    comps[delmask] = 0
    return comps

#########################################################
# def keep_maxarea(compsorig, maxarea):
    # """Filter by area=@minarea"""
    # info(inspect.stack()[0][3] + '()')
    # comps = compsorig.copy()
    # vals, counts = np.unique(comps, return_counts=True)
    # inds = np.where(counts < minarea)
    # mask = np.isin(comps, vals[inds])
    # comps[mask] = 0
    # return comps
##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', required=True, help='Input image in BW')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    np.random.seed(0)
    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    mask = np.array(PIL.Image.open(args.mask))
    comps = get_connected_components(mask, 4, args.outdir)
    plot_distribution(comps, args.outdir)
    ZOOMSCALE = 512 / 32 # Scale used in the diffusion
    minarea1 = (40 * 40) * 5 / ZOOMSCALE
    minarea2 = (200 * 200) * 2 / ZOOMSCALE
    plot_connected_comps(mask, comps, minarea1, minarea2, args.outdir)
    # quantiles = get_quantiles(comps, .25, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
