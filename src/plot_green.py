#!/usr/bin/env python3
"""Plot green ratios"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from datetime import datetime
import igraph
from myutils import info, create_readme
from matplotlib.collections import LineCollection


##########################################################
def limiarize_values(z, relticks):
    m = np.min(z); M = np.max(z)
    vals = -np.ones(z.shape, dtype=int)

    for i in range(len(relticks) - 1):
        a = (M - m) * relticks[i] + m
        b = (M - m) * relticks[i+1] + m
        vals[(z >= a) & (z < b)] = i
    return vals

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphml', required=True, help='Graph in graphml format')
    parser.add_argument('--accessib', required=True, help='Accessibility in txt format')
    parser.add_argument('--counts', required=True, help='Counts csv path')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    readmepath = create_readme(sys.argv, args.outdir)

    plt.style.use("dark_background")
    countsdf = pd.read_csv(args.counts)
    inds = (countsdf.count1 + countsdf.count2 > (512*512)*4)
    countsdf = countsdf[inds]

    cmap = 'Greens'
    g = igraph.Graph.Read(args.graphml)
    g.simplify()

    es = []
    for e in g.es:
        if (not inds[e.source]) or (not inds[e.target]): continue
        es.append([ [float(g.vs[e.source]['x']), float(g.vs[e.source]['y'])],
                [float(g.vs[e.target]['x']), float(g.vs[e.target]['y'])], ])

    # countsdf = countsdf[countsdf.ratio0 != -1]
    fig, ax = plt.subplots(figsize=(20, 20))
    count2 = countsdf.count2.values
    count2norm = count2 / (countsdf.count1 + count2)
    cmap = matplotlib.cm.get_cmap(cmap, 1000)

    # count2 = limiarize_values(count2, [0, .33, .66, 1.01])
    # cmap = matplotlib.cm.get_cmap(cmap, 3)
    
    sc = ax.scatter(countsdf.lon, countsdf.lat, c=count2norm, linewidths=0,
            alpha=.8, cmap=cmap)
    segs = LineCollection(es, colors='gray', linewidths=.2, alpha=.5)
    ax.add_collection(segs)
    plt.colorbar(sc)
    plt.savefig(pjoin(args.outdir, 'green.pdf'))

    # correlate with accessibility
    z = pd.read_csv(args.accessib, header=None)[0].values
    
    # z = limiarize_values(z, [0, .33, .66, 1.01])
    # z = limiarize_values(z, [0, .25, .5, .75, 1.01])


    import scipy; import scipy.stats
    corr, p = scipy.stats.pearsonr(count2norm, z[inds])
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(count2, z[inds], alpha=0.5)
    ax.set_title('Pearson: {:.02f}'.format(corr))
    ax.set_ylabel('Accessibility')
    ax.set_xlabel('Green')
    plt.savefig(pjoin(args.outdir, 'correlation.png'))
    info('Corr:{}'.format(corr))
    
    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
