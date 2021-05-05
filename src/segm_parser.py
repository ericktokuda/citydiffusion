#!/usr/bin/env python3
"""Tools for enriching the segmentation data into the graph"""

import argparse, time, os, math
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme, graph, geo, plot
import igraph
import imageio
import pandas as pd
from PIL import Image
import rasterio.features
import shapely.geometry
import pickle

##########################################################
def deg2num(lon_deg, lat_deg, zoom, imgsize=256):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (int(xtile), int(ytile),
            int(math.modf(xtile)[0]*imgsize), int(math.modf(ytile)[0]*imgsize))

##########################################################
def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

##########################################################
def get_enclosing_grid(x, y, zoom, imgsize, mskdir, m=3):
    """Get neibhouring tiles"""
    # info(inspect.stack()[0][3] + '()')
    pad = m//2
    x0tile, y0tile, xpix, ypix = deg2num(x, y, zoom, imgsize) # get current tile

    c = [xpix + pad*imgsize, ypix + pad*imgsize]
    grid = np.zeros((imgsize*m, imgsize*m), dtype=int)

    for i in range(-pad, +pad+1):
        x0 = (i + pad) * imgsize
        for j in range(-pad, +pad+1):
            y0 = (j + pad) * imgsize
            impath = pjoin(mskdir, str(x0tile+i), '{}.png'.format(y0tile+j))
            try:
                grid[y0:y0+imgsize, x0:x0+imgsize,] = imageio.imread(impath)
            except:
                info('Error loading ' + impath)
                grid[y0:y0+imgsize, x0:x0+imgsize,] = -1
                continue
    return grid, c

########################################################## DEFINES
def bisection_root(f, a, b, tickstol=0.001, valtol=0.001):
    if f(a) * f(b) > 0:
        info('Root is not inside the bounds [{}, {}]'.format(a, b))
        return -1
    
    i = 0
    while b - a >  tickstol:
        i += 1
        c = (b + a) / 2
        if np.abs(f(c)) < valtol: return c

        if f(a) * f(c) < 0: b = c
        else: a = c
    return c

##########################################################
def dist_to_deltalon(dist, lonref, latref):
    """Get the variation in longitude based on a reference lat and lon.
    For the same distance, lon varies more than lat."""
    def get_delta_lon_from_d(lon1, lat1=latref, lon2=lonref, lat2=latref):
        return dist - geo.haversine(lon1, lat1, lon2, lat2)

    lon2 = lonref + 20 # a large enough diff in lon to contain the @dist
    lon2 = bisection_root(get_delta_lon_from_d, lonref, lon2, 0.00001, 0.00001)
    return np.abs(lon2 - lonref)

##########################################################
def get_counts(img, pos, ballmask, labelvals):
    """Get counts of values in @img centered on x, y"""
    # info(inspect.stack()[0][3] + '()')
    counts = np.zeros(len(labelvals))
    x0, y0 = pos
    rx = ballmask.shape[0] // 2 # assuming ballmask has size odd
    ry = ballmask.shape[1] // 2
    roi = img[x0-rx:x0+rx+1, y0-ry:y0+ry+1]
    
    pixels, count = np.unique(roi[ballmask].flatten(), return_counts=True)

    for i, v in enumerate(pixels): # Count just the relevant data
        if v not in labelvals: continue
        j = labelvals.index(v)
        counts[j] = count[i]

    return counts

##########################################################
def get_circle_mask(radiuspix):
    """Gets a mask of size 2*@radiuspix-1. The center pixel is included in the
    radius."""
    n =  radiuspix*2 - 1
    x0 = radiuspix - 1
    y0 = radiuspix - 1
    radius2 = radiuspix ** 2

    circlemask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if (i - x0)**2 + (j - y0)**2 < radius2:
                circlemask[i, j] = True

    return circlemask

##########################################################
def get_deltapix_from_dist(realdist, lonref, latref, zoom, imgsize):
    """Gets a geodesic distance ball"""
    deltalon = dist_to_deltalon(realdist, lonref, latref)
    xtile1, ytile1, xpix1, ypix1 = deg2num(lonref, latref, zoom, imgsize)
    xtile2, ytile2, xpix2, ypix2 = deg2num(lonref+deltalon, latref, zoom, imgsize)
    return (xtile2-xtile1) * imgsize + (xpix2 - xpix1)

##########################################################
def get_geodesic_mask(realdist, lonref, latref, zoom, imgsize):
    """Gets a geodesic distance ball"""
    deltalon = dist_to_deltalon(realdist, lonref, latref)
    xtile1, ytile1, xpix1, ypix1 = deg2num(lonref, latref, zoom, imgsize)
    xtile2, ytile2, xpix2, ypix2 = deg2num(lonref+deltalon, latref, zoom, imgsize)
    dx = (xtile2-xtile1) * imgsize + (xpix2 - xpix1)
    n =  dx*2 - 1
    mask = np.zeros((n, n), dtype=int)
    pad_rel = (n // 2) / imgsize
    lat0, lon0 = num2deg(xtile1 + pad_rel, ytile1 + pad_rel, zoom) # Middle

    for i in range(n):
        xtile1rel = xtile1 + i / imgsize
        for j in range(n):
            lat1, lon1 = num2deg(xtile1rel, ytile1 + j / imgsize, zoom)
            d = geo.haversine(lon0, lat0, lon1, lat1)
            if d < realdist: mask[i, j] = True
    return mask

    b1 = interp(a1, a0, b0, a2, b2)
##########################################################
def interp(a1, a0, b0, a2, b2):
    """Linearly interpolate a1 to obtain b1 considering the simmetries
    a0,b0 and a2,b2"""
    # info(inspect.stack()[0][3] + '()')
    return (a1 - a0) / (a2 - a0) * (b2 - b0) + b0

##########################################################
def trim_graph(graphmlpath, maskpath, rect, outdir):
    """Crop the graph (@graphmlpath) considering the @mask, with extremities
    given by @rec (xmin, ymin, xmax, ymax) and outputs to @outdir"""
    info(inspect.stack()[0][3] + '()')
    g = graph.simplify_graphml(graphmlpath) # g = igraph.Graph.Read(graphmlpath)
    mask = np.array(Image.open(maskpath))
    if mask.ndim == 3:
        mask = 0.299*mask[:, :, 0] + 0.587*mask[:, :, 1] + 0.114*mask[:, :, 2]
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)

    shapes = rasterio.features.shapes(mask)
    polygons = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]
    ul = [rect[0], rect[1]]
    lr = [rect[2] + 1, rect[3] + 1] # We want the lower right corner
    latmax, lonmin = num2deg(ul[0], ul[1], 18)
    latmin, lonmax = num2deg(lr[0], lr[1], 18)

    info('g.vcount():{}'.format(g.vcount()))
    majpolygon = polygons[np.argmax([p.area for p in polygons])]

    xs, ys = np.array(g.vs['x']), np.array(g.vs['y'])

    info('Identifying points inside the mask...')

    pixcoords = np.ndarray((g.vcount(), 2), dtype=float)
    inside = np.zeros(g.vcount(), dtype=bool)
    for i in range(g.vcount()):
        y1 = interp(ys[i], latmin, 0, latmax, mask.shape[0])
        x1 = interp(xs[i], lonmin, 0, lonmax, mask.shape[1])
        pixcoords[i, :] = x1, y1
        p = shapely.geometry.Point([x1, y1])
        inside[i] = majpolygon.contains(p)

    info('Plotting...')
    plt.figure(figsize=(16, 16))
    plt.imshow(mask)
    plt.scatter(pixcoords[:, 0][inside], pixcoords[:, 1][inside], c='k')
    plt.savefig(pjoin(outdir, 'masked.png'))

    g2 = g.induced_subgraph(np.where(inside)[0])
    g3 = g2.components(mode='weak').giant()

    figscale = .003
    _, ax = plt.subplots(figsize=(mask.shape[1]*figscale, mask.shape[0]*figscale))
    plot.plot_graph(g3, pjoin(outdir, 'map.png'), ax=ax)
    pickle.dump(g3, open(pjoin(outdir, 'masked.pkl'), 'wb'))
    return g3

##########################################################
def trim_xnet(xnetpath, maskpath, rect, outdir):
    """Crop the graph (@graphmlpath) considering the @mask, with extremities
    given by @rec (xmin, ymin, xmax, ymax) and outputs to @outdir"""
    info(inspect.stack()[0][3] + '()')
    from myutils import xnet
    g = xnet.xnet2igraph(xnetpath)

    mask = np.array(Image.open(maskpath))
    if mask.ndim == 3:
        mask = 0.299*mask[:, :, 0] + 0.587*mask[:, :, 1] + 0.114*mask[:, :, 2]
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)

    shapes = rasterio.features.shapes(mask)
    polygons = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]
    ul = [rect[0], rect[1]]
    lr = [rect[2] + 1, rect[3] + 1] # We want the lower right corner
    latmax, lonmin = num2deg(ul[0], ul[1], 18)
    latmin, lonmax = num2deg(lr[0], lr[1], 18)

    info('g.vcount():{}'.format(g.vcount()))
    majpolygon = polygons[np.argmax([p.area for p in polygons])]

    xs, ys = np.array(g.vs['posx']), np.array(g.vs['posy'])

    info('Identifying points inside the mask...')

    pixcoords = np.ndarray((g.vcount(), 2), dtype=float)
    inside = np.zeros(g.vcount(), dtype=bool)
    for i in range(g.vcount()):
        y1 = interp(ys[i], latmin, 0, latmax, mask.shape[0])
        x1 = interp(xs[i], lonmin, 0, lonmax, mask.shape[1])
        pixcoords[i, :] = x1, y1
        p = shapely.geometry.Point([x1, y1])
        inside[i] = majpolygon.contains(p)

    info('Plotting...')
    plt.figure(figsize=(16, 16))
    plt.imshow(mask)
    plt.scatter(pixcoords[:, 0][inside], pixcoords[:, 1][inside], c='k')
    plt.savefig(pjoin(outdir, 'masked.png'))

    g2 = g.induced_subgraph(np.where(inside)[0])
    g3 = g2.components(mode='weak').giant()

    figscale = .003
    _, ax = plt.subplots(figsize=(mask.shape[1]*figscale, mask.shape[0]*figscale))
    plot.plot_graph(g3, pjoin(outdir, 'map.png'), ax=ax)
    pickle.dump(g3, open(pjoin(outdir, 'masked.pkl'), 'wb'))
    xnet.igraph2xnet(g3, pjoin(outdir, 'graph.xnet'))
    return g3

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mskdir', required=True, help='Binary masks dir')
    parser.add_argument('--ballradius', default=.01, type=float,
            help='Ball radius (in meters)')
    parser.add_argument('--graphml', required=True, help='Path to the graph')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output dir')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    countspath = pjoin(args.outdir, 'counts.csv')
    readmepath = create_readme(sys.argv, args.outdir)
    labels = {'invalid': -1, 'non-green': 0, 'green': 65535}
    nlabels = len(labels)
    zoom = 18
    imgsize = 512

    g = graph.simplify_graphml(args.graphml, directed=False)
    info('nvertices:{}, nedges:{}'.format(g.vcount(), g.ecount()))
    coords = np.array([ [float(v['x']), float(v['y'])] for v in g.vs ])

    radiuspix = get_deltapix_from_dist(args.ballradius, np.mean(coords[:, 0]),
            np.mean(coords[:, 1]), zoom, imgsize)
    roimask = get_circle_mask(radiuspix) # For short distances, it is a good approx
    info('roimask.shape:{}'.format(roimask.shape))

    # roimask = get_geodesic_mask(args.ballradius, np.mean(coords[:, 0]),
            # np.mean(coords[:, 1]), zoom, imgsize)

    m = 2 * math.ceil(roimask.shape[0] / imgsize) + 1
    counts = -np.ones((g.vcount(), nlabels), dtype=int)

    for i, v in enumerate(coords): #loop in graph nodes (x,y)
        info('i:{}'.format(i))
        grid, pos = get_enclosing_grid(v[0], v[1], zoom, imgsize, args.mskdir, m)
        counts[i, :] = get_counts(grid, pos, roimask, list(labels.values()))

    info('Exporting counts to {}'.format(countspath))
    df = pd.DataFrame({'lon': coords[:, 0], 'lat': coords[:, 1]})
    for i in range(counts.shape[1]):
        df['count{}'.format(i)] = counts[:, i]

    df.to_csv(countspath, index=True, index_label='id')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
