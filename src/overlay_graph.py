import os, shutil, glob, math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import requests        
from PIL import Image
import os
from os.path import join as pjoin


# CONSTANT
zoom =  18
size = 64 #image size
VAR = '/var/tmp/ekt248fg6h4jh8m90d8bn6d/'


def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

# Coord reference
x1,y1 = num2deg(96253, 146873, zoom)
x2,y2 = num2deg(96254, 146874, zoom)
lat_deg_per_pixel = (x2-x1) / size
lon_deg_per_pixel = (y2-y1) / size

def coord2pixel(lat, lon, size=512, zoom=18):
  xtile, ytile = deg2num(lat, lon,zoom)
  y1, x1 = num2deg(xtile, ytile, zoom)

  y_point = (abs(y1)- abs(lat)) / lat_deg_per_pixel
  x_point = (abs(x1)- abs(lon)) / lon_deg_per_pixel

  return round(x_point), round(y_point)

orig_img_path = pjoin(VAR, 'RP_IMGS/images/')
pred_img_path = pjoin(VAR, 'RP_IMGS/predicted/')

# Extraído de https://www.openstreetmap.org/

# Coordenadas de interesse [?]
coord = [[-21.1581, -47.8601],    # USP Ribeirão Preto
         [-21.20231, -47.79017],  # Arena EuroBike
         [-21.19928, -47.81632],  # Praça Nações Unidas
        ]

import igraph
g = igraph.Graph.Read(os.path.join(VAR, 'ribeirao.graphml'))
coord2 = np.array([[float(xx),float(yy) ] for xx, yy in zip(g.vs['x'], g.vs['y'])]) 
edges = np.array([ [e.source, e.target] for e in g.es])


nodeedges = {}
for i, e in enumerate(edges):
  for v in e:
    if v in nodeedges: nodeedges[v].add(i)
    else: nodeedges[v] = set([i])

tiles = {}
for i, c in enumerate(coord2):
  k = '{},{}'.format(*deg2num(c[1], c[0], zoom))
  if k not in tiles.keys(): tiles[k] = [i]
  else: tiles[k].append(i)

x = np.random.choice(list(tiles.keys()), size=2)
for k in x:
  xtile, ytile = [int(xx) for xx in k.split(',')]
  pixs = np.array([ coord2pixel(p[1], p[0]) for p in [coord2[p] for p in tiles[k]]])
  img_path = os.path.join(orig_img_path, str(xtile), str(ytile)+'.png')
  if not os.path.exists(img_path): continue
  plt.imshow(imread(img_path))
  plt.scatter(pixs[:, 0], pixs[:, 1], c='r')
  plt.axis('off')
  plt.savefig(pjoin(VAR, '{}.png'.format(k)))
  plt.show()

np.random.seed(0)
# x = np.random.choice(list(tiles.keys()), size=100)
# x = x[0]
# x0, y0 = [int(xx) for xx in x.split(',')]

xs = [int(xx) for xx in sorted(os.listdir(orig_img_path))]
x0 = np.min(xs); xsmax = np.max(xs)
ys = [int(xx.replace('.png', '')) for xx in sorted(os.listdir('{}/{}'.format(orig_img_path, x0)))]
y0 = np.min(ys); ysmax = np.max(ys)

nrows = xsmax - x0
ncols = ysmax - y0
imgsize = 10000

# x0 = xs[101]
# y0 = ys[100]
# x0 = xs[np.random.randint(len(xs)-10)]
# y0 = ys[np.random.randint(len(ys)-10)]
# nrows = 2
# ncols = 2
# imgsize = 1000

import cv2

img = np.zeros((nrows*size, ncols*size, 3), dtype=np.uint8)
fig, ax = plt.subplots(figsize=(.01*imgsize*ncols/nrows, imgsize*.01), dpi=100)
# fig, ax = plt.subplots(figsize=(figsize*.5, figsize))
for i in range(ncols):
  xtile = x0 + i
  for j in range(nrows):
    ytile = y0 + j
    k = '{},{}'.format(xtile, ytile)
    img_path = os.path.join(orig_img_path, str(xtile), str(ytile)+'.png')
    # print(img_path, os.path.exists(img_path))
    if not os.path.exists(img_path): continue
    z = imread(img_path)
    z = cv2.resize(z, (size, size))
    img[j*size:(j+1)*size, i*size:(i+1)*size, :] = z
newcoord = {}
nodeids = []
for i in range(ncols):
  xtile = x0 + i
  print(i, xtile)
  for j in range(nrows):
    ytile = y0 + j
    k = '{},{}'.format(xtile, ytile)
    if k in tiles:
      nodeids.extend(tiles[k])
      pixs = np.array([ coord2pixel(p[1], p[0], size=size) for p in [coord2[p] for p in tiles[k]]])
      
      pixs[:, 0] += i*size
      pixs[:, 1] += j*size
      for jj, p in enumerate(tiles[k]):
        newcoord[p] = pixs[jj, :] 
      ax.scatter(pixs[:, 0], pixs[:, 1], c='r', s=20, alpha=.5)

edgeids = []
for n in nodeids:
  for e in nodeedges[n]:
    if edges[e][0] in nodeids and edges[e][1] in nodeids: 
      edgeids.append(e)
  
from matplotlib.collections import LineCollection
segs = []
for e in edgeids:
  v1, v2 = edges[e]
  segs.append([ newcoord[v1], newcoord[v2] ])
segs = np.array(segs).astype(int)
# line_segments = LineCollection(segs, colors='k', linewidths=5)
line_segments = LineCollection(segs, colors='b', linewidths=.7, alpha=.5)
ax.add_collection(line_segments)
ax.imshow(img)
plt.axis('off')
plt.savefig(pjoin(VAR, 'all_{}_{}.png'.format(imgsize, size)))
