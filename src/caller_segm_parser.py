from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from segm_parser import trim_graph
import pandas as pd
import os


graphmldir = '/home/frodo/results/citydiffusion/20210415-mask2graph/graphml/'
maskdir = '/home/frodo/results/citydiffusion/20210415-mask2graph/mask/'
rect = [102674,140018,102736,140103]
bounds = pd.read_csv('/home/frodo/results/citydiffusion/20210415-mask2graph/bounds.csv', index_col='city')
outroot = '/tmp/out/'

cities = ['FS', 'LO', 'RP', 'SJ', 'SO', 'UB']
for c in cities:
    graphmlpath = os.path.join(graphmldir, c + '.graphml')
    maskpath = os.path.join(maskdir, c + '.jpg')
    rect = bounds.loc[c].to_numpy()
    outdir = os.path.join(outroot, c)
    os.makedirs(outdir, exist_ok=True)
    trim_graph(graphmlpath, maskpath, rect, outdir)

