from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from segm_parser import trim_xnet
import pandas as pd
import os


graphmldir = '/home/frodo/results/citydiffusion/20210415-mask2graph/graphml/'
xnetdir = '/home/frodo/results/citydiffusion/20210501-guilherme_xnet/'
maskdir = '/home/frodo/results/citydiffusion/20210415-mask2graph/mask/'
rect = [102674,140018,102736,140103]
bounds = pd.read_csv('/home/frodo/results/citydiffusion/20210415-mask2graph/bounds.csv', index_col='city')
outroot = '/tmp/out/'

cities = ['FS', 'LO', 'RP', 'SJ', 'SO', 'UB']
# cities = ['LO']
for c in cities:
    xnetpath = os.path.join(xnetdir, c + '.xnet')
    maskpath = os.path.join(maskdir, c + '.jpg')
    rect = bounds.loc[c].to_numpy()
    outdir = os.path.join(outroot, c)
    os.makedirs(outdir, exist_ok=True)
    trim_xnet(xnetpath, maskpath, rect, outdir)

