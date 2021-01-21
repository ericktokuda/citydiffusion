import os, shutil

# filename = 'y95908_x145091_k5' #UB
# filename = 'y95939_x145096_k5'
# filename = 'y95939_x145120_k5'

# filename = 'y96243_x146822_k5' #RP
# filename = 'y96247_x146839_k5'
# filename = 'y96252_x146839_k5'
# filename = 'y96257_x146842_k5'
# filename = 'y96260_x146850_k5'
# filename = 'y96276_x146884_k5'

ymin, xmin, ntiles = filename.split('_')
ymin = int(ymin[1:])
xmin = int(xmin[1:])
ntiles = int(ntiles[1:])

ys = range(ymin, ymin + ntiles)
xs = range(xmin, xmin + ntiles)

outdir = os.path.join('/tmp/', filename)

for y in ys:
    ydir = os.path.join(outdir, str(y))
    os.makedirs(ydir, exist_ok=True)
    for x in xs:
        shutil.copy('{}/{}.png'.format(y, x), ydir)
