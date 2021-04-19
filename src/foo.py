from segm_parser import trim_graph


graphmlpath = '/home/dufresne/temp/20210415-mask2graph/0_graphml/FS.graphml'
maskpath = '/home/dufresne/temp/20210415-mask2graph/20201114-masks/FS.jpg'
rect = [102674,140018,102736,140103]
outpath = '/tmp/fsurban.graphml'

trim_graph(graphmlpath, maskpath, rect, outpath)
