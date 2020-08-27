import glob 
import os.path as osp
import random




def make_dataset(directory, opt, erode_seg=True):
    # opt: 'train' or 'val'
    img = glob.glob(osp.join(directory, opt + '_img/*/*.jpg'))
    img = sorted(img)
    skg = glob.glob(osp.join(directory, opt + '_skg/*/*.jpg'))
    skg = sorted(skg)
    seg = glob.glob(osp.join(directory, opt + '_seg/*/*.jpg'))
    seg = sorted(seg)
    txt = glob.glob(osp.join(directory, opt + '_txt/*/*.jpg'))
    #txt = glob.glob(osp.join(directory, opt + '_dtd_txt/*/*.jpg'))
    extended_txt = []
    #import pdb; pdb.set_trace()
    for i in range(len(skg)):
        extended_txt.append(txt[i%len(txt)])
    random.shuffle(extended_txt)
    

    if erode_seg:
        eroded_seg = glob.glob(osp.join(directory, 'eroded_' + opt + '_seg/*/*.jpg'))
        eroded_seg = sorted(eroded_seg)
        return list(zip(img, skg, seg , eroded_seg, extended_txt))
    else:
        return list(zip(img, skg, seg, extended_txt))
