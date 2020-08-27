#pip install torch
import torch.utils.data as data
from functions.makeDataset import make_dataset
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageFolder(data.Dataset):
    def __init__(self, opt, root, transform=None, target_transform=None,
                 loader=default_loader, erode_seg=True):
     
        self.root = root
        self.imgs = make_dataset(root, opt, erode_seg=erode_seg)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.erode_seg = erode_seg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        if self.erode_seg:
            img_path, skg_path, seg_path, eroded_seg_path, txt_path = self.imgs[index]
        else:
            img_path, skg_path, seg_path, txt_path = self.imgs[index]
        
        img = self.loader(img_path)
        skg = self.loader(skg_path)
        seg = self.loader(seg_path)
        txt = self.loader(txt_path)

        if self.erode_seg:
            eroded_seg = self.loader(eroded_seg_path)
        else:
            eroded_seg = None

        if self.transform is not None:
            if self.erode_seg:
                img, skg, seg, eroded_seg, txt = self.transform([img, skg, seg, eroded_seg, txt])
            else:
                img, skg, seg, txt = self.transform([img, skg, seg, txt])
                eroded_seg = seg

        return img, skg, seg, eroded_seg, txt


    def __len__(self):
        return len(self.imgs)