import torch
import numpy as np
from skimage import color


class toRGB_(object):
    """
    Transform to convert loaded into LAB space. 
    """
    
    def __init__(self):
        self.space = 'LAB'
        
    def __call__(self, images):
        images = np.transpose(images.numpy(), (1, 2, 0))
        rgb_images = [(np.array(image)/255.0) for image in images]
        return rgb_images


class toRGB(object):
    """
    Transform to convert loaded into RGB color space. 
    """
    
    def __init__(self, space ='LAB'):
        self.space = space
        
    def __call__(self, images):
        if self.space =='LAB':
            # npimg = np.transpose(np.array(images), (1, 2, 0))
            # print(image)
            rgb_img = [np.transpose(color.lab2rgb(np.transpose(image, (1,2,0))), (2,0,1)) for image in images]
        elif self.space =='RGB':
            # print np.shape(images)
            # images = np.transpose(images.numpy(), (1, 2, 0))
            rgb_img = [(np.array(image)/255.0) for image in images]

        return rgb_img

def vis_image(img, color='lab'):
    if torch.cuda.is_available():
        img = img.cpu()

    img = img.numpy()

    if color == 'lab':
        ToRGB = toRGB()
    elif color =='rgb':
        ToRGB = toRGB('RGB')

    # print np.shape(img)
    img_np = ToRGB(img)

    return img_np

def vis_patch(img, skg, texture_location, color='lab'):
    batch_size, _, _, _ = img.size()
    if torch.cuda.is_available():
        img = img.cpu()
        skg = skg.cpu()

    img = img.numpy()
    skg = skg.numpy()

    if color == 'lab':
        ToRGB = toRGB()
        
    elif color =='rgb':
        ToRGB = toRGB('RGB')
        
    img_np = ToRGB(img)
    skg_np = ToRGB(skg)

    vis_skg = np.copy(skg_np)
    vis_img = np.copy(img_np)

    # print np.shape(vis_skg)
    for i in range(batch_size):
        for text_loc in texture_location[i]:
            xcenter, ycenter, size = text_loc
            xcenter = max(xcenter-int(size/2),0) + int(size/2)
            ycenter = max(ycenter-int(size/2),0) + int(size/2)
            vis_skg[
                i, :,
                int(xcenter-size/2):int(xcenter+size/2),
                int(ycenter-size/2):int(ycenter+size/2)
            ] = vis_img[
                    i, :,
                    int(xcenter-size/2):int(xcenter+size/2),
                    int(ycenter-size/2):int(ycenter+size/2)
                ]

    return vis_skg