# need to install PIL on virtual environment
from PIL import Image
import glob
import os.path as osp
import random
import math
import torchvision
import numbers
from skimage import color
import numpy as np
import torch



class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.transform = torchvision.transforms.CenterCrop(size)

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        return [self.transform(img) for img in imgs]



class Scale(object):
    """Rescale multiple input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.transform = torchvision.transforms.Scale(size)

    def __call__(self, imgs):
        """
        Args:
            imgs (list of PIL.Image): Images to be scaled.
        Returns:
            list of PIL.Image: Rescaled images.
        """       
        return [self.transform(img) for img in imgs]

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs


class toLAB(object):
    """
    Transform to convert loaded into LAB space. 
    """
    
    def __init__(self):
        self.space = 'LAB'
        
    def __call__(self, images):
        lab_images = [color.rgb2lab(np.array(image)/255.0) for image in images]
        return lab_images

class toTensor(object):
    """Transforms a Numpy image to torch tensor"""
    
    def __init__(self):
        self.space = 'RGB'
        
    def __call__(self, pics):
        imgs = [torch.from_numpy(pic.transpose((2, 0, 1))) for pic in pics]
        return imgs


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, min_resize=0.08,max_resize=1.0,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.resize_size = (min_resize,max_resize)

    def __call__(self, imgs):
        for attempt in range(10):
            area = imgs[0].size[0] * imgs[0].size[1]
            target_area = random.uniform(self.resize_size[0], self.resize_size[1]) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
                x1 = random.randint(0, imgs[0].size[0] - w)
                y1 = random.randint(0, imgs[0].size[1] - h)

                imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
                assert([img.size == (w, h) for img in imgs])

                return [img.resize((self.size, self.size), self.interpolation) for img in imgs]

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(imgs))


def get_transforms(args):
    transforms_list = [
        RandomSizedCrop(args.image_size, args.resize_min, args.resize_max),
        RandomHorizontalFlip(),
        toTensor()
    ]
    if args.color_space == 'lab':
        transforms_list.insert(2, toLAB())
    elif args.color_space == 'rgb':
        transforms_list.insert(2, toRGB('RGB'))

    transforms = Compose(transforms_list)
    return transforms