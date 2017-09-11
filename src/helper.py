import numpy as np
from skimage import io, transform, img_as_ubyte
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
           if h > w:
               new_h, new_w = self.output_size*h/w, self.output_size
           else:
               new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        img = img_as_ubyte(img)
        # complete from here
        bb = [bb[0]*new_w/w, bb[1]*new_h/h, bb[2]*new_w/w, bb[3]*new_h/h]
        return {'image': img, 'bb':bb}

class CropPrev(object):
    """Crop the previous image using the bounding box specifications.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        image = img_as_ubyte(image)
        if (len(image.shape) == 2):
            image = np.repeat(image[...,None],3,axis=2)
        im = Image.fromarray(image)
        w = bb[2]
        h = bb[3]
        left = bb[0]-w/2
        upper = bb[1]-h/2
        right = left + 2*w
        lower = upper + 2*h
        box = (left, upper, right, lower)

        res = np.asarray(im.crop(box))
        bb = [bb[0]-left, bb[1]-upper, w, h]
        return {'image':res, 'bb':bb}

class CropCurr(object):
    """Crop the current image using the bounding box specifications.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):
        image, prevbb, currbb = sample['image'], sample['prevbb'], sample['currbb']
        image = img_as_ubyte(image)
        if (len(image.shape) == 2):
            image = np.repeat(image[...,None],3,axis=2)
        im = Image.fromarray(image)
        w = prevbb[2]
        h = prevbb[3]
        left = prevbb[0]-w/2
        upper = prevbb[1]-h/2
        right = left + 2*w
        lower = upper + 2*h
        box = (left, upper, right, lower)
        res = np.asarray(im.crop(box))
        bb = [currbb[0]-left, currbb[1]-upper, currbb[2], currbb[3]]
        return {'image':res, 'bb':bb}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bb': torch.from_numpy(bb)}

