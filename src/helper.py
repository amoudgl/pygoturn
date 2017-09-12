import numpy as np
from skimage import io, transform, img_as_ubyte
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        prev_img, curr_img, currbb = sample['previmg'], sample['currimg'], sample['currbb']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        prev_img = prev_img.transpose((2, 0, 1))
        curr_img = curr_img.transpose((2, 0, 1))
        return {'previmg': torch.from_numpy(prev_img),
                'currimg': torch.from_numpy(curr_img),
                'currbb': torch.from_numpy(currbb)
                }

class Normalize(object):
    """Normalize sample images"""

    def __call__(self, sample):
        prev_img, curr_img, currbb = sample['previmg'], sample['currimg'], sample['currbb']
        self.mean = [104, 117, 123]
        prev_img = prev_img.astype(float)
        curr_img = curr_img.astype(float)
        prev_img -= np.array(self.mean).astype(float)
        curr_img -= np.array(self.mean).astype(float)
        currbb /= 10;
        return {'previmg': prev_img,
                'currimg': curr_img,
                'currbb': currbb
                }


def show_batch(sample_batched):
    """Show images with bounding boxes for a batch of samples."""

    previmg_batch, currimg_batch, currbb_batch = \
            sample_batched['previmg'], sample_batched['currimg'], sample_batched['currbb']
    batch_size = len(previmg_batch)
    im_size = previmg_batch.size(2)
    grid1 = utils.make_grid(previmg_batch)
    grid2 = utils.make_grid(currimg_batch)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(grid1.numpy().transpose((1, 2, 0)))
    axarr[0].set_title('Previous frame images')
    axarr[1].imshow(grid2.numpy().transpose((1, 2, 0)))
    axarr[1].set_title('Current frame images with bounding boxes')
    for i in range(batch_size):
        bb = currbb_batch[i]
        bb = bb.numpy()
        rect = patches.Rectangle((bb[0]+i*im_size, bb[1]),bb[2],bb[3],linewidth=1,edgecolor='r',facecolor='none')
        axarr[1].add_patch(rect)
    plt.show()

