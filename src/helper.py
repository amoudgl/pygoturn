import numpy as np
import math
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
    """Rescale image and bounding box.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    """
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
    """Crop the previous frame image using the bounding box specifications.

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
        w = bb[2]-bb[0]
        h = bb[3]-bb[1]
        left = bb[0]-w/2
        top = bb[1]-h/2
        right = left + 2*w
        bottom = top + 2*h
        box = (left, top, right, bottom)
        box = tuple([int(math.floor(x)) for x in box])
        res = np.array(im.crop(box))
        bb = [bb[0]-left, bb[1]-top, bb[2]-left, bb[3]-top]
        return {'image':res, 'bb':bb}

class CropCurr(object):
    """Crop the current frame image using the bounding box specifications.

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
        w = prevbb[2]-prevbb[0]
        h = prevbb[3]-prevbb[1]
        left = prevbb[0]-w/2
        top = prevbb[1]-h/2
        right = left + 2*w
        bottom = top + 2*h
        box = (left, top, right, bottom)
        box = tuple([int(math.floor(x)) for x in box])
        res = np.array(im.crop(box))
        bb = [currbb[0]-left, currbb[1]-top, currbb[2]-left, currbb[3]-top]
        return {'image':res, 'bb':bb}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        prev_img = prev_img.transpose((2, 0, 1))
        curr_img = curr_img.transpose((2, 0, 1))
        if 'currbb' in sample:
            currbb = sample['currbb']
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float(),
                    'currbb': torch.from_numpy(currbb).float()
                   }
        else:
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float()
                   }

class Normalize(object):
    """Returns image with zero mean and scales bounding box by factor of 10."""

    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.mean = [104, 117, 123]
        prev_img = prev_img.astype(float)
        curr_img = curr_img.astype(float)
        prev_img -= np.array(self.mean).astype(float)
        curr_img -= np.array(self.mean).astype(float)

        if 'currbb' in sample:
            currbb = sample['currbb']
            currbb = currbb*(10./227);
            return {'previmg': prev_img,
                    'currimg': curr_img,
                    'currbb': currbb
                    }
        else:
            return {'previmg': prev_img,
                    'currimg': curr_img
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
        rect = patches.Rectangle((bb[0]+i*im_size, bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
        axarr[1].add_patch(rect)
    plt.show()

# given currbb output from model and previous bounding box values in
# the original image dimensions, return the current bouding box values
# in the orignal image dimensions
def inverse_transform(currbb, orig_prevbb):
    # unscaling
    patch_width = (orig_prevbb[2]-orig_prevbb[0])*2
    patch_height = (orig_prevbb[3]-orig_prevbb[1])*2
    # input image size to network
    net_w = 227
    net_h = 227
    unscaledbb = [currbb[0]*patch_width/net_w,
                  currbb[1]*patch_height/net_h,
                  currbb[2]*patch_width/net_w,
                  currbb[3]*patch_height/net_h]
    # uncropping
    bb = orig_prevbb
    w = bb[2]-bb[0]
    h = bb[3]-bb[1]
    left = bb[0]-w/2
    top = bb[1]-h/2
    orig_currbb = [left+unscaledbb[0], top+unscaledbb[1], left+unscaledbb[2], top+unscaledbb[3]]
    return orig_currbb
