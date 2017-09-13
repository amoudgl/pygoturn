from __future__ import print_function, division
import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from helper import *

import warnings
warnings.filterwarnings("ignore")

class ALOVDataset(Dataset):
    """ALOV Tracking Dataset"""

    def __init__(self, root_dir, target_dir, transform=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.y = []
        self.x = []
        self.transform = transform
        envs = os.listdir(target_dir)  
        for env in envs:
            env_videos = os.listdir(root_dir + env)
            for vid in env_videos:
                vid_src = self.root_dir + env + "/" + vid
                vid_ann = self.target_dir + env + "/" + vid + ".ann"
                frames = os.listdir(vid_src)
                frames.sort()
                frames = [vid_src + "/" + frame for frame in frames]
                f = open(vid_ann, "r")
                annotations = f.readlines()
                f.close()
                frame_idxs = [int(ann.split(' ')[0])-1 for ann in annotations]
                frames = np.array(frames)
                for i in xrange(len(frame_idxs)-1):
                    idx = frame_idxs[i]
                    next_idx = frame_idxs[i+1]
                    self.x.append([frames[idx], frames[next_idx]])
                    self.y.append([annotations[i], annotations[i+1]])
                # frames = list(frames[frame_idxs])
                # self.x.extend(frames)
        self.len = len(self.y) # subtract -1 because of tuple input
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
	sample = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.get_bb(self.y[idx][0]) 
        currbb = self.get_bb(self.y[idx][1])
        # Crop previous image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        crop_prev = CropPrev(128)
        crop_curr = CropCurr(128)
        scale = Rescale((227,227))
        transform_prev = transforms.Compose([crop_prev, scale])
        prev_img = transform_prev({'image':prev, 'bb':prevbb})['image']
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_obj = crop_curr({'image':curr, 'prevbb':prevbb, 'currbb':currbb})
        curr_obj = scale(curr_obj)
        curr_img = curr_obj['image']
        currbb = curr_obj['bb']
        currbb = np.asarray(currbb)
        sample = {'previmg': prev_img, 
                  'currimg': curr_img,
                  'currbb' : currbb
                  }
	return sample

    # given annotation, returns bounding box in the format: (left, upper, width, height)
    def get_bb(self, ann):
        ann = ann.strip().split(' ')
        left = min(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        top = min(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        right = max(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        bottom = max(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        return [left, top, right, bottom]
        # return [left, upper, right-left, lower-upper]

    # helper function to display images with ground truth bounding box
    def show(self, idx, i):
        im = io.imread(self.x[idx][i])
        bb = self.get_bb(self.y[idx][i])
        fig,ax = plt.subplots(1)
        ax.imshow(im)
        rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.show()

    def show_sample(self, idx):
        x = self.get_sample(idx)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(x['previmg'])
        ax2.imshow(x['currimg'])
        bb = x['currbb']
        rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rect)
        plt.show()
