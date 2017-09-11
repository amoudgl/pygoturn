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

    def __init__(self, root_dir, target_dir, transform_prev=None, transform_curr=None):
	self.root_dir = root_dir
	self.target_dir = target_dir
	self.transform_prev = transform_prev
        self.transform_curr = transform_curr
	self.y = []
	self.x = []
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
		self.y.extend(annotations)
		frame_idxs = [int(ann.split(' ')[0])-1 for ann in annotations]
		frames = np.array(frames)
		frames = list(frames[frame_idxs])
		self.x.extend(frames)
	self.len = len(self.y)

    def __len__(self):
	return self.len

    def __getitem__(self, idx):
	prev = io.imread(self.x[idx])
        curr = io.imread(self.x[idx+1])
	prev_ann = self.y[idx].strip().split(' ')
        curr_ann = self.y[idx+1].strip().split(' ')
	prevbb = [float(prev_ann[3]), float(prev_ann[4]), float(prev_ann[1])-float(prev_ann[3]), float(prev_ann[6])-float(prev_ann[4])]
        currbb = [float(curr_ann[3]), float(curr_ann[4]), float(curr_ann[1])-float(curr_ann[3]), float(curr_ann[6])-float(curr_ann[4])]
        if self.transform_prev:
            sample = {'image':prev, 'bb':prevbb}
            x = self.transform_prev(sample)
            prev, prevbb = x['image'], x['bb']
        if self.transform_curr:
            sample = {'image':curr, 'bb':currbb}
            x = self.transform_curr(sample)
            curr, currbb = x['image'], x['bb']
	sample = {'previmg': prev, 
                  'currimg': curr,
                  'currbb' : currbb
                  }
	return sample

    # helper function to display images with ground truth bounding box
    def show(self, idx):
	im = io.imread(self.x[idx])
	ann = self.y[idx].strip().split(' ')
        bb = [float(ann[3]), float(ann[4]), float(ann[1])-float(ann[3]), float(ann[6])-float(ann[4])]
	fig,ax = plt.subplots(1)
	ax.imshow(im)
	rect = patches.Rectangle((bb[0], bb[1]),bb[2],bb[3],linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)
	plt.show()
