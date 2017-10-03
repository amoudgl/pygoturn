# necessary imports
import os
import time
import copy
import datasets
import argparse
import model
import torch
from torch.autograd import Variable
from torchvision import transforms
from helper import ToTensor, Normalize, show_batch
import torch.optim as optim
import numpy as np
from helper import *
from get_bbox import *


use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='GOTURN Training')
parser.add_argument('-weights', '--model-weights', default='../saved_checkpoints/exp3/model_n_epoch_47_loss_2.696.pth', type=str, help='path to trained model')
parser.add_argument('-save', '--save-directory', default='', type=str, help='path to save directory')
parser.add_argument('-data', '--data-directory', default='../data/alov300/imagedata++/02-SurfaceCover/02-SurfaceCover_video00002', type=str, help='path to video frames')

class Tester:
    """Test Dataset for Tester"""
    def __init__(self, root_dir, model_path, save_dir=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.model_path = model_path
        self.model = model.GoNet()
        if use_gpu:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        frames = os.listdir(root_dir)
        self.len = len(frames)-1
        frames = [root_dir + "/" + frame for frame in frames]
        frames = np.array(frames)
        frames.sort()
        self.x = []
        for i in xrange(self.len):
            self.x.append([frames[i], frames[i+1]])
        self.x = np.array(self.x)
        # code for previous rectange
        init_bbox = bbox_coordinates(self.x[0][0])
        print init_bbox
        self.prev_rect = init_bbox  


    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return self.transform(sample)

    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.prev_rect
        # Crop previous image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        crop = CropPrev(128)
        scale = Rescale((227,227))
        transform = transforms.Compose([crop_prev, scale])
        prev_img = transform({'image':prev, 'bb':prevbb})['image']
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_img = transform({'image':curr, 'prevbb':prevbb})['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        return sample

    def get_rect(self, sample):
        x1, x2 = sample['previmg'], sample['currimg']
        x1 = x1[None,:,:,:]
        x2 = x2[None,:,:,:]
        y = self.model(x1, x2)
        bb = y.data.cpu().numpy().transpose((1,0))
        bb = bb[:,0]
        bb = bb*10
        prevbb = self.prev_rect
        w = prevbb[2]-prevbb[0]
        h = prevbb[3]-prevbb[1]
        new_w = 2*w
        new_h = 2*h
        ww = 227
        hh = 227
        # unscale
        bb = np.array([bb[0]*new_w/ww, bb[1]*new_h/hh, bb[2]*new_w/ww, bb[3]*new_h/hh])
        left = prevbb[0]-w/2
        top = prevbb[1]-h/2
        # uncrop
        bb = np.array([bb[0]+left, bb[1]+top, bb[2]+left, bb[3]+top])
        return bb

    def test(self):
        # show initial image with rectange
        fig,ax = plt.subplots(1)
        for i in xrange(self.len):
            sample = self[i]
            bb = self.get_rect(sample)
            # show rectangle
            im = io.imread(self.x[i][1])
            ax.clear()
            ax.imshow(im)
            rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            self.prev_rect = bb
        plt.show()

def main():
    args = parser.parse_args()
    print args
    tester = Tester(args.data_directory, args.model_weights, args.save_directory)
    #tester.test()

if __name__ == "__main__":
    main()
