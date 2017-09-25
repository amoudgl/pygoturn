# necessary imports
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

class Tester:
    """Test Dataset for Tester"""
    def __init__(self, root_dir, model_path, save_dir=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.model_path = model_path
        self.model = model.GoNet()
        self.model.load_state_dict(torch.load(model_path))
        frames = os.listdir(root_dir)
        self.len = len(frames)-1
        frames = [root_dir + "/" + frame for frame in frames]
        frames = np.array(frames)
        frames.sort()
        self.x = []
        for i in xrange(self.len):
            x.append([frames[i], frames[i+1]])
        self.x = np.array(self.x)
        # code for previous rectange
        self.prev_rect = init_rect


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
        for i in xrange(self.len):
            sample = self[i]
            curr_rect = self.get_rect(sample)
            # show rectangle
            self.prev_rect = curr_rect

def main():
    model_path = '../saved_checkpoints/exp2/model_n_epoch_14_loss_6.698.pth'
    data_dir = '/Neutron7/abhinav.moudgil/goturn/data/alov300/imagedata++/02-SurfaceCover/02-SurfaceCover_video00002'
    save_dir = ''
    tester = Tester(data_dir, model_path, save_dir)
    tester.test()

if __name__ == "__main__":
    main()
