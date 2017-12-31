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
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-weights', '--model-weights', default='../saved_checkpoints/exp3/model_n_epoch_47_loss_2.696.pth', type=str, help='path to trained model')
parser.add_argument('-save', '--save-directory', default='', type=str, help='path to save directory')
parser.add_argument('-data', '--data-directory', default='../data/OTB/Girl', type=str, help='path to video frames')

class TesterOTB:
    """Tester for OTB formatted sequences"""
    def __init__(self, root_dir, model_path, save_dir=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.model_path = model_path
        self.model = model.GoNet()
        if use_gpu:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        frames = os.listdir(root_dir + '/img')
        frames = [root_dir + "/img/" + frame for frame in frames]
        self.len = len(frames)-1
        frames = np.array(frames)
        frames.sort()
        self.x = []
        for i in xrange(self.len):
            self.x.append([frames[i], frames[i+1]])
        self.x = np.array(self.x)
#         uncomment to select rectangle manually
#         init_bbox = bbox_coordinates(self.x[0][0])
        f = open(root_dir + '/groundtruth_rect.txt')
        lines = f.readlines()
        init_bbox = lines[0].strip().split('\t')
        init_bbox = [float(x) for x in init_bbox]
        init_bbox = [init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2], init_bbox[1]+init_bbox[3]] 
        init_bbox = np.array(init_bbox)
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
        crop_prev = CropPrev(128)
        scale = Rescale((227,227))
        transform_prev = transforms.Compose([crop_prev, scale])
        prev_img = transform_prev({'image':prev, 'bb':prevbb})['image']
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_img = transform_prev({'image':curr, 'bb':prevbb})['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        return sample

    def get_rect(self, sample):
        x1, x2 = sample['previmg'], sample['currimg']
        if use_gpu:
            x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        else:
            x1, x2 = Variable(x1), Variable(x2)
        x1 = x1[None,:,:,:]
        x2 = x2[None,:,:,:]
        y = self.model(x1, x2)
        bb = y.data.cpu().numpy().transpose((1,0))
        bb = bb[:,0]
        bb = list(bb*10)
        bb = inverse_transform(bb, self.prev_rect)
        print bb
        return bb

    def test(self):
        fig,ax = plt.subplots(1)
        for i in xrange(self.len):
            sample = self[i]
            bb = self.get_rect(sample)
            im = io.imread(self.x[i][1])
            # show rectangle
            ax.clear()
            ax.imshow(im)
            rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            self.prev_rect = bb
        plt.show()

def main():
    args = parser.parse_args()
    print args
    tester = TesterOTB(args.data_directory, args.model_weights, args.save_directory)
    tester.test()

if __name__ == "__main__":
    main()
