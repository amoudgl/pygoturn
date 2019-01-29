import os
import time
import argparse
import re

import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from skimage import io

from model import GoNet
from helper import ToTensor, Normalize, CropPrev, Rescale, inverse_transform

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Girl', type=str,
                    help='path to video frames')


class GOTURN:
    """Tester for OTB formatted sequences"""
    def __init__(self, root_dir, model_path, use_gpu=False):
        self.root_dir = root_dir
        self.use_gpu = use_gpu
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        crop_prev = CropPrev()
        scale = Rescale((224, 224))
        self.transform_prev = transforms.Compose([crop_prev, scale])
        self.model_path = model_path
        self.model = GoNet()
        self.gt = []
        if use_gpu:
            self.model = self.model.cuda()
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=lambda storage,
                                    location: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        frames = os.listdir(root_dir + '/img')
        frames = [root_dir + "/img/" + frame for frame in frames]
        self.len = len(frames)-1
        frames = np.array(frames)
        frames.sort()
        self.x = []
        f = open(root_dir + '/groundtruth_rect.txt')
        lines = f.readlines()
        lines[0] = re.sub('\t', ',', lines[0])
        lines[0] = re.sub(' +', ',', lines[0])
        init_bbox = lines[0].strip().split(',')
        init_bbox = [float(x) for x in init_bbox]
        init_bbox = [init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2],
                     init_bbox[1] + init_bbox[3]]
        init_bbox = np.array(init_bbox)
        self.prev_rect = init_bbox
        self.img = []
        for i in range(self.len):
            self.x.append([frames[i], frames[i+1]])
            img_prev = io.imread(frames[i])
            img_curr = io.imread(frames[i+1])
            self.img.append([img_prev, img_curr])
            lines[i+1] = re.sub('\t', ',', lines[i+1])
            lines[i+1] = re.sub(' +', ',', lines[i+1])
            bb = lines[i+1].strip().split(',')
            bb = [float(x) for x in bb]
            bb = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
            self.gt.append(bb)
        self.x = np.array(self.x)
        # uncomment to select rectangle manually
        # init_bbox = bbox_coordinates(self.x[0][0])
        print(init_bbox)

    # returns transformed pytorch tensor which is passed directly to network
    def __getitem__(self, idx):
        sample = self._get_sample(idx)
        return self.transform(sample)

    # returns cropped and scaled previous frame and current frame
    # in numpy format
    def _get_sample(self, idx):
        prev = self.img[idx][0]
        curr = self.img[idx][1]
        prevbb = self.prev_rect
        prev_img = self.transform_prev({'image': prev, 'bb': prevbb})['image']
        curr_img = self.transform_prev({'image': curr, 'bb': prevbb})['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        return sample

    # given previous frame and next frame, regress the bounding box coordinates
    # in the original image dimensions
    def get_rect(self, sample):
        x1, x2 = sample['previmg'], sample['currimg']
        if self.use_gpu:
            x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        else:
            x1, x2 = Variable(x1), Variable(x2)
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        y = self.model(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bb = list(bb*(224./10))  # unscaling
        bb = inverse_transform(bb, self.prev_rect)
        return bb

    # loop through all the frames of test sequence and track target object
    def test(self):
        self.model.eval()
        st = time.time()
        for i in range(self.len):
            sample = self[i]
            bb = self.get_rect(sample)
            self.prev_rect = bb
            print("frame: {}".format(i+1), bb)
        end = time.time()
        print("fps: {:.3f}".format(self.len/(end-st)))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    use_gpu = torch.cuda.is_available()
    tester = GOTURN(args.data_directory, args.model_weights, use_gpu)
    tester.test()
