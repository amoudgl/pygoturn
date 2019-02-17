import os
import time
import argparse
import re

import torch
import numpy as np
import cv2

from model import GoNet
from helper import NormalizeToTensor, Rescale, crop_sample, bgr2rgb
from boundingbox import BoundingBox

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Man', type=str,
                    help='path to video frames')


class GOTURN:
    """Tester for OTB formatted sequences"""
    def __init__(self, root_dir, model_path, device):
        self.root_dir = root_dir
        self.device = device
        self.transform = NormalizeToTensor()
        self.scale = Rescale((224, 224))
        self.model_path = model_path
        self.model = GoNet()
        self.gt = []
        self.opts = None
        self.curr_img = None
        checkpoint = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
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
            img_prev = cv2.imread(frames[i])
            img_prev = bgr2rgb(img_prev)
            img_curr = cv2.imread(frames[i+1])
            img_curr = bgr2rgb(img_curr)
            self.img.append([img_prev, img_curr])
            lines[i+1] = re.sub('\t', ',', lines[i+1])
            lines[i+1] = re.sub(' +', ',', lines[i+1])
            bb = lines[i+1].strip().split(',')
            bb = [float(x) for x in bb]
            bb = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
            self.gt.append(bb)
        self.x = np.array(self.x)
        print(init_bbox)

    def __getitem__(self, idx):
        """
        Returns transformed torch tensor which is passed to the network.
        """
        sample = self._get_sample(idx)
        return self.transform(sample)

    def _get_sample(self, idx):
        """
        Returns cropped previous and current frame at the previous predicted
        location. Note that the images are scaled to (224,224,3).
        """
        prev = self.img[idx][0]
        curr = self.img[idx][1]
        prevbb = self.prev_rect
        prev_sample, opts_prev = crop_sample({'image': prev, 'bb': prevbb})
        curr_sample, opts_curr = crop_sample({'image': curr, 'bb': prevbb})
        curr_img = self.scale(curr_sample, opts_curr)['image']
        prev_img = self.scale(prev_sample, opts_prev)['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        self.curr_img = curr
        self.opts = opts_curr
        return sample

    def get_rect(self, sample):
        """
        Regresses the bounding box coordinates in the original image dimensions
        for an input sample.
        """
        x1, x2 = sample['previmg'], sample['currimg']
        x1 = x1.unsqueeze(0).to(self.device)
        x2 = x2.unsqueeze(0).to(self.device)
        y = self.model(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])

        # inplace conversion
        bbox.unscale(self.opts['search_region'])
        bbox.uncenter(self.curr_img, self.opts['search_location'],
                      self.opts['edge_spacing_x'], self.opts['edge_spacing_y'])
        return bbox.get_bb_list()

    def test(self):
        """
        Loops through all the frames of test sequence and tracks the target.
        Prints predicted box location on console with frame ID.
        """
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
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    tester = GOTURN(args.data_directory, args.model_weights, device)
    tester.test()
