from __future__ import print_function, division
import os
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import cv2
from torch.utils.data import Dataset

from helper import shift_crop_training_sample, crop_sample, Rescale, BoundingBox, cropPadImage

warnings.filterwarnings("ignore")


class ALOVDataset(Dataset):
    """ALOV Tracking Dataset"""

    def __init__(self, root_dir, target_dir, transform=None, input_size=227):
        super(ALOVDataset, self).__init__()
        self.exclude = ['01-Light_video00016',
                        '01-Light_video00022',
                        '01-Light_video00023',
                        '02-SurfaceCover_video00012',
                        '03-Specularity_video00003',
                        '03-Specularity_video00012',
                        '10-LowContrast_video00013']
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.input_size = input_size
        self.transform = transform
        self.x, self.y = self._parse_data(root_dir, target_dir)
        self.len = len(self.y)

    # return size of dataset
    def __len__(self):
        return self.len

    # return transformed sample
    def __getitem__(self, idx):
        sample, _ = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    def _parse_data(self, root_dir, target_dir):
        x = []
        y = []
        envs = os.listdir(target_dir)
        num_anno = 0
        print('Parsing ALOV dataset...')
        for env in envs:
            env_videos = os.listdir(root_dir + env)
            for vid in env_videos:
                if vid in self.exclude:
                    continue
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
                num_anno += len(annotations)
                for i in range(len(frame_idxs)-1):
                    idx = frame_idxs[i]
                    next_idx = frame_idxs[i+1]
                    x.append([frames[idx], frames[next_idx]])
                    y.append([annotations[i], annotations[i+1]])
        x = np.array(x)
        y = np.array(y)
        self.len = len(y)
        print('ALOV dataset parsing done.')
        print('Total number of annotations in ALOV dataset = %d' % (num_anno))
        return x, y

    # return sample without transformation
    # for visualization purpose
    def get_sample(self, idx):
        opts_curr = {}
        curr_sample = {}
        curr_img = self.get_orig_sample(idx, 1)['image']
        currbb = self.get_orig_sample(idx, 1)['bb']
        prevbb = self.get_orig_sample(idx, 0)['bb']
        bbox_curr_shift = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
        rand_search_region, rand_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, curr_img)
        bbox_curr_gt = BoundingBox(prevbb[0], prevbb[1], prevbb[2], prevbb[3])
        bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
        curr_sample['image'] = rand_search_region
        curr_sample['bb'] = bbox_gt_recentered.get_bb_list()

        # additional options for visualization
        opts_curr['edge_spacing_x'] = edge_spacing_x
        opts_curr['edge_spacing_y'] = edge_spacing_y
        opts_curr['search_location'] = rand_search_location
        opts_curr['search_region'] = rand_search_region

        # build prev sample
        prev_sample = self.get_orig_sample(idx, 0)
        prev_sample, opts_prev = crop_sample(prev_sample)

        # scale
        scale = Rescale((self.input_size, self.input_size))
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)
        training_sample = {'previmg': scaled_prev_obj['image'],
                           'currimg': scaled_curr_obj['image'],
                           'currbb': scaled_curr_obj['bb']}
        return training_sample, opts_curr

    # returns original image and bounding box
    def get_orig_sample(self, idx, i=1):
        curr = cv2.imread(self.x[idx][i])
        currbb = self.get_bb(self.y[idx][i])
        sample = {'image': curr, 'bb': currbb}
        return sample

    # returns bounding box in the format: (left, upper, width, height)
    def get_bb(self, ann):
        ann = ann.strip().split(' ')
        left = min(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        top = min(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        right = max(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        bottom = max(float(ann[2]), float(ann[4]),
                     float(ann[6]), float(ann[8]))
        return [left, top, right, bottom]

    # helper function to display image at a particular index with grount truth
    # bounding box
    # arguments: (idx, i)
    #     idx: index
    #     i: 0 for previous frame and 1 for current frame
    def show(self, idx, i):
        sample = self.get_orig_sample(idx, i)
        image = sample['image']
        bb = sample['bb']
        bb = [int(val) for val in bb]
        image = cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.imshow('Current image with bb', image)

    # helper function to display sample, which is passed to neural net
    # display previous frame and current frame with bounding box
    def show_sample(self, idx):
        x, _ = self.get_sample(idx)
        prev_image = x['previmg']
        curr_image = x['currimg']
        bb = x['currbb']
        bb = [int(val) for val in bb]
        curr_image = cv2.rectangle(curr_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        concat_image = np.hstack((prev_image, curr_image))
        cv2.imshow('alov dataset sample: ' + str(idx), concat_image)
        cv2.waitKey(0)

class ILSVRC2014_DET_Dataset(Dataset):
    """ImageNet 2014 Detection Dataset"""

    def __init__(self, image_dir,
                 bbox_dir,
                 bb_params,
                 transform=None,
                 input_size=224):
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.sz = input_size
        self.bb_params = bb_params
        self.x, self.y = self._parse_data(self.image_dir, self.bbox_dir)

    # return transformed sample
    def __getitem__(self, idx):
        sample, _ = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    # returns size of dataset
    def __len__(self):
        return self.len

    # parses xml file
    # returns list of all the bounding boxes in the given file
    def get_bb(self, bbox_filepath):
        tree = ET.parse(bbox_filepath)
        root = tree.getroot()
        sz = [float(root.find('size').find('width').text),
              float(root.find('size').find('height').text)]
        bboxes = []
        for obj in root.findall('object'):
            xmin = obj.find('bndbox').find('xmin').text
            ymin = obj.find('bndbox').find('ymin').text
            xmax = obj.find('bndbox').find('xmax').text
            ymax = obj.find('bndbox').find('ymax').text
            bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
            bboxes.append(bbox)
        return sz, bboxes

    # return sample without transformation
    # for visualization purpose
    def get_sample(self, idx):
        sample = self.get_orig_sample(idx)
        # unscaled current image crop with box
        curr_sample, opts_curr = shift_crop_training_sample(sample, self.bb_params)
        # unscaled previous image crop with box
        prev_sample, opts_prev = crop_sample(sample)
        scale = Rescale((self.sz, self.sz))
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)
        training_sample = {'previmg': scaled_prev_obj['image'],
                           'currimg': scaled_curr_obj['image'],
                           'currbb': scaled_curr_obj['bb']}
        return training_sample, opts_curr

    # returns original image and bounding box
    def get_orig_sample(self, idx):
        curr = cv2.imread(self.x[idx])
        currbb = self.y[idx]
        sample = {'image': curr, 'bb': currbb}
        return sample

    # given list of object annotations, filter objects which cover atleast
    # 66% of the image in either dimension
    def filter_ann(self, sz, ann):
        ans = []
        for an in ann:
            an_width = an[2]-an[0]
            an_height = an[3]-an[1]
            area_constraint = an_width > 0 and \
                an_height > 0 and an_width*an_height > 0
            if an_width <= (0.66)*sz[0] and \
               an_height <= (0.66)*sz[1] and \
               area_constraint:
                ans.append(an)
        return ans

    # returns object bounding boxes in 'y' vector
    # and corresponding image path in 'x' vector
    def _parse_data(self, image_dir, bbox_dir):
        print('Parsing ImageNet dataset...')
        folders = os.listdir(image_dir)
        x = []  # contains path to image files
        y = []  # contains bounding boxes
        for folder in folders:
            images = os.listdir(os.path.join(image_dir, folder))
            bboxes = os.listdir(os.path.join(bbox_dir, folder))
            images.sort()
            bboxes.sort()
            images = [os.path.join(os.path.join(image_dir, folder), image) for image in images]
            bboxes = [os.path.join(os.path.join(bbox_dir, folder), bbox) for bbox in bboxes]
            annotations = []
            for bbox, image in zip(bboxes, images):
                sz, ann = self.get_bb(bbox)
                # filter bounding boxes
                ann = self.filter_ann(sz, ann)
                if ann:
                    annotations.extend(ann)
                    length = len(ann)*[image]
                    x.extend(length)
            if annotations:
                y.extend(annotations)
        self.len = len(y)
        print('ImageNet dataset parsing done.')
        # should return 239283
        print('Total number of annotations in ImageNet Dataset =', self.len)
        return x, y

    # displays object at specific index with bounding box
    def display_object(self, idx):
        sample = self.get_orig_sample(idx)
        image = sample['image']
        bb = sample['bb']
        bb = [int(val) for val in bb]
        image = cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.imshow('Current image with bb', image)
        cv2.waitKey(0)

    # helper function to display sample, which is passed to neural net
    # display previous frame and current frame with bounding box
    def show_sample(self, idx):
        x, _ = self.get_sample(idx)
        prev_image = x['previmg']
        curr_image = x['currimg']
        bb = x['currbb']
        bb = [int(val) for val in bb]
        curr_image = cv2.rectangle(curr_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        concat_image = np.hstack((prev_image, curr_image))
        cv2.imshow('imagenet dataset sample: ' + str(idx), concat_image)
        cv2.waitKey(0)
