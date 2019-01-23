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
import re
import numpy as np
from helper import *
from get_bbox import *

args = None
use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-weights', '--model-weights', default='../saved_checkpoints/exp3/model_n_epoch_47_loss_2.696.pth', type=str, help='path to trained model')
parser.add_argument('-save', '--save-directory', default='', type=str, help='path to save directory')
parser.add_argument('-data', '--data-directory', default='../data/OTB/Girl', type=str, help='path to video frames')

class TesterOTB:
    """Tester for OTB formatted sequences"""
    def __init__(self, root_dir, model_path, save_dir=None):
        self.root_dir = root_dir
        self.save_dir = save_dir
        print(self.save_dir)
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.model_path = model_path
        self.model = model.GoNet()
        if use_gpu:
            self.model = self.model.cuda()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        frames = os.listdir(root_dir + '/img')
        frames = [root_dir + "/img/" + frame for frame in frames]
        self.len = len(frames)-1
        frames = np.array(frames)
        frames.sort()
        self.x = []
        self.gt = []
        f = open(root_dir + '/groundtruth_rect.txt')
        lines = f.readlines()
        lines[0] = re.sub('\t',',', lines[0])
        lines[0] = re.sub(' +',',',lines[0])
        init_bbox = lines[0].strip().split(',')
        init_bbox = [float(x) for x in init_bbox]
        init_bbox = [init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2], init_bbox[1]+init_bbox[3]]
        init_bbox = np.array(init_bbox) 
        self.prev_rect = init_bbox
        for i in range(self.len):
            self.x.append([frames[i], frames[i+1]])
            lines[i+1] = re.sub('\t',',', lines[i+1])
            lines[i+1] = re.sub(' +',',',lines[i+1])
            bb = lines[i+1].strip().split(',')
            bb = [float(x) for x in bb]
            bb = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
            self.gt.append(bb)
        self.x = np.array(self.x)
#         uncomment to select rectangle manually
#         init_bbox = bbox_coordinates(self.x[0][0])
        print(init_bbox)
        

    # returns transformed pytorch tensor which is passed directly to the network
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return self.transform(sample)

    # returns cropped and scaled previous frame and current frame
    # in numpy format
    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.prev_rect
        # Crop previous image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        crop_prev = CropPrev()
        scale = Rescale((224,224))
        transform_prev = transforms.Compose([crop_prev, scale])
        prev_img = transform_prev({'image':prev, 'bb':prevbb})['image']
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_img = transform_prev({'image':curr, 'bb':prevbb})['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        return sample

    # given previous frame and next frame, regress the bounding box coordinates
    # in the original image dimensions
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
        bb = list(bb*(224./10)) # unscaling
        bb = inverse_transform(bb, self.prev_rect)
#         print(bb)
        return bb


    def axis_aligned_iou(self, boxA, boxB):
        # convert x1,y1,w,h to x1,y1,x2,y2
#         boxA = [boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3]]
#         boxB = [boxB[0], boxB[1], boxB[0]+boxB[2], boxB[1]+boxB[3]]

            # make sure that x1,y1,x2,y2 of a box are valid 
        assert(boxA[0] <= boxA[2])
        assert(boxA[1] <= boxA[3])
        assert(boxB[0] <= boxB[2])
        assert(boxB[1] <= boxB[3])

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou    
    
    
    # loop through all the frames of test sequence and track target object
    def test(self):
        fig,ax = plt.subplots(1)
        self.model.eval()
        for i in range(self.len):
            sample = self[i]
            bb = self.get_rect(sample)
            im = io.imread(self.x[i][1])
            # show rectangle
            ax.clear()
            ax.imshow(im)
            rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            self.prev_rect = bb
            print('frame: %d, IoU = %f' % (i+2, self.axis_aligned_iou(self.gt[i], bb)))
            plt.savefig(os.path.join(self.save_dir,str(i+2)+'.jpg'))

def main():
    args = parser.parse_args()
    print(args)
    tester = TesterOTB(args.data_directory, args.model_weights, args.save_directory)
    tester.test()

if __name__ == "__main__":
    main()
