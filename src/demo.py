import os
import argparse

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from test import GOTURN

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Man', type=str,
                    help='path to video frames')
parser.add_argument('-s', '--save-directory',
                    default='../result',
                    type=str, help='path to save directory')


def axis_aligned_iou(boxA, boxB):
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


def save(ax, im, bb, gt_bb, idx):
    ax.clear()
    ax.imshow(im)
    goturn_box = patches.Rectangle((bb[0], bb[1]),
                                   bb[2]-bb[0], bb[3]-bb[1],
                                   linewidth=2, edgecolor='r',
                                   facecolor='none')
    gt_box = patches.Rectangle((gt_bb[0], gt_bb[1]),
                               gt_bb[2]-gt_bb[0], gt_bb[3]-gt_bb[1],
                               linewidth=2, edgecolor='w',
                               facecolor='none')
    ax.add_patch(goturn_box)
    ax.add_patch(gt_box)
    props = dict(boxstyle='round', facecolor='red', alpha=0.5)
    ax.text(0.05, 0.95, "GOTURN", transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.set_axis_off()
    plt.savefig(os.path.join(args.save_directory, str(idx)+'.jpg'))


def main(args):
    use_gpu = torch.cuda.is_available()
    tester = GOTURN(args.data_directory,
                    args.model_weights,
                    use_gpu)
    fig, ax = plt.subplots(1)
    if os.path.exists(args.save_directory):
        print('Save directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)
    # save initial frame with bounding box
    save(ax, tester.img[0][0], tester.prev_rect, tester.prev_rect, 1)
    tester.model.eval()

    # loop through sequence images
    for i in range(tester.len):
        # get torch input tensor
        sample = tester[i]

        # predict box
        bb = tester.get_rect(sample)
        gt_bb = tester.gt[i]
        tester.prev_rect = bb

        # save current image with predicted rectangle and gt box
        im = tester.img[i][1]
        save(ax, im, bb, gt_bb, i+2)

        # print stats
        print('frame: %d, IoU = %f' % (
            i+2, axis_aligned_iou(gt_bb, bb)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
