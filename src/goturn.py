import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from got10k.trackers import Tracker

from model import GoNet
from helper import ToTensor, Normalize, CropPrev, Rescale, inverse_transform


class TrackerGOTURN(Tracker):
    """GOTURN got10k class for benchmark and evaluation on tracking datasets.

    This class overrides the default got10k 'Tracker' class methods namely,
    'init' and 'update'.

    Attributes:
        cuda: flag if gpu device is available
        device: device on which GOTURN is evaluated ('cuda:0', 'cpu')
        net: GOTURN pytorch model
        prev_box: previous bounding box
        prev_img: previous tracking image
        transform_img: crops image based on box and scales it to (224,224,3)
        transform_tensor: normalizes images in input sample
            (prev_img, curr_img, box) and converts it to tensor
    """
    def __init__(self, net_path=None, **kargs):
        super(TrackerGOTURN, self).__init__(
            name='PyTorchGOTURN', is_deterministic=True)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = GoNet()
        if net_path is not None:
            checkpoint = torch.load(
                net_path, map_location=lambda storage, loc: storage)
            self.net.load_state_dict(checkpoint)
            self.net.eval()
        self.net = self.net.to(self.device)

        # setup transforms
        self.prev_img = None  # previous image in numpy format
        self.prev_box = None  # follows format: [xmin, ymin, xmax, ymax]
        self.transform_img = transforms.Compose([CropPrev(),
                                                Rescale((224, 224))])
        self.transform_tensor = transforms.Compose([Normalize(), ToTensor()])

    # assumes that initial box has the format: [xmin, ymin, width, height]
    def init(self, image, box):
        image = np.asarray(image)

        # goturn helper functions expect box in [xmin, ymin, xmax, ymax] format
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        self.prev_box = box
        self.prev_img = image

    # given current image, returns target box
    def update(self, image):
        # crop current and previous image at previous box location
        image = np.asarray(image)
        prev = self.transform_img(
            {'image': self.prev_img, 'bb': self.prev_box})['image']
        curr = self.transform_img(
            {'image': image, 'bb': self.prev_box})['image']
        sample = {'previmg': prev, 'currimg': curr}
        sample = self.transform_tensor(sample)

        # do forward pass to get box
        box = np.array(self._get_rect(sample))

        # update previous box and image
        self.prev_img = image
        self.prev_box = np.copy(box)

        # convert [xmin, ymin, xmax, ymax] box to [xmin, ymin, width, height]
        # for correct evaluation by got10k toolkit
        box[2] = box[2]-box[0]
        box[3] = box[3]-box[1]
        return box

    # given previous frame and next frame, regress the bounding box coordinates
    # in the original image dimensions
    def _get_rect(self, sample):
        x1, x2 = sample['previmg'], sample['currimg']
        if self.cuda:
            x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        else:
            x1, x2 = Variable(x1), Variable(x2)
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        y = self.net(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bb = list(bb*(224./10))  # unscaling
        bb = inverse_transform(bb, self.prev_box)
        return bb
