import torch
import numpy as np
import cv2
from torchvision import transforms
from got10k.trackers import Tracker

from model import GoNet
from helper import NormalizeToTensor, Rescale, BoundingBox, crop_sample


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
        transform_tensor: normalizes images and returns torch tensor.
        otps: bounding box config to unscale and uncenter network output.
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
            self.net.load_state_dict(checkpoint['state_dict'])
            self.net.eval()
        self.net = self.net.to(self.device)

        # setup transforms
        self.prev_img = None  # previous image in numpy format
        self.prev_box = None  # follows format: [xmin, ymin, xmax, ymax]
        self.scale = Rescale((224, 224))
        self.transform_tensor = transforms.Compose([NormalizeToTensor()])
        self.opts = None

    def init(self, image, box):
        """
        Initiates the tracker at given box location.
        Aassumes that the initial box has format: [xmin, ymin, width, height]
        """
        image = np.array(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # goturn helper functions expect box in [xmin, ymin, xmax, ymax] format
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        self.prev_box = box
        self.prev_img = image

    def update(self, image):
        """
        Given current image, returns target box.
        """
        image = np.array(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # crop current and previous image at previous box location
        prev_sample, opts_prev = crop_sample({'image': self.prev_img,
                                             'bb': self.prev_box})
        curr_sample, opts_curr = crop_sample({'image': image,
                                             'bb': self.prev_box})
        self.opts = opts_curr
        self.curr_img = image
        curr_img = self.scale(curr_sample, opts_curr)['image']
        prev_img = self.scale(prev_sample, opts_prev)['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
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

    def _get_rect(self, sample):
        """
        Performs forward pass through the GOTURN network to regress
        bounding box coordinates in the original image dimensions.
        """
        x1, x2 = sample['previmg'], sample['currimg']
        x1 = x1.unsqueeze(0).to(self.device)
        x2 = x2.unsqueeze(0).to(self.device)
        y = self.net(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])

        # inplace conversion
        bbox.unscale(self.opts['search_region'])
        bbox.uncenter(self.curr_img,
                      self.opts['search_location'],
                      self.opts['edge_spacing_x'],
                      self.opts['edge_spacing_y'])
        return bbox.get_bb_list()
