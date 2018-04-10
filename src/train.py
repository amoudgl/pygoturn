# necessary imports
import os, sys
import time
import copy
import datasets
from datasets import ALOVDataset, ILSVRC2014_DET_Dataset
import argparse
import model
import torch
from torch.autograd import Variable
from torchvision import transforms
from helper import ToTensor, Normalize, show_batch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from helper import *
from multiprocessing.dummy import Pool as ThreadPool
from tensorboardX import SummaryWriter

# constants
use_gpu = torch.cuda.is_available()
kSaveModel = 20000 # save model after every 20000 steps
kGeneratedExamplesPerImage = 10; # generate 10 synthetic samples per image in a dataset
transform = transforms.Compose([Normalize(), ToTensor()])
writer = SummaryWriter()

args = None
parser = argparse.ArgumentParser(description='GOTURN Training')
parser.add_argument('-n', '--num-batches', default=500000, type=int, help='number of total batches to run')
parser.add_argument('-lr', '--learning-rate', default=1e-6, type=float, help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--lr-decay-step', default=100000, type=int, help='steps after which learning rate decays')
parser.add_argument('-save', '--save-directory', default='../saved_checkpoints/exp3/', type=str, help='path to save directory')
parser.add_argument('-lshift', '--lambda-shift-frac', default=5, type=float, help='lambda-shift for random cropping')
parser.add_argument('-lscale', '--lambda-scale-frac', default=15, type=float, help='lambda-scale for random cropping')
parser.add_argument('-minsc', '--min-scale', default=-0.4, type=float, help='min-scale for random cropping')
parser.add_argument('-maxsc', '--max-scale', default=0.4, type=float, help='max-scale for random cropping')
parser.add_argument('-seed', '--manual-seed', default=800, type=int, help='set manual seed value')

def main():

    global args
    args = parser.parse_args()
    print(args)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_gpu:
        torch.cuda.manual_seed(args.manual_seed)
    # load datasets
    alov = ALOVDataset('../data/alov300/imagedata++/',
                       '../data/alov300/alov300++_rectangleAnnotation_full/',
                       transform)
    imagenet = ILSVRC2014_DET_Dataset('../data/imagenet_img/',
                                       '../data/imagenet_bbox/',
                                       transform,
                                       args.lambda_shift_frac,
                                       args.lambda_scale_frac,
                                       args.min_scale,
                                       args.max_scale)
    # list of datasets to train on
    #datasets = [imagenet]
	# datasets = [imagenet]
    datasets = [alov, imagenet]
    # load model
    net = model.GoNet()
    loss_fn = torch.nn.L1Loss(size_average = False)
    if use_gpu:
        net = net.cuda()
        loss_fn = loss_fn.cuda()

    # initialize optimizer
    optimizer = optim.SGD(net.classifier.parameters(),
                          lr=args.learning_rate,
                          weight_decay=0.0005)

    if os.path.exists(args.save_directory):
        print('Directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)

    # start training
    net = train_model(net, datasets, loss_fn, optimizer)

def exp_lr_scheduler(optimizer, step, init_lr, gamma, snapshot=50000):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * gamma
    if step % snapshot == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr

# batch formation regimen pseudocode
# curr_idx = 0
# running_batch = np.zeros((50,4))
# while 1:
#    temp_batch = tr_batch()
#    print "curr_idx = ", curr_idx
#    if curr_idx + 11 <= 50:
#        running_batch[curr_idx:curr_idx+11] = temp_batch
#        curr_idx = curr_idx + 11
#    elif (curr_idx + 11) > 50:
#        count_in = 50-curr_idx
#        print "count_in = ", count_in
#        running_batch[curr_idx:curr_idx+count_in] = temp_batch[:count_in]
#        model_batch = running_batch
#        curr_idx = (curr_idx+11)%50
#        running_batch = np.zeros((50,4))
#        running_batch[:curr_idx] = temp_batch[count_in:]
#        break

def get_training_batch(running_batch_idx, running_batch, dataset):
    done = 0
    N = kGeneratedExamplesPerImage+1
    train_batch = None

    x1_batch, x2_batch, y_batch = make_transformed_samples(dataset, args)
    if running_batch_idx + N <= 50:
        running_batch['previmg'][running_batch_idx:running_batch_idx+N,:,:,:] = x1_batch
        running_batch['currimg'][running_batch_idx:running_batch_idx+N,:,:,:] = x2_batch
        running_batch['currbb'][running_batch_idx:running_batch_idx+N,:] = y_batch
        running_batch_idx = (running_batch_idx+N)
    elif running_batch_idx + N > 50:
        done = 1
        count_in = 50-running_batch_idx
 #       print "count_in =", count_in
        if count_in > 0:
            running_batch['previmg'][running_batch_idx:running_batch_idx+count_in,:,:,:] = x1_batch[:count_in,:,:,:]
            running_batch['currimg'][running_batch_idx:running_batch_idx+count_in,:,:,:] = x2_batch[:count_in,:,:,:]
            running_batch['currbb'][running_batch_idx:running_batch_idx+count_in,:] = y_batch[:count_in,:]
            running_batch_idx = (running_batch_idx+N) % 50
            train_batch = running_batch
            running_batch = {'previmg': torch.Tensor(50, 3, 227, 227),
                            'currimg': torch.Tensor(50, 3, 227, 227),
                            'currbb': torch.Tensor(50, 4)}
            running_batch['previmg'][:running_batch_idx,:,:,:] = x1_batch[count_in:,:,:,:]
            running_batch['currimg'][:running_batch_idx,:,:,:] = x2_batch[count_in:,:,:,:]
            running_batch['currbb'][:running_batch_idx,:] = y_batch[count_in:,:]
        else:
            train_batch = running_batch
            running_batch_idx = 0
            running_batch['previmg'][running_batch_idx:running_batch_idx+N,:,:,:] = x1_batch
            running_batch['currimg'][running_batch_idx:running_batch_idx+N,:,:,:] = x2_batch
            running_batch['currbb'][running_batch_idx:running_batch_idx+N,:] = y_batch
            running_batch_idx = (running_batch_idx+N)

    return running_batch, train_batch, done, running_batch_idx
            

def make_transformed_samples(dataset, args):

    idx = np.random.randint(dataset.len, size=1)[0]
    orig_sample = dataset.get_orig_sample(idx) # unscaled original sample (single image and bb)
    true_sample = dataset.get_sample(idx) # cropped scaled sample (two frames and bb)
    true_tensor = transform(true_sample)

    origimg = orig_sample['image']
    origbb = orig_sample['bb']
    x1_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 3, 227, 227)
    x2_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 3, 227, 227)
    y_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 4)

    # initialize batch with the true sample
    x1_batch[0,:,:,:] = true_tensor['previmg']
    x2_batch[0,:,:,:] = true_tensor['currimg']
    y_batch[0,:] = true_tensor['currbb']

    for i in range(kGeneratedExamplesPerImage):
        sample = {'image': origimg, 'bb': origbb}
        prevbb = random_crop(sample,
                             args.lambda_scale_frac,
                             args.lambda_shift_frac,
                             args.min_scale,
                             args.max_scale)

        # Crop previous image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        crop_curr = transforms.Compose([CropCurr()])
        scale = Rescale((227,227))
        transform_prev = transforms.Compose([CropPrev(), scale])
        prev_img = transform_prev({'image':origimg, 'bb':origbb})['image']
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_obj = crop_curr({'image':origimg, 'prevbb':prevbb, 'currbb':origbb})
        curr_obj = scale(curr_obj)
        curr_img = curr_obj['image']
        currbb = curr_obj['bb']
        currbb = np.array(currbb)
        sample = {'previmg': prev_img,
                'currimg': curr_img,
                'currbb' : currbb
                }
        sample = transform(sample)
        x1_batch[i+1,:,:,:] = sample['previmg']
        x2_batch[i+1,:,:,:] = sample['currimg']
        y_batch[i+1,:] = sample['currbb']

    return x1_batch, x2_batch, y_batch

def train_model(model, datasets, criterion, optimizer):

    global writer
    since = time.time()
    curr_loss = 0
    lr = args.learning_rate
    running_batch_idx = 0
    running_batch = {'previmg': torch.Tensor(50, 3, 227, 227),
		     'currimg': torch.Tensor(50, 3, 227, 227),
		     'currbb': torch.Tensor(50, 4)}

    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)
	
    itr = 0
    while itr < args.num_batches:

        model.train()
        if itr > 0 and itr % args.lr_decay_step == 0:
            optimizer, lr = exp_lr_scheduler(optimizer, itr, lr, args.gamma)

        # train on datasets
        # usually ALOV and ImageNet
        for i, dataset in enumerate(datasets):
            # generate random index
            # rand_idx = np.random.randint(sz, size=1)[0]

            # get training batch by generating new synthetic samples
            # x1, x2, y = make_training_samples(rand_idx, dataset, args)

            running_batch, train_batch, done, running_batch_idx = get_training_batch(running_batch_idx, running_batch, dataset)
#            print 'running_batch_idx =', running_batch_idx
            if done:
                x1 = train_batch['previmg']
                x2 = train_batch['currimg']
                y = train_batch['currbb']
                # wrap them in Variable
                if use_gpu:
                    x1, x2, y = Variable(x1.cuda()), \
                        Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
                else:
                    x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(x1, x2)
                loss = criterion(output, y)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                curr_loss = loss.data[0]
                itr = itr + 1
                writer.add_scalar('train/batch_loss', curr_loss, itr)
                print('[training] step = %d/%d, loss = %f' % (itr, args.num_batches, curr_loss))
                # sys.stdout.flush()

    #         val_loss = evaluate(model, dataloader, criterion, epoch)
    #         print('Validation Loss: {:.4f}'.format(val_loss))
                if itr > 0 and itr % kSaveModel == 0:
                    path = args.save_directory + 'model_n_batch_' + str(itr) + '_loss_' + str(round(curr_loss, 3)) + '.pth'
                    torch.save(model.state_dict(), path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    return model

def evaluate(model, dataloader, criterion, epoch):
    model.eval()
    dataset = dataloader.dataset
    running_loss = 0
    # test on a sample sequence from training set itself
    for i in xrange(64):
        sample = dataset[i]
        sample['currimg'] = sample['currimg'][None,:,:,:]
        sample['previmg'] = sample['previmg'][None,:,:,:]
        x1, x2 = sample['previmg'], sample['currimg']
        y = sample['currbb']
        x1 = Variable(x1.cuda())
        x2 = Variable(x2.cuda())
        y = Variable(y.cuda(), requires_grad=False)
        output = model(x1, x2)
        loss = criterion(output, y)
        running_loss += loss.data[0]
        print('[validation] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.data[0]))

    seq_loss = running_loss/64
    return seq_loss

if __name__ == "__main__":
    main()
