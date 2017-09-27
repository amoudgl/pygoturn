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
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

# constants
use_gpu = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='GOTURN Training')
parser.add_argument('-n', '--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size (default: 1)')
parser.add_argument('-lr', '--learning-rate', default=1e-6, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('-dir', '--save-directory', default='../saved_checkpoints/exp3/', type=str, help='path to save directory')

def main():

    args = parser.parse_args()
    print args
    # load dataset
    transform = transforms.Compose([Normalize(), ToTensor()])
    alov = datasets.ALOVDataset('../data/alov300/imagedata++/',
                                '../data/alov300/alov300++_rectangleAnnotation_full/',
                                transform)
    dataloader = DataLoader(alov, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # load model
    net = model.GoNet()
    loss_fn = torch.nn.L1Loss(size_average = False)
    if use_gpu:
        net = net.cuda()
        loss_fn = loss_fn.cuda()
    optimizer = optim.SGD(net.classifier.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if os.path.exists(args.save_directory):
        print('Directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)

    # start training
    net = train_model(net, dataloader, loss_fn, optimizer, args.epochs, args.learning_rate, args.save_directory)

def train_model(model, dataloader, criterion, optimizer, num_epochs, lr, save_dir):
    since = time.time()
    dataset_size = dataloader.dataset.len

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        optimizer = exp_lr_scheduler(optimizer, epoch, lr)
        running_loss = 0.0
        i = 0
        # iterate over data
        for data in dataloader:
            # get the inputs and labels
            x1, x2, y = data['previmg'], data['currimg'], data['currbb']

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
            print('[training] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.data[0]))
            i = i + 1
            running_loss += loss.data[0]

        epoch_loss = running_loss / dataset_size
        print('Loss: {:.4f}'.format(epoch_loss))
        val_loss = evaluate(model, dataloader, criterion, epoch)
        print('Validation Loss: {:.4f}'.format(val_loss))
        path = save_dir + 'model_n_epoch_' + str(epoch) + '_loss_' + str(round(epoch_loss, 3)) + '.pth'
        torch.save(model.state_dict(), path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
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

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

if __name__ == "__main__":
    main()
