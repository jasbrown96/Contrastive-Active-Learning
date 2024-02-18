from __future__ import print_function

import numpy as np
import torch
import os
import glob

import os
import sys
import argparse
import time
import math

import torchvision
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import load_MSTAR, train_test_split, targets_to_labels, polar_transform
from networks.resnet_big import SupConResNet, SupConResNetHead
from losses import SupConLoss

from torch.utils.data import Dataset

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=25,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')  #Changed this to resnet18 from 50
    parser.add_argument('--dataset', type=str, default='mstar',
                        choices=['mstar'], help='dataset') #added mstar as default
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--head', type=bool, default=True, help='Add extra MLP to encoder')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5, #0.07
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/{}/{}_models'.format(opt.method,opt.dataset)
    opt.tb_path = './save/{}/{}_tensorboard'.format(opt.method,opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_head_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial,opt.head)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    from torchvision.utils import _log_api_usage_once
    class RandomCenterCrop(torch.nn.Module):
      #Randomly center crop the image. If fix_ratio is true, randomly center crop of size (a,a) for size_lower<a<size_upper. if fix_ratio is False, then
      #sample randomly center crop of size (a,b) for size_lower < a,b< size_upper.

        def __init__(self, size_lower, size_upper = None, fix_ratio = True):
            super().__init__()
            _log_api_usage_once(self)
            self.size_lower = size_lower

            if size_upper is None:
              self.size_upper = size_upper
            else:
              self.size_upper = size_upper

            self.fix_ratio = fix_ratio


        def forward(self, img):
            if self.fix_ratio:
              size = np.random.randint(low=self.size_lower, high=self.size_upper)
              return torchvision.transforms.functional.center_crop(img,size)
            else:
              size_a = np.random.randint(low=self.size_lower, high=self.size_upper)
              size_b = np.random.randint(low=self.size_lower, high=self.size_upper)
              return torchvision.transforms.functional.center_crop(img, (size_a,size_b))

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(size={self.size})"

    mstar_transforms = torchvision.transforms.Compose(
        [
            RandomCenterCrop(40,88),
            torchvision.transforms.Resize(size=opt.size, antialias=None ), #Moved down from 128 b.c. memory?
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.GaussianBlur((7,7)),
        ])

    use_phase = True
    hdr, fields, mag, phase = load_MSTAR()
    labels, target_names = targets_to_labels(hdr)

    if use_phase:
        data = polar_transform(mag,phase)
    else:
        data = np.reshape(mag,(mag.shape[0],1,mag.shape[1],mag.shape[2]))

    class mstar_dataset(Dataset):
        def __init__(self, ims, labels, transform = mstar_transforms):
            self.ims = torch.from_numpy(ims).float()
            self.transform = transform
            self.labels = labels
        def __len__(self):
            return self.ims.shape[0]
        def __getitem__(self,idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return [self.transform(self.ims[idx]), self.transform(self.ims[idx])] #, self.labels[idx]

    train_dataset = mstar_dataset(data, labels, transform = mstar_transforms)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    if opt.head:
        model = SupConResNetHead(name=opt.model)
    else:
        model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, images in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = images[0].shape[0]
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        #if opt.method == 'SupCon':
    #        loss = criterion(features, labels)
        if opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)


    if opt.method == 'SimCLR':
        print('Unsupervised Learning')

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
