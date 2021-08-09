import argparse
import pandas as pd

model_names = ['resnet18', 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./', help='path to svs files')
parser.add_argument('-outdir', metavar='DIR', default='./', help='output path')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--subsample', default=1, type=float, metavar='%',
                    help='fraction (0 < n < 1) of tile library to use in training')

import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
import torchvision.transforms as transforms

from models import ResNetSimCLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from pdb import set_trace
from dataloaders import TileLoader, ContrastiveLearningViewGenerator
import torch.nn as nn
import utils

def main():
    args = parser.parse_args()
    gpu = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    args.tile_size=224
    args.device = gpu


    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    data = torch.load('/lila/data/fuchs/hassan/cholangio/tStage_library.pth', encoding='latin1')['library']
    data = data.astype({"SlideID": int})
    data = data.sample(frac=args.subsample)
    train_library = data[data.Split == 'train'].reset_index(drop=True)

    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=args.tile_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          utils.GaussianBlur(kernel_size=int(0.1 * args.tile_size)),
                                          transforms.ToTensor()])



    clr_augmentations = ContrastiveLearningViewGenerator(data_transforms, 2)
    train_dataset = TileLoader(train_library, clr_augmentations, args.data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(gpu)

    criterion = torch.nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    scaler = GradScaler(enabled=args.fp16_precision)

    for epoch in range(args.epochs):
        lossMeter = utils.AverageMeter()
        accMeter = utils.AverageMeter()
        n_iter = 0
        for images, _ in train_loader:
            images = torch.cat(images, dim=0)
            images = images.to(gpu)

            with autocast(enabled=args.fp16_precision):
                features = model(images)
                logits, labels = info_nce_loss(args, features)
                loss = criterion(logits, labels)
                lossMeter.update(loss.item(), features.size(0))

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            top1, top5 = utils.accuracy(logits, labels, topk=(1, 5))
            accMeter.update(top1.item(), features.size(0))
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, n_iter+1, len(train_loader), loss=lossMeter, acc=accMeter))

            n_iter += 1

        # warmup for the first 10 epochs
        if epoch >= 10:
            scheduler.step()


        utils.save_error(epoch, lossMeter.avg, accMeter.avg,
                         os.path.join(args.outdir, 'convergence.csv'))

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch)
        if epoch % 100 == 0:
            utils.save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.outdir, checkpoint_name))




def info_nce_loss(args, features):
    views = 2 #default, do not change
    labels = torch.cat([torch.arange(args.batch_size) for i in range(views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels


if __name__ == "__main__":
    main()
