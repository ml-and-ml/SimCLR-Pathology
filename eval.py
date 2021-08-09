import argparse
import pandas as pd

model_names = ['resnet18', 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./', help='path to svs files')
parser.add_argument('-library', metavar='FILE', default='./*.csv', help='path to csv tile library')
parser.add_argument('-checkpoint', metavar='FILE', default='./*.pth.tar', help='path to training checkpoint')
parser.add_argument('-outdir', metavar='DIR', default='./', help='output path')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--subsample', default=1, type=float, metavar='%',
                    help='fraction (0 < n < 1) of tile library to use in training')

import os
import utils
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models
from torch.cuda.amp import autocast

from dataloaders import TileLoader
from pdb import set_trace
import torch.nn as nn
from sklearn.manifold import TSNE
from openslide import OpenSlide

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def main():
    args = parser.parse_args()
    gpu = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    args.tile_size=224
    args.device = gpu
    torch.manual_seed(1917)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    data = pd.read_csv(args.library)
    # data = torch.load('/lila/data/fuchs/hassan/cholangio/icc_library10x.pth', encoding='latin1')['library']
    # data = data.astype({"SlideID": int})
    # test_library = data[data.Split == 'val'].sample(frac=args.subsample).reset_index(drop=True)
    test_library = data.sample(frac=args.subsample).reset_index(drop=True)

    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = TileLoader(test_library, data_transforms, args.data)

    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)


    model = models.resnet50(pretrained=False, num_classes=args.out_dim).to(args.device)
    #model = models.resnet50(pretrained=True, num_classes=1000).to(args.device) #pretrained resnet for comparison

    #checkpoint = torch.load('/lila/data/fuchs/hassan/clr/res50/checkpoint_0100.pth.tar', map_location=args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):

        if k.startswith('module.backbone.'):
            if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
                state_dict[k[len("module.backbone."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.fc = Identity()
    model = nn.DataParallel(model)

    dataset_embedding = torch.zeros(len(valid_loader.dataset), 2048).to(gpu)
    model.eval()
    with torch.no_grad():

        lossMeter = utils.AverageMeter()
        accMeter = utils.AverageMeter()
        n_iter = 0
        for images, _ in valid_loader:
            images = images.to(gpu)

            with autocast(enabled=args.fp16_precision):
                features = model(images).squeeze()
                dataset_embedding[(n_iter * args.batch_size):(n_iter * args.batch_size + len(images))] = features

            print('Train: [{0}][{1}/{2}]'.format(
                0, n_iter+1, len(valid_loader)))

            n_iter += 1

    ########save dataset_embedding to export tiles features

    #tSNE experiment

    samples = torch.randperm(len(valid_loader.dataset))[:500]
    list_of_images = [None] * 500
    data_path = '/scratch/cholangio/'
    for i in range(0, len(list_of_images)):

        n = samples[i].item()
        svs = OpenSlide(os.path.join(data_path, str(test_library.SlideID.iloc[n]) + '.svs'))
        patch = svs.read_region([test_library.x.iloc[n], test_library.y.iloc[n]], 0, [224, 224])
        list_of_images[i] = patch

    tsne = TSNE(2, verbose=1, perplexity=40)

    tsne_proj = tsne.fit_transform(dataset_embedding[samples].cpu().numpy())

    print('tsne done')

    ax = plt.gca()
    ax.clear()
    ax.set_xlim(-6720, 6720)
    ax.set_ylim(-6720, 6720)

    for i in range(500):
        img = list_of_images[i]
        spread = 200
        tx, ty = tsne_proj[i] * spread
        limits = 122
        ax.imshow(img, extent=(tx - limits, tx + limits, ty - limits, ty + limits))
    plt.show()



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x




if __name__ == "__main__":
    main()
