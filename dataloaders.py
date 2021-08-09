import os
import torch
import random
import numpy as np
import torch.utils.data
from pdb import set_trace
from openslide import OpenSlide
from itertools import permutations

class TileLoader(torch.utils.data.Dataset):

    def __init__(self, library, transform, slide_dir):
        data_path = slide_dir
        unique_ids = library.SlideID.unique()
        print('Loading {} Slides'.format(len(unique_ids)))
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name) + '.svs')))
        self.data = list(zip(library["SlideID"], library["x"], library["y"], library.index.values))
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.len = len(library)
        self.order_set = list(permutations([0,1,2]))


    def __getitem__(self, index):
        im_id, x, y, local_ind = self.data[index]
        img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 0, [224, 224])
        img = self.transform(img.convert('RGB'))

        return img, 0

    def __len__(self):
        return self.len



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


