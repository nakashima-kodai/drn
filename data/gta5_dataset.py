import os
import random
import torch
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class GTA5Dataset(BaseDataset):
    def name(self):
        return 'GTA5Dataset'

    def initialize(self, opt):
        self.opt = opt

        dir_image = os.path.join(opt.dataroot, 'images')
        dir_label = os.path.join(opt.dataroot, 'labels')
        dir_color = os.path.join(opt.dataroot, 'colors')

        self.paths_image = sorted(make_dataset(dir_image))
        self.paths_label = sorted(make_dataset(dir_label))
        self.paths_color = sorted(make_dataset(dir_color))

        self.dataset_size = len(self.paths_image)

    def __getitem__(self, index):
        # load image
        path_image = self.paths_image[index]
        image = Image.open(path_image).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image = transform_image(image)

        # load label
        path_label = self.paths_label[index]
        label = Image.open(path_label).convert('L')  # gray scale
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label = transform_label(label) * 255.0  # [0, 1] -> [0, 255]

        # load color label
        path_color = self.paths_color[index]
        color = Image.open(path_color).convert('RGB')
        color = transform_label(color)

        input_dict = {'image':image, 'label':label, 'color':color}
        return input_dict

    def __len__(self):
        return self.dataset_size
