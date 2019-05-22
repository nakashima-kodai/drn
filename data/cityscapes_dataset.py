import os
import random
import torch
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class CityscapesDataset(BaseDataset):
    def name(self):
        return 'CityscapesDataset'

    def initialize(self, opt):
        self.opt = opt

        dir_image = os.path.join(opt.dataroot, opt.phase+'_img')
        dir_label = os.path.join(opt.dataroot, opt.phase+'_label')

        self.paths_image = sorted(make_dataset(dir_image))
        self.paths_label = sorted(make_dataset(dir_label))

        self.dataset_size = len(self.paths_image)

    def __getitem__(self, index):
        path_image = self.paths_image[index]
        image = Image.open(path_image).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image = transform_image(image)

        path_label = self.paths_label[index]
        label = Image.open(path_label)
        label = self.id2trainid(label)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label = transform_label(label) * 255

        input_dict = {'image':image, 'label':label}
        return input_dict

    def __len__(self):
        return self.dataset_size
