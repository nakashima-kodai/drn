import os
import random
import torch
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class UnalignedDataset(BaseDataset):
    def name(self):
        return 'UnalinedDataset'

    def initialize(self, opt):
        self.opt = opt
        phase = 'train' if opt.isTrain else 'test'

        dir_A = os.path.join(opt.dataroot, phase + 'A')
        dir_B = os.path.join(opt.dataroot, phase + 'B')

        self.paths_A = make_dataset(dir_A)
        self.paths_B = make_dataset(dir_B)

        self.dataset_size = min(len(self.paths_A), len(self.paths_B))

        self.shuffle_indices()

    def shuffle_indices(self):
        self.indices_A = list(range(len(self.paths_A)))
        self.indices_B = list(range(len(self.paths_B)))
        random.shuffle(self.indices_A)
        random.shuffle(self.indices_B)

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_indices()

        path_A = self.paths_A[self.indices_A[index]]
        image_A = Image.open(path_A).convert('RGB')
        params = get_params(self.opt, image_A.size)
        transform = get_transform(self.opt, params)
        image_A = transform(image_A)

        path_B = self.paths_B[self.indices_B[index]]
        image_B = Image.open(path_B).convert('RGB')
        image_B = transform(image_B)

        input_dict = {'image_A': image_A, 'image_B': image_B}
        return input_dict

    def __len__(self):
        return self.dataset_size
