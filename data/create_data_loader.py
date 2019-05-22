import torch.utils.data


def create_data_loader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader

def create_dataset(opt):
    dataset = None
    if opt.dataset_name == 'cityscapes':
        from .cityscapes_dataset import CityscapesDataset
        dataset = CityscapesDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % (opt.dataset_name))

    dataset.initialize(opt)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset

class CustomDatasetDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.dataset = create_dataset(opt)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=not opt.no_shuffle, drop_last=True)

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return len(self.dataset)
