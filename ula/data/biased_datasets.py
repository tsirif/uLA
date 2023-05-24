from typing import List, Tuple, Union
from pathlib import Path
import os
import numpy
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class BiasedDataset(Dataset):
    NUM_CLASSES : Tuple[int]

    def __init__(self, datapaths, transform=None):
        super(BiasedDataset, self).__init__()
        self.transform = transform

        if isinstance(datapaths, (str, Path)):
            datapaths = [datapaths]
        if all(str(datapath).endswith("npz") for datapath in datapaths):
            self.data = []
            self.targets = []
            for datapath in datapaths:
                numpy_data = numpy.load(datapath)
                self.data.append(numpy_data["data"])
                self.targets.append(numpy_data["targets"])
            self.data = numpy.concatenate(self.data, axis=0)
            self.targets = numpy.concatenate(self.targets, axis=0)
            return
        print('Loading data from', datapaths)

        self.image_files = []
        for datapath in datapaths:
            self.image_files.extend(glob(os.path.join(datapath, '**', '*.png'), recursive=True))

        assert len(self.image_files) > 0

        data = []
        targets = []
        for i, path in enumerate(tqdm(self.image_files)):
            target_attr = int(path.split('_')[-2])
            bias_attr = int(path.split('_')[-1].split('.')[0])
            data.append(numpy.array(pil_loader(path))[numpy.newaxis, :, :, :])
            targets.append(numpy.array([target_attr, bias_attr])[numpy.newaxis, :])
            if i > 0 and (i % 1000 == 0 or i == len(self.image_files) - 1):
                data = numpy.concatenate(data, axis=0)
                targets = numpy.concatenate(targets, axis=0)
                data = [data]
                targets = [targets]
        self.data = data[0]
        self.targets = targets[0]

        filename = '_'.join([str(datapath).split('/')[-1] for datapath in datapaths])
        filename = os.path.join(os.path.dirname(datapaths[0]), filename + '.npz')
        numpy.savez_compressed(filename, data=self.data, targets=self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, *self.targets[index]


class ColoredMNIST(BiasedDataset):
    NUM_CLASSES = (10, 10)
    NUM_CHANNELS = 3
    IMG_SIZE = 28


class CorruptedCIFAR10(BiasedDataset):
    NUM_CLASSES = (10, 10)
    NUM_CHANNELS = 3
    IMG_SIZE = 32
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
