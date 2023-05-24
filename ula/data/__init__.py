from typing import Sequence
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ula.data.mpi3d import MPI3D
from ula.data.biased_datasets import (ColoredMNIST, CorruptedCIFAR10)
from ula.data.celeba import CelebADataset
from ula.data.cub import CUBDataset


DATASET_CLASSES = {
    "mpi3d": MPI3D,
    "colored_mnist": ColoredMNIST,
    "corrupted_cifar10": CorruptedCIFAR10,
    "celeba": CelebADataset,
    "waterbirds": CUBDataset,
}


def get_img_size(dataset: str):
    return DATASET_CLASSES[dataset].IMG_SIZE


def get_num_channels(dataset: str):
    return DATASET_CLASSES[dataset].NUM_CHANNELS


def get_num_classes(dataset: str) -> Sequence[int]:
    return DATASET_CLASSES[dataset].NUM_CLASSES


def get_data_mean(dataset: str):
    try:
        return DATASET_CLASSES[dataset].MEAN
    except AttributeError:
        return IMAGENET_DEFAULT_MEAN


def get_data_std(dataset: str):
    try:
        return DATASET_CLASSES[dataset].STD
    except AttributeError:
        return IMAGENET_DEFAULT_STD


from ula.data import classification_dataloader, pretrain_dataloader

__all__ = [
    "classification_dataloader",
    "pretrain_dataloader",
    "DATASET_CLASSES",
]
