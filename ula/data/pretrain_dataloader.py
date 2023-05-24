import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Type, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ula.data.utils import dataset_with_index
from ula.data.mpi3d import MPI3D
from ula.data.biased_datasets import (ColoredMNIST, CorruptedCIFAR10)
from ula.data.celeba import CelebADataset
from ula.data.cub import CUBDataset


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    train_data_path: Optional[List[Union[str, Path]]] = None,
    download: bool = True,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = [sandbox_folder / "datasets",]

    if dataset == 'mpi3d':
        assert len(train_data_path) == 1
        train_dataset = dataset_with_index(MPI3D)(train_data_path[0],
                                                  transform=transform)

    elif dataset == 'colored_mnist':
        train_dataset = dataset_with_index(ColoredMNIST)(train_data_path,
            transform=transform)

    elif dataset == 'corrupted_cifar10':
        train_dataset = dataset_with_index(CorruptedCIFAR10)(train_data_path,
            transform=transform)

    elif dataset == 'celeba':
        assert len(train_data_path) == 1
        full_dataset = CelebADataset(train_data_path[0], transform=transform)
        train_dataset = full_dataset.get_splits(['train'])['train']

    elif dataset == 'waterbirds':
        assert len(train_data_path) == 1
        full_dataset = CUBDataset(train_data_path[0], transform=transform)
        train_dataset = full_dataset.get_splits(['train'])['train']

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
