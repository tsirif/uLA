import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset)

from ula.data.utils import dataset_with_index
from ula.data.augmentations import (prepare_train_transform, prepare_minimal_transform)
from ula.data.mpi3d import MPI3D
from ula.data.biased_datasets import (ColoredMNIST, CorruptedCIFAR10)
from ula.data.celeba import CelebADataset
from ula.data.cub import CUBDataset


def prepare_datasets(
    dataset: str,
    split: str,
    T: Callable,
    data_path: Optional[List[Union[str, Path]]] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        split (str): 'train' or 'valid' or 'test'.
        T (Callable): pipeline of transformations for dataset.
        data_path (Optional[Union[str, Path]], optional): path where the
            data is located. Defaults to None.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_path = sandbox_folder / "datasets"

    if not isinstance(data_path, list):
        data_path = [data_path]

    if dataset == 'mpi3d':
        data_path = data_path[0]
        dataset = dataset_with_index(MPI3D)(data_path, transform=T)

    elif dataset == 'colored_mnist':
        dataset = dataset_with_index(ColoredMNIST)(data_path, transform=T)

    elif dataset == 'corrupted_cifar10':
        dataset = dataset_with_index(CorruptedCIFAR10)(data_path, transform=T)

    elif dataset == 'celeba':
        data_path = data_path[0]
        full_dataset = CelebADataset(data_path, transform=T)
        dataset = full_dataset.get_splits([split])[split]

    elif dataset == 'waterbirds':
        data_path = data_path[0]
        full_dataset = CUBDataset(data_path, transform=T)
        dataset = full_dataset.get_splits([split])[split]

    return dataset


def prepare_val_data(
    dataset: str,
    valid_data_path: Optional[Union[str, Path]] = None,
    split: str = 'valid',
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    valid_noise_prob: float=0.,
    **transform_kwargs
) -> DataLoader:
    T_valid = prepare_minimal_transform(dataset, split,
        valid_noise_prob=valid_noise_prob, **transform_kwargs)
    val_dataset = prepare_datasets(
        dataset, split,
        T=T_valid,
        data_path=valid_data_path,
        download=download
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    return val_loader, val_dataset


def prepare_train_data(
    dataset: str,
    train_data_path: Optional[List[Union[str, Path]]] = None,
    split: str = 'train',
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    augment: str = 'minimal',
    **transform_kwargs
) -> Tuple[DataLoader, Optional[torch.Tensor]]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """
    T_train = prepare_train_transform(dataset, augment=augment, **transform_kwargs)

    splits = split.split('+')
    assert(splits[0] == 'train')
    train_dataset = prepare_datasets(
        dataset, 'train',
        T=T_train,
        data_path=train_data_path,
        download=download,
    )
    if len(splits) > 1:
        train_dataset = ConcatDataset([train_dataset] + [
            prepare_datasets(
                dataset, s,
                T=T_train,
                data_path=train_data_path,
                download=download,
            ) for s in splits[1:]
        ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, train_dataset
