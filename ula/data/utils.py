from typing import Type
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class AttributeDataset(Dataset):
    def __init__(self, data_path: Path,
                 transform=None,
                 query_attr_idx=None):
        super(AttributeDataset, self).__init__()
        dataset = np.load(data_path)
        self.data = dataset['imgs']
        self.attr = torch.LongTensor(dataset['latents_classes'])

        attr_names_path = data_path.parent / "attr_names.pkl"
        with open(str(attr_names_path), "rb") as f:
            self.attr_names = pickle.load(f)

        self.num_attrs =  self.attr.size(1)
        self.set_query_attr_idx(query_attr_idx)
        self.transform = transform

    def set_query_attr_idx(self, query_attr_idx):
        if query_attr_idx is None:
            query_attr_idx = torch.arange(self.num_attrs)

        self.query_attr = self.attr[:, query_attr_idx]

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.query_attr[index]
        if self.transform is not None:
            image = self.transform(image)

        return (image, *list(attr))


def make_attr_labels(target_labels, bias_aligned_ratio, num_classes=10):
    num_samples_per_class = np.array(
        [
            torch.sum(target_labels == label).item()
            for label in range(num_classes)
        ]
    )
    ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + (
        1 - bias_aligned_ratio
    ) / (num_classes - 1) * (1 - np.eye(num_classes))

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis]
        * np.cumsum(ratios_per_class, axis=1)
    ).round()
    num_corruptions_per_class = np.concatenate(
        [
            corruption_milestones_per_class[:, 0, np.newaxis],
            np.diff(corruption_milestones_per_class, axis=1),
        ],
        axis=1,
    )

    attr_labels = torch.zeros_like(target_labels)
    for label in range(num_classes):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            attr_labels[idx] = np.min(
                np.nonzero(corruption_milestones > corruption_idx)[0]
            ).item()

    return attr_labels
