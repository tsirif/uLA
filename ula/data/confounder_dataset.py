import os
from typing import Optional, Callable, Dict

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class Subset(Dataset):
    """
    Subsets a dataset while preserving original indexing.

    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.confounder_array = self.get_confounder_array(re_evaluate=True)
        self.label_array = self.get_label_array(re_evaluate=True)
        self.targets = self.get_targets(re_evaluate=True)

    def __getitem__(self, idx):
        return idx, *self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_confounder_array(self, re_evaluate=True):
        # setting re_evaluate=False helps us over-write the confounder array if necessary (2-group DRO)
        if re_evaluate:
            confounder_array = self.dataset.get_confounder_array()[self.indices]
            assert len(confounder_array) == len(self)
            return confounder_array
        else:
            return self.confounder_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array

    def get_targets(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_targets()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.targets


class ConfounderDataset(Dataset):
    data_dir: str
    filename_array: np.ndarray
    y_array: np.ndarray
    confounder_array: np.ndarray
    targets: np.ndarray
    transform: Optional[Callable] = None
    split_array: np.ndarray
    split_dict: Dict[str, int]

    def __init__(
        self,
        root_dir,
    ):
        raise NotImplementedError

    def get_confounder_array(self):
        return self.confounder_array

    def get_label_array(self):
        return self.y_array

    def get_targets(self):
        return self.targets

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        c = self.confounder_array[idx]

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, y, c

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "valid",
                             "test"), f"{split} is not a valid split"
            mask = self.split_array == self.split_dict[split]

            #  num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(
                    np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name
