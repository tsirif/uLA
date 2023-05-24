import os
import pandas as pd
import numpy as np

from ula.data.confounder_dataset import ConfounderDataset


def get_anno(filename):
    with open(filename) as f:
        f.readline()
        columns = ['image_id'] + f.readline().split()
        lines = [line.split() for line in f]
        return pd.DataFrame.from_dict(dict(zip(columns, zip(*lines))))


def get_splits(filename):
    with open(filename) as f:
        columns = ['image_id', 'split']
        lines = [line.split() for line in f]
        return pd.DataFrame.from_dict(dict(zip(columns, zip(*lines))))


class CelebADataset(ConfounderDataset):
    NUM_CLASSES = (2, 2)
    NUM_CHANNELS = 3
    IMG_SIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """
    def __init__(
        self,
        root_dir,
        transform=None,
    ):
        self.root_dir = root_dir
        self.target_name = "Blond_Hair"
        self.confounder_names = ["Male",]
        self.transform = transform

        # Read in attributes
        self.attrs_df = get_anno(
            os.path.join(self.root_dir, "list_attr_celeba.txt"))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, "img_align_celeba")
        self.filename_array = self.attrs_df["image_id"].values
        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df = self.attrs_df.astype(int)
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        self.confounder_array = np.matmul(
            confounders.astype(np.int),
            np.power(2, np.arange(len(self.confounder_idx))))

        # Compatibility with rest of code
        self.targets = np.stack([self.y_array, self.confounder_array], axis=-1)

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype("int")

        # Read in train/val/test splits
        self.split_df = get_splits(
            os.path.join(self.root_dir, "list_eval_partition.txt"))
        self.split_array = self.split_df["split"].values
        self.split_array = self.split_array.astype(int)
        self.split_dict = {
            "train": 0,
            "valid": 1,
            "test": 2,
        }

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)
