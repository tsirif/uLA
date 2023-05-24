import os
import pandas as pd
import numpy as np

from ula.data.confounder_dataset import ConfounderDataset


class CUBDataset(ConfounderDataset):
    NUM_CLASSES = (2, 2)
    NUM_CHANNELS = 3
    IMG_SIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    """
    CUB dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(
        self,
        data_dir,
        transform=None,
        metadata_csv_name="metadata.csv"
    ):
        self.data_dir = data_dir
        self.target_name = "y"
        self.confounder_names = ["place",]
        self.transform = transform

        # Read in metadata
        print(f"Reading '{os.path.join(self.data_dir, metadata_csv_name)}'")
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, metadata_csv_name))

        # Get the y values
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 2

        # Compatibility with rest of code
        self.targets = np.stack([self.y_array, self.confounder_array], axis=-1)

        # Map to groups
        self.n_groups = 2 ** 2
        assert self.n_groups == 4, "check the code if you are running otherwise"
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {
            "train": 0,
            "valid": 1,
            "test": 2,
        }
