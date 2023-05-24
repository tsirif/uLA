from argparse import ArgumentParser
from pathlib import Path

from ula.data import (DATASET_CLASSES,
    get_img_size, get_data_mean, get_data_std)


def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    SUPPORTED_DATASETS = list(DATASET_CLASSES.keys())
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)

    # dataset path
    parser.add_argument("--train_data_path", type=Path, nargs='+', required=True)
    parser.add_argument("--train_data_split", default="train", type=str,
        choices=["train", "train+valid"])
    parser.add_argument("--valid_data_path", type=Path, nargs="*")
    parser.add_argument("--test_data_path", type=Path, nargs="*")


def model_selection_args(parser: ArgumentParser):
    """Adds model selection-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add model selection args to.
    """

    parser.add_argument("--model_selection_metric", type=str,
        default=None)
    parser.add_argument("--model_selection_mode", type=str,
        choices=['max', 'min'], default='max')
    parser.add_argument("--hyperopt_metric_momentum", type=float, default=0)
    parser.add_argument("--select_best_model", action="store_true",
        help="Save best model across training epochs according " \
             "to `model_selection_metric` and perform hyperparameter " \
             "search according to it.")
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.1)


def augmentations_args(parser: ArgumentParser, dataset: str):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """
    # train augmentation strategy
    parser.add_argument("--augment", type=str, default="minimal",
        choices=["none", "minimal", "strong", "auto"])

    # cropping
    parser.add_argument("--num_crops_per_aug", type=int, default=[2], nargs="+",
        help="Number of crops used in SSL per unique augmentation config.")

    # color jitter
    parser.add_argument("--brightness", type=float, default=[0.4], nargs="+")
    parser.add_argument("--contrast", type=float, default=[0.4], nargs="+")
    parser.add_argument("--saturation", type=float, default=[0.4], nargs="+")
    parser.add_argument("--hue", type=float, default=[0.1], nargs="+")
    parser.add_argument("--color_jitter_prob", type=float, default=[0.8], nargs="+")

    # other augmentation probabilities
    parser.add_argument("--sap_noise_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--gray_scale_prob", type=float, default=[0.2], nargs="+")
    parser.add_argument("--horizontal_flip_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--equalization_prob", type=float, default=[0.0], nargs="+")

    # cropping
    parser.add_argument("--crop_size", type=int, nargs='+',
        default=[get_img_size(dataset)])
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")
    parser.add_argument("--max_scale", type=float, default=[1.0], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", action="store_true")


def custom_dataset_args(parser: ArgumentParser, dataset: str):
    """Adds custom data-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """
    # for custom dataset
    parser.add_argument("--mean", type=float, nargs='+',
        default=get_data_mean(dataset))
    parser.add_argument("--std", type=float, nargs='+',
        default=get_data_std(dataset))
