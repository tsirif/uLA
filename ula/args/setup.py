import argparse
from pathlib import Path

import pytorch_lightning as pl
from ula.args.dataset import (
    augmentations_args,
    custom_dataset_args,
    dataset_args,
    model_selection_args,
)
from ula.args.utils import (
    additional_setup_train,
    additional_setup_pretrain,
    additional_setup_dataset,
    additional_setup_architecture
)
from ula.methods import METHODS
from ula.methods import SUPERVISED_METHODS


def parse_args_pretrain() -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add a seed
    parser.add_argument("--seed", type=int, default=-1,
        help="If -1, then a system random number is used between 0, 2**32-1.")
    parser.add_argument("--num_workers", type=int, default=4)

    # add model selection metric to minimize in hparam search
    model_selection_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str, choices=list(METHODS.keys()))

    # wandb
    parser.add_argument("--name", type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", default=Path("pretrained_models"), type=Path)
    parser.add_argument("--checkpoint_frequency", default=1, type=int)
    parser.add_argument("--auto_resume", action="store_true")

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    augmentations_args(parser, temp_args.dataset)
    custom_dataset_args(parser, temp_args.dataset)

    # parse args
    args = parser.parse_args()
    additional_setup_dataset(args)
    additional_setup_architecture(args)
    additional_setup_pretrain(args)

    return args


def parse_args_train() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add method-specific arguments
    parser.add_argument("--method", type=str, default='ula',
        choices=list(SUPERVISED_METHODS.keys()))

    # general train
    parser.add_argument("--seed", type=int, default=-1,
        help="If -1, then a system random number is used between 0, 2**32-1.")
    parser.add_argument("--num_workers", type=int, default=4)

    # add model selection metric to minimize in hparam search
    model_selection_args(parser)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # wandb
    parser.add_argument("--name", type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", default=Path("supervised_models"), type=Path)
    parser.add_argument("--checkpoint_frequency", default=1, type=int)
    parser.add_argument("--auto_resume", action="store_true")

    # THIS LINE IS KEY TO PULL WANDB AND SAVE_CHECKPOINT
    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = SUPERVISED_METHODS[temp_args.method].add_model_specific_args(parser)

    augmentations_args(parser, temp_args.dataset)
    custom_dataset_args(parser, temp_args.dataset)

    # parse args
    args = parser.parse_args()
    additional_setup_dataset(args)
    additional_setup_architecture(args)
    additional_setup_train(args)

    return args
