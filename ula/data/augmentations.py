import random
from typing import Any, Tuple, Union, Sequence

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from ula.data import DATASET_CLASSES


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization(nn.Module):
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)


class ImagenetTransform(BaseTransform):
    def __init__(
        self,
        crop_size: int,
        mean: Sequence[int], std: Sequence[int],
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        sap_noise_prob: float = 0.0,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """
        super().__init__()
        if not isinstance(crop_size, (list, tuple)):
            crop_size = (crop_size, crop_size)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class CelebATransform(BaseTransform):
    def __init__(
        self,
        crop_size: int,
        mean: Sequence[int], std: Sequence[int],
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        sap_noise_prob: float = 0.0,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """
        super().__init__()
        if not isinstance(crop_size, (list, tuple)):
            crop_size = (crop_size, crop_size)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class NumpyTransform(BaseTransform):
    def __init__(
        self,
        crop_size: int,
        mean: Sequence[int], std: Sequence[int],
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        sap_noise_prob: float = 0.0,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """
        super().__init__()
        if not isinstance(crop_size, (list, tuple)):
            crop_size = (crop_size, crop_size)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                     crop_size,
                     scale=(min_scale, max_scale),
                     interpolation=transforms.InterpolationMode.BICUBIC,
                 ),
                 transforms.RandomApply(
                     [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                     p=color_jitter_prob,
                 ),
                 transforms.RandomGrayscale(p=gray_scale_prob),
                 transforms.RandomApply([transforms.GaussianBlur(7)], p=gaussian_prob),
                 transforms.RandomSolarize(215, p=solarization_prob),
                 transforms.RandomEqualize(p=equalization_prob),
                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                 transforms.ConvertImageDtype(torch.float32),
                 transforms.Normalize(mean=mean, std=std)
            ]
        )


class ColoredMNISTTransform(BaseTransform):
    def __init__(
        self,
        crop_size: int,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        sap_noise_prob: float = 0.0,
        gaussian_prob: float = 0.5,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(crop_size, (list, tuple)):
            crop_size = (crop_size, crop_size)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.ToTensor()
            ]
        )


def prepare_strong_transform(dataset: str, **kwargs) -> Any:
    """Prepares transforms for a specific dataset. Optionally uses multi crop.

    Args:
        dataset (str): name of the dataset.

    Returns:
        Any: a transformation for a specific dataset.
    """

    if dataset == "colored_mnist":
        return ColoredMNISTTransform(**kwargs)
    elif dataset == "celeba":
        return CelebATransform(**kwargs)
    elif dataset in list(DATASET_CLASSES.keys()):
        return ImagenetTransform(**kwargs)
    else:
        raise ValueError(f"{dataset} is not currently supported.")


def prepare_minimal_transform(dataset: str, split: str,
        crop_size: Union[Sequence[int], int] = 224,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
        **kwargs
        ) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """
    if not isinstance(crop_size, (list, tuple)):
        crop_size = (crop_size, crop_size)

    if split == 'test':
        split = 'valid'

    mpi3d_pipeline = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=crop_size,
                    scale=(0.7, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    colored_mnist_pipeline = {
        "train": transforms.ToTensor(),
        "valid": transforms.ToTensor(),
    }

    corrupted_cifar10_pipeline = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    celeba_pipeline = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    cub_pipeline = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize([int(x * (256./224)) for x in crop_size]),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    pipelines = {
        "mpi3d": mpi3d_pipeline,
        "colored_mnist": colored_mnist_pipeline,
        "corrupted_cifar10": corrupted_cifar10_pipeline,
        "celeba": celeba_pipeline,
        "waterbirds": cub_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    return pipeline[split]


def prepare_train_transform(dataset: str, augment: str = 'minimal', **transform_kwargs):
    if augment == 'none':
        T_train = prepare_minimal_transform(dataset, split='valid', **transform_kwargs)
    elif augment == 'minimal':
        T_train = prepare_minimal_transform(dataset, split='train', **transform_kwargs)
    elif augment == 'strong':
        T_train = prepare_strong_transform(dataset, **transform_kwargs)
    elif augment == 'auto':
        T_train = create_transform(
            input_size=transform_kwargs.get('crop_size', 224),
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m7-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.10,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=transform_kwargs.get('mean', IMAGENET_DEFAULT_MEAN),
            std=transform_kwargs.get('std', IMAGENET_DEFAULT_STD),
        )
    else:
        raise ValueError(f'Unknown augmentation strategy: {augment}')
    return T_train
