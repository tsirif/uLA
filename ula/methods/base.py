import logging
from copy import deepcopy
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (resnet18, resnet50)
from ula.utils.lars import LARS
from ula.utils.metrics import (accuracy_at_k, weighted_mean, bias_super_balanced_accuracy)
from ula.utils.momentum import MomentumUpdater
from ula.utils.misc import calculate_empirical_prior
from torch.optim.lr_scheduler import (MultiStepLR, LinearLR)
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim: int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x.flatten(1))


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseMethod(pl.LightningModule):
    _BACKBONES = {
        "mlp": MLP,
        "resnet18": resnet18,
        "resnet50": resnet50,
    }
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "warmup",
        "warmup_cosine",
        "step",
        "none",
    ]

    def __init__(
        self,
        train_dataset,
        backbone: str,
        backbone_args: dict,
        num_classes: Sequence[int],
        task: Sequence[int],
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        classifier_wd: float,
        extra_optimizer_args: Dict,
        scheduler: str,
        num_large_crops: int,
        num_small_crops: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.00003,
        warmup_epochs: float = 10,
        total_max_epochs: Optional[int] = None,
        scheduler_interval: str = "epoch",
        lr_decay_steps: Sequence = None,
        no_channel_last: bool = False,
        use_sla: bool = False,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            backbone (str): architecture of the base backbone.
            num_classes (Tuple[int]): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .
            min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
                Defaults to 0.00003.
            warmup_epochs (float): number of warmup epochs. Defaults to 10.
            scheduler_interval (str): interval to update the lr scheduler. Defaults to 'step'.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()

        # resnet backbone related
        self.backbone_args = backbone_args

        # training related
        self.num_classes = torch.tensor(num_classes)
        self.task = torch.tensor(task)
        if len(self.num_classes) > 1:
            self.num_classes = self.num_classes[self.task]
        else:
            self.task = torch.tensor([0,])
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.classifier_wd = classifier_wd
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.total_max_epochs = total_max_epochs or max_epochs
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        self.no_channel_last = no_channel_last
        self.use_sla = use_sla

        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops

        # all the other parameters
        self.extra_args = kwargs

        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0

        # Calculate task prior from training set
        task_prior = calculate_empirical_prior(
            torch.tensor(train_dataset.targets)
            )
        excl = set(range(len(num_classes))) - set(list(self.task.numpy().tolist()))
        excl = list(excl)
        task_prior = task_prior.permute(*self.task, *excl).contiguous()
        if excl:
            excl = range(len(self.task), len(num_classes))
            task_prior = task_prior.sum(dim=tuple(excl))
        # Convert to Bayesian estimate with Dirichlet(1,...,1) prior
        task_prior = (task_prior * len(train_dataset) + 1) / (len(train_dataset) + self.num_classes.prod())
        self.register_buffer('task_prior', task_prior)
        print('Task prior:\n', self.task_prior)

        assert backbone in BaseMethod._BACKBONES
        self.base_model = self._BACKBONES[backbone]
        self.backbone_name = backbone

        self.backbone = self.build_backbone()
        self.classifiers = self.build_classifiers()

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # can provide up to ~20% speed up
        if not no_channel_last:
            self = self.to(memory_format=torch.channels_last)

    def build_backbone(self) -> nn.Module:
        kwargs = self.backbone_args.copy()
        img_size = kwargs.pop('img_size', 224)
        img_channels = kwargs.pop('img_channels', 3)

        method = self.extra_args['method']

        if self.backbone_name == 'mlp':
            self.features_dim = 100
            input_dim = img_channels * img_size ** 2
            backbone = self.base_model(
                input_dim,
                h_dim=self.features_dim)

        elif self.backbone_name.startswith("resnet"):
            backbone = self.base_model(method, **kwargs)
            self.features_dim = backbone.inplanes
            # remove fc layer
            backbone.fc = nn.Identity()
            if img_size == 32:
                backbone.conv1 = nn.Conv2d(
                    img_channels, 64, kernel_size=3, stride=1, padding=2, bias=False)
                backbone.maxpool = nn.Identity()
            elif img_size == 64:
                backbone.conv1 = nn.Conv2d(
                    img_channels, 64, kernel_size=3, stride=1, padding=3, bias=False)
                backbone.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            elif img_size == 224:
                pass
            else:
                raise ValueError(f'Unsupported image size for resnets: {img_size}')

        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        backbone = backbone.to(self.device)
        return backbone

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # backbone args
        BACKBONES = BaseMethod._BACKBONES

        parser.add_argument("--backbone", choices=BACKBONES, type=str)

        parser.add_argument("--task", type=int, nargs='+', default=[0])

        # general train
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--classifier_wd", type=float, default=0.)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--use_sla", type=eval, default=False,
            choices=[True,False])

        parser.add_argument(
            "--optimizer", choices=BaseMethod._OPTIMIZERS.keys(), type=str, required=True
        )
        # lars args
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=0.002, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")
        # adamw args
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=BaseMethod._SCHEDULERS, type=str,
            default="none"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.00003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)
        parser.add_argument("--total_max_epochs", default=None, type=int)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="epoch", type=str
        )

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    def build_classifiers(self) -> Dict[str, List[nn.Module]]:
        classifier = nn.ModuleList([nn.Linear(self.features_dim, nc) for nc in self.num_classes])
        return nn.ModuleDict(dict(base=classifier))

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifiers.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": self.classifier_wd,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.scheduler.lower() == "none":
            return optimizer

        if "warmup" in self.scheduler:
            batches_per_epoch = self.trainer.estimated_stepping_batches / self.max_epochs
            max_warmup_steps = (
                self.warmup_epochs * batches_per_epoch
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.total_max_epochs * batches_per_epoch
                if self.scheduler_interval == "step"
                else self.total_max_epochs
            )
            if self.scheduler == "warmup_cosine":
                scheduler = {
                    "scheduler": LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=max_warmup_steps,
                        max_epochs=max_scheduler_steps,
                        warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                        eta_min=self.min_lr,
                    ),
                    "interval": self.scheduler_interval,
                    "frequency": 1,
                }
            else:
                scheduler = {
                    "scheduler": LinearLR(
                        optimizer,
                        start_factor=self.warmup_start_lr / self.lr,
                        total_iters=max_warmup_steps,
                    ),
                    "interval": self.scheduler_interval,
                    "frequency": 1,
                }
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, none, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """
        This improves performance marginally. It should be fine
        since we are not affected by any of the downsides descrited in
        https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad

        Implemented as in here
        https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#set-grads-to-none
        """
        try:
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()

    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        if not self.no_channel_last and X.ndim == 4:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        feats_ = feats.detach()
        logits = [classifier(feats_) for classifier in self.classifiers['base']]
        return {
            "base_logits": logits,
            "feats": feats,
        }

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Basic multicrop forward method that performs the forward pass
        for the multicrop views. Children classes can override this method to
        add new outputs but should still call this function. Make sure
        that this method and its overrides always return a dict.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        return {"feats": feats}

    def _base_shared_step(self, X: torch.Tensor, targets: List[torch.Tensor]) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (List[torch.Tensor]): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self(X)
        self.EXCLUDE_FROM_SUMMARIZATION = set(out.keys())
        log_prior = self.task_prior.log()

        total_total_loss = 0.
        for logits_key in self.classifiers.keys():
            logits = out[logits_key + '_logits']

            total_loss = 0.
            avg_score = 0.
            volume_score = 1.
            for i, (target, logit, task, nc) in enumerate(zip(targets, logits, self.task, self.num_classes)):
                assert(nc > 1)
                if self.training and self.use_sla:
                    targets_ = targets.copy()
                    targets_[i] = slice(0, self.task_prior.size(i))
                    a = log_prior[targets_]
                    if i == 0:
                        a = a.permute(1, 0).contiguous()
                    logit = logit + F.log_softmax(a, dim=-1)
                loss = F.cross_entropy(logit, target, ignore_index=-1)

                correct = (logit.argmax(-1) == target).float().squeeze()
                out[f'{logits_key}/linear/correct_{task}'] = correct
                self.EXCLUDE_FROM_SUMMARIZATION |= set([f'correct_{task}'])

                # handle when the number of classes is smaller than 5
                top_k_max = min(5, nc)
                acc1, acc5 = accuracy_at_k(logit, target, top_k=(1, top_k_max))
                avg_score += acc1 / len(self.num_classes)
                volume_score *= acc1 ** (1. / len(self.num_classes))
                metrics = {f"class_loss_{task}": loss, f"acc1_{task}": acc1, f"acc5_{task}": acc5}

                total_loss = total_loss + loss
                out.update({logits_key + '/linear/' + k: v for k, v in metrics.items()})

            out[logits_key + '/linear/loss'] = total_loss
            out[logits_key + '/linear/avg_score'] = avg_score
            out[logits_key + '/linear/volume_score'] = volume_score
            total_total_loss = total_total_loss + total_loss

        out['loss'] = total_total_loss

        return out

    def base_training_step(self, X: torch.Tensor, targets: List[torch.Tensor]) -> Dict:
        """Allows user to re-write how the forward step behaves for the training_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets)

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """
        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        outs = [self.base_training_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            multicrop_outs = [self.multicrop_forward(x) for x in X[self.num_large_crops :]]
            for k in multicrop_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        # loss and stats
        metrics = dict()
        for k, v in outs.items():
            if k.split('/')[-1] in self.EXCLUDE_FROM_SUMMARIZATION:
                continue
            outs[k] = sum(v) / self.num_large_crops
            metrics['train/' + k] = outs[k]

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if not bool(torch.isfinite(outs['loss'])):
            raise RuntimeError('Caught nan or infinity..')

        return outs

    def base_validation_step(self, X: torch.Tensor, targets: List[torch.Tensor]) -> Dict:
        """Allows user to re-write how the forward step behaves for the validation_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets)

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """
        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        batch_size = X.size(0)

        outs = self.base_validation_step(X, targets)

        metrics = {
            "batch_size": batch_size,
            f"val_{dataloader_idx}/targets": torch.stack(targets, dim=-1),
        }
        self.EXCLUDE_FROM_SUMMARIZATION |= set(['batch_size', 'targets'])
        for k, v in outs.items():
            metrics[f"val_{dataloader_idx}/" + k] = v
        return metrics

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Dict[str, Any]:

        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        batch_size = X.size(0)

        outs = self.base_validation_step(X, targets)

        metrics = {
            "batch_size": batch_size,
            f"test_{dataloader_idx}/targets": torch.stack(targets, dim=-1),
        }
        self.EXCLUDE_FROM_SUMMARIZATION |= set(['batch_size', 'targets'])
        for k, v in outs.items():
            metrics[f"test_{dataloader_idx}/" + k] = v
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        metrics = dict()
        if not isinstance(outs[0], list):
            outs = [outs,]

        for i, outs_ in enumerate(outs):  # For all validation datasets
            for key in outs_[0].keys():
                if key.split('/')[-1] in self.EXCLUDE_FROM_SUMMARIZATION:
                    continue
                metrics[key] = weighted_mean(outs_, key, "batch_size")

            targets = torch.cat([outs__[f'val_{i}/targets'] for outs__ in outs_])

            # Compute group-balanced and worst-case accuracies given bias-supervision
            if len(self.num_classes) > 1:
                for logits_key in self.classifiers.keys():
                    for j, (task, nc) in enumerate(zip(self.task, self.num_classes)):
                        assert(nc > 1)
                        correct = torch.cat([outs__[f'val_{i}/{logits_key}/linear/correct_{task}'] for outs__ in outs_])
                        s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                            correct, targets)
                        metrics[f"val_{i}/{logits_key}/linear/s_balanced_acc1_{task}"] = 100. * s_balanced_acc
                        metrics[f"val_{i}/{logits_key}/linear/s_worst_acc1_{task}"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)

    def test_epoch_end(self, outs: List[Dict[str, Any]]):

        metrics = dict()
        if not isinstance(outs[0], list):
            outs = [outs,]

        for i, outs_ in enumerate(outs):  # For all validation datasets
            for key in outs_[0].keys():
                if key.split('/')[-1] in self.EXCLUDE_FROM_SUMMARIZATION:
                    continue
                metrics[key] = weighted_mean(outs_, key, "batch_size")

            targets = torch.cat([outs__[f'test_{i}/targets'] for outs__ in outs_])
            if len(self.num_classes) > 1:
                for logits_key in self.classifiers.keys():
                    for j, (task, nc) in enumerate(zip(self.task, self.num_classes)):
                        assert(nc > 1)
                        correct = torch.cat([outs__[f'test_{i}/{logits_key}/linear/correct_{task}'] for outs__ in outs_])
                        s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                            correct, targets)
                        metrics[f"test_{i}/{logits_key}/linear/s_balanced_acc1_{task}"] = 100. * s_balanced_acc
                        metrics[f"test_{i}/{logits_key}/linear/s_worst_acc1_{task}"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)


class BaseMomentumMethod(BaseMethod):
    def __init__(
        self,
        base_tau_momentum: float,
        final_tau_momentum: float,
        momentum_classifier: bool,
        **kwargs,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum backbone and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Args:
            base_tau_momentum (float): base value of the weighting decrease coefficient (should be
                in [0,1]).
            final_tau_momentum (float): final value of the weighting decrease coefficient (should be
                in [0,1]).
            momentum_classifier (bool): whether or not to train a classifier on top of the momentum
                backbone.
        """

        super().__init__(**kwargs)

        # momentum backbone
        kwargs = self.backbone_args.copy()
        self.momentum_backbone = deepcopy(self.backbone)
        self.momentum_backbone.requires_grad_(False)

        # momentum classifier
        if momentum_classifier:
            self.momentum_classifiers = deepcopy(self.classifiers)
        else:
            self.momentum_classifiers = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifiers is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifiers.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": self.classifier_wd,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.backbone, self.momentum_backbone)]

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super().add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", action="store_true")

        return parent_parser

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        out = {'feats': feats}
        if self.momentum_classifiers is not None:
            feats_ = feats.detach()
            logits = [classifier(feats_) for classifier in self.momentum_classifiers['base']]
            out['base_logits'] = logits
        return out

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum backbone and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum backbone / classifier.
        """

        out = self.momentum_forward(X)
        log_prior = self.task_prior.log()

        if self.momentum_classifiers is None:
            return out

        total_total_loss = 0.
        for logits_key in self.momentum_classifiers.keys():
            logits = out[logits_key + '_logits']

            total_loss = 0.
            avg_score = 0.
            volume_score = 1.
            for i, (target, logit, task, nc) in enumerate(zip(targets, logits, self.task, self.num_classes)):
                assert(nc > 1)
                if self.training and self.use_sla:
                    targets_ = targets.copy()
                    targets_[i] = slice(0, self.task_prior.size(i))
                    a = log_prior[targets_]
                    if i == 0:
                        a = a.permute(1, 0).contiguous()
                    logit = logit + F.log_softmax(a, dim=-1)
                loss = F.cross_entropy(logit, target, ignore_index=-1)

                correct = (logit.argmax(-1) == target).float().squeeze()
                out[f'{logits_key}/linear/correct_{task}'] = correct

                # handle when the number of classes is smaller than 5
                top_k_max = min(5, logit.size(1))
                acc1, acc5 = accuracy_at_k(logit, target, top_k=(1, top_k_max))
                avg_score += acc1 / len(self.num_classes)
                volume_score *= acc1
                metrics = {f"class_loss_{task}": loss, f"acc1_{task}": acc1, f"acc5_{task}": acc5}

                total_loss = total_loss + loss
                out.update({logits_key + '/linear/' + k: v for k, v in metrics.items()})

            out[logits_key + '/linear/loss'] = total_loss
            out[logits_key + '/linear/avg_score'] = avg_score
            out[logits_key + '/linear/volume_score'] = volume_score ** (1./len(self.num_classes))
            total_total_loss = total_total_loss + total_loss

        out['loss'] = total_total_loss

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding the crops in the momentum backbone
        and classifier, and computing statistics.
        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum backbone and the classification
                loss and logits of the momentum classifier.
        """

        outs = super().training_step(batch, batch_idx)

        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = {
            "momentum/" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifiers is not None:
            # momentum loss and stats
            metrics = dict()
            for k, v in momentum_outs.items():
                if k.split('/')[-1] in self.EXCLUDE_FROM_SUMMARIZATION:
                    continue
                momentum_outs[k] = sum(v) / self.num_large_crops
                metrics['train/' + k] = momentum_outs[k]

            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum/loss"]

        outs.update(momentum_outs)
        return outs

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            batches_per_epoch = self.trainer.estimated_stepping_batches / self.max_epochs
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=batches_per_epoch * self.total_max_epochs,
            )
        self.last_step = self.trainer.global_step

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.
        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().validation_step(batch, batch_idx, dataloader_idx)

        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        batch_size = X.size(0)

        outs = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_classifiers is not None:
            metrics = {
                "batch_size": batch_size,
                f"val_{dataloader_idx}/targets": torch.stack(targets, dim=-1),
            }
            self.EXCLUDE_FROM_SUMMARIZATION |= set(['batch_size', 'targets'])
            for k, v in outs.items():
                metrics[f"val_{dataloader_idx}/momentum/" + k] = v

        return dict(base=parent_metrics, momentum=metrics)

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.
        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().test_step(batch, batch_idx, dataloader_idx)

        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        batch_size = X.size(0)

        outs = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_classifiers is not None:
            metrics = {
                "batch_size": batch_size,
                f"test_{dataloader_idx}/targets": torch.stack(targets, dim=-1),
            }
            self.EXCLUDE_FROM_SUMMARIZATION |= set(['batch_size', 'targets'])
            for k, v in outs.items():
                metrics[f"test_{dataloader_idx}/momentum/" + k] = v

        return dict(base=parent_metrics, momentum=metrics)

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum backbone / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """
        if not isinstance(outs[0], list):
            outs = [outs,]
        parent_outs = [[out["base"] for out in outs_] for outs_ in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifiers is None:
            return

        momentum_outs = [[out["momentum"] for out in outs_] for outs_ in outs]
        metrics = dict()
        for i, momentum_outs_ in enumerate(momentum_outs):
            for key in momentum_outs_[0].keys():
                if key.split('/')[-1] in self.EXCLUDE_FROM_SUMMARIZATION:
                    continue
                metrics[key] = weighted_mean(momentum_outs_, key, "batch_size")

            targets = torch.cat([outs__[f'val_{i}/targets'] for outs__ in momentum_outs_])
            if len(self.num_classes) > 1:
                for logits_key in self.classifiers.keys():
                    for j, (task, nc) in enumerate(zip(self.task, self.num_classes)):
                        assert(nc > 1)
                        correct = torch.cat([outs__[f'val_{i}/momentum/{logits_key}/linear/correct_{task}'] for outs__ in momentum_outs_])

                        s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                            correct, targets)
                        metrics[f"val_{i}/momentum/{logits_key}/linear/s_balanced_acc1_{task}"] = 100. * s_balanced_acc
                        metrics[f"val_{i}/momentum/{logits_key}/linear/s_worst_acc1_{task}"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)

    def test_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        if not isinstance(outs[0], list):
            outs = [outs,]
        parent_outs = [[out["base"] for out in outs_] for outs_ in outs]
        super().test_epoch_end(parent_outs)

        if self.momentum_classifiers is None:
            return

        momentum_outs = [[out["momentum"] for out in outs_] for outs_ in outs]
        metrics = dict()
        for i, momentum_outs_ in enumerate(momentum_outs):
            for key in momentum_outs_[0].keys():
                if  key.split('/')[-1] in self.EXCLUDE_FROM_SUMMARIZATION:
                    continue
                metrics[key] = weighted_mean(momentum_outs_, key, "batch_size")

            targets = torch.cat([outs__[f'test_{i}/targets'] for outs__ in momentum_outs_])
            if len(self.num_classes) > 1:
                for logits_key in self.classifiers.keys():
                    for j, (task, nc) in enumerate(zip(self.task, self.num_classes)):
                        assert(nc > 1)
                        correct = torch.cat([outs__[f'test_{i}/momentum/{logits_key}/linear/correct_{task}'] for outs__ in momentum_outs_])

                        s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                            correct, targets)
                        metrics[f"test_{i}/momentum/{logits_key}/linear/s_balanced_acc1_{task}"] = 100. * s_balanced_acc
                        metrics[f"test_{i}/momentum/{logits_key}/linear/s_worst_acc1_{task}"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)
