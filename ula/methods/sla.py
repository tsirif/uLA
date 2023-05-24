import logging
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import (
        ExponentialLR, MultiStepLR, ReduceLROnPlateau, LinearLR)
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ula.utils.misc import calculate_empirical_prior
from ula.utils.metrics import (accuracy_at_k, weighted_mean, bias_super_balanced_accuracy)


class MLP(nn.Module):
    def __init__(
            self, in_dim, h_dim: int=256):
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


class SupervisedLogitAdjustment(pl.LightningModule):
    BIAS_IDX = 0
    TARGET_IDX = -1

    _BACKBONES = {
        "mlp": MLP,
        "resnet18": torchvision.models.resnet18,
        "resnet50": torchvision.models.resnet50,
        "pre_resnet50": torchvision.models.resnet50,
    }
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "warmup",
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        train_dataset,
        task: Sequence[int],
        backbone: str,
        backbone_args: Dict,
        class_loss_func: Callable,
        num_classes: Sequence[int],
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        extra_optimizer_args: Dict,
        scheduler: str,
        lr: float,
        weight_decay: float,
        temperature: Sequence[float],
        min_lr: float = 0.0,
        warmup_start_lr: float = 3e-5,
        warmup_epochs: float = 0,
        total_max_epochs: Optional[int] = None,
        scheduler_interval: str = "step",
        lr_decay_steps: Optional[Sequence[int]] = None,
        no_channel_last: bool = False,
        **kwargs,
    ):
        super().__init__()

        # training related
        self.class_loss_func = class_loss_func or nn.CrossEntropyLoss(ignore_index=-1)

        self.max_epochs = max_epochs
        self.total_max_epochs = total_max_epochs or max_epochs
        self.optimizer = optimizer
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last
        self.τ = temperature

        # all the other parameters
        self.extra_args = kwargs

        # Task related variables and args
        self.num_classes = torch.tensor(num_classes)
        self.task = torch.tensor(task)
        self.target_variable = task[self.TARGET_IDX]
        self.bias_variable = task[self.BIAS_IDX]
        self.num_classes = self.num_classes[self.task]
        self.task_classes = int(self.num_classes[self.TARGET_IDX])

        # Train dataloader info
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        # Hyperparameters needed to create the classifier
        assert backbone in SupervisedLogitAdjustment._BACKBONES
        self.base_model = self._BACKBONES[backbone]
        self.backbone_name = backbone
        self.backbone_args = backbone_args

        # Build classification network
        self.classifier = self.setup_classifier()

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

    def setup_classifier(self) -> nn.Module:
        kwargs = self.backbone_args.copy()
        img_size = kwargs.pop('img_size', 224)
        img_channels = kwargs.pop('img_channels', 3)

        if self.backbone_name == 'mlp':
            self.features_dim = 100 
            input_dim = img_channels * img_size ** 2
            backbone = self.base_model(input_dim, h_dim=self.features_dim)

        elif self.backbone_name.startswith("resnet"):
            backbone = self.base_model(**kwargs)
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

        elif self.backbone_name == 'pre_resnet50':
            backbone = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.features_dim = backbone.inplanes
            backbone.fc = nn.Identity()
            backbone.requires_grad_(True)

        else:
            raise ValueError(f'Unsupported backbone: {self.backbone_name}')

        # Build network
        classifier = nn.Sequential(
            backbone,
            nn.Linear(self.features_dim, self.task_classes),
            )

        # can provide up to ~20% speed up
        if not self.no_channel_last:
            classifier = classifier.to(memory_format=torch.channels_last)

        classifier = classifier.to(self.device)
        return classifier

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")
        # backbone args
        BACKBONES = SupervisedLogitAdjustment._BACKBONES

        parser.add_argument("--backbone", choices=BACKBONES, type=str)

        # if we want to finetune the backbone
        parser.add_argument("--task", type=int, nargs='+', default=0)

        parser.add_argument("--temperature", type=float, default=1.)

        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--optimizer", choices=SupervisedLogitAdjustment._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        # lars args
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=0.002, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")
        # adamw args
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=SupervisedLogitAdjustment._SCHEDULERS, type=str, default="none"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=0, type=int)
        parser.add_argument("--total_max_epochs", type=int, default=None)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
        )

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        learnable_params = [
            {
                "name": "classifier",
                "params": filter(lambda p: p.requires_grad, self.classifier.parameters()),
                "weight_decay": self.weight_decay,
            },
        ]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
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

        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)

        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.5)

        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)

        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup, warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self,
            X: torch.tensor,
            targets: Optional[List[torch.Tensor]]=None) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        if not self.no_channel_last and X.ndim == 4:
            X = X.to(memory_format=torch.channels_last)

        logits = self.classifier(X)

        # During training adjust the logits
        if self.training:
            assert targets is not None
            prior = self.task_prior
            cond = prior / prior.sum(dim=-1, keepdim=True)
            cond = cond[targets[:self.TARGET_IDX]]
            logits = logits + self.τ * torch.log(cond).detach()

        return dict(logits=logits)

    def classify_(self, task_idx, logit, target, num_classes, prefix=None):
        loss = self.class_loss_func(logit, target)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, num_classes)
        acc1, acc5 = accuracy_at_k(logit, target, top_k=(1, top_k_max))
        metrics = {f"class_loss_{task_idx}": loss, f"acc1_{task_idx}": acc1, f"acc5_{task_idx}": acc5}
        if prefix:
            metrics = {prefix + '/' + k: v for k, v in metrics.items()}
        return loss, metrics

    def get_metrics(self,
            logits: torch.Tensor,
            targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (List[torch.Tensor]): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """
        out = dict()
        loss, metrics = self.classify_(
            self.target_variable, logits, targets, self.task_classes)
        out.update(metrics)
        out['acc1'] = out[f'acc1_{self.target_variable}']
        out['loss'] = loss

        return out

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """
        _, X, *targets = batch
        targets = [targets[i] for i in self.task]

        outs = self(X, targets)
        logits = outs['logits']

        out = self.get_metrics(logits, targets[self.TARGET_IDX])

        loss = out['loss']
        metrics = {'train/' + k: v for k, v in out.items()}
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Caught nan or infinity..')
        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """
        _, X, *targets = batch
        targets = [targets[i] for i in self.task]

        outs = self(X)
        logits = outs['logits']

        # Get (unbalanced) validation metrics
        out = self.get_metrics(logits, targets[self.TARGET_IDX])

        # Prepare data for bias-supervised balanced validation
        correct = (logits.argmax(-1) == targets[self.TARGET_IDX]).float().squeeze()
        out['unbiased/correct'] = correct

        metrics = {f'val_{dataloader_idx}/' + k: v for k, v in out.items()}
        metrics['targets'] = torch.stack(targets, dim=-1)
        metrics['batch_size'] = X.size(0)

        return metrics

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Dict[str, Any]:
        _, X, *targets = batch
        targets = [targets[i] for i in self.task]

        outs = self(X)
        logits = outs['logits']

        out = self.get_metrics(logits, targets[self.TARGET_IDX])

        # Prepare data for bias-supervised balanced/worst-case testing
        correct = (logits.argmax(-1) == targets[self.TARGET_IDX]).float().squeeze()
        out['unbiased/correct'] = correct

        metrics = {f'test_{dataloader_idx}/' + k: v for k, v in out.items()}
        metrics['targets'] = torch.stack(targets, dim=-1)
        metrics['batch_size'] = X.size(0)

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

        def skip_key(x):
            if x == 'batch_size':  return True
            if x == 'targets':  return True
            if 'biased' in x:  return True
            return False

        for i, outs_ in enumerate(outs):  # For all validation datasets
            for key in outs_[0].keys():
                if skip_key(key):
                    continue
                metrics[key] = weighted_mean(outs_, key, "batch_size")

            # bias-supervised balanced iid validation
            targets = torch.cat([outs__['targets'] for outs__ in outs_])
            unbiased_correct_preds = torch.cat([outs__[f'val_{i}/unbiased/correct'] for outs__ in outs_])
            s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                unbiased_correct_preds, targets)
            metrics[f"val_{i}/s_balanced/acc1"] = 100. * s_balanced_acc
            metrics[f"val_{i}/s_worst/acc1"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)

    def test_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        metrics = dict()
        if not isinstance(outs[0], list):
            outs = [outs,]

        def skip_key(x):
            if x == 'batch_size':  return True
            if x == 'targets':  return True
            if 'biased' in x:  return True
            return False

        for i, outs_ in enumerate(outs):  # For all test datasets
            for key in outs_[0].keys():
                if skip_key(key):
                    continue
                metrics[key] = weighted_mean(outs_, key, "batch_size")

            # bias-supervised balanced iid validation
            targets = torch.cat([outs__['targets'] for outs__ in outs_])
            unbiased_correct_preds = torch.cat([outs__[f'test_{i}/unbiased/correct'] for outs__ in outs_])
            metrics[f"test_{i}/acc1"] = 100. * unbiased_correct_preds.mean()
            s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                unbiased_correct_preds, targets)
            metrics[f"test_{i}/s_balanced/acc1"] = 100. * s_balanced_acc
            metrics[f"test_{i}/s_worst/acc1"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)

