import os
import math
import copy
import json
import logging
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import (
        ExponentialLR, MultiStepLR, ReduceLROnPlateau, LinearLR, LambdaLR)
import torchvision
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ula.utils.misc import (calculate_empirical_prior, EMA)
from ula.utils.metrics import (accuracy_at_k, weighted_mean,
    bias_super_balanced_accuracy, bias_unsuper_balanced_accuracy)


def _str_to_list_float(x):
    try:
        assert x[0] == '['
        assert x[-1] == ']'
    except AssertionError:
        return float(x)
    return list(map(float, x[1:-1].split(',')))


def make_sure_it_is_list(which, length) -> List:
    if not isinstance(which, (list, tuple)):
        which = [which,]
    if len(which) == 1:
        which = which * length
    assert len(which) == length
    return list(which)


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


class UnsupervisedLogitAdjustment(pl.LightningModule):
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
        train_dataset: Dataset,
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
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        min_lr: float = 0.0,
        warmup_start_lr: float = 3e-5,
        warmup_epochs: float = 0,
        total_max_epochs: Optional[int] = None,
        scheduler_interval: str = "step",
        lr_decay_steps: Optional[Sequence[int]] = None,
        no_channel_last: bool = False,
        confusion_mat_ema: float = 0.999,
        use_group_mixup: bool = False,
        train_mode: str = 'scratch',
        gen0_max_steps: int = -1,
        gen0_max_epochs: Optional[int] = None,
        gen0_freeze_backbone: bool = True,
        biased_network_state: Optional[str]=None,
        biased_network_epochs: Optional[int]=None,
        biased_network_args: Optional[str]=None,
        **kwargs,
    ):
        super().__init__()

        # training related
        self.class_loss_func = class_loss_func or nn.CrossEntropyLoss(ignore_index=-1)
        self.max_epochs = max_epochs
        self.total_max_epochs = total_max_epochs or max_epochs
        generation_epochs = torch.tensor(gen0_max_epochs or []).cumsum(0).view(-1)
        self.generation_epochs = torch.cat([generation_epochs, torch.tensor([self.total_max_epochs])])
        print(f"Training {len(self.generation_epochs)} generations, at:",
            self.generation_epochs.numpy())
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.lr = make_sure_it_is_list(lr, len(self.generation_epochs))
        self.weight_decay = make_sure_it_is_list(weight_decay, len(self.generation_epochs))
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last

        # all the other parameters
        self.extra_args = kwargs

        # Task related variables and args
        self.num_classes = torch.tensor(num_classes)
        self.task = torch.tensor(task)
        self.target_variable = task[self.TARGET_IDX]
        self.bias_variable = task[self.BIAS_IDX]
        if len(self.num_classes) > 1:
            self.num_classes = self.num_classes[self.task]
        else:
            self.task = torch.tensor([0,])
        self.task_classes = int(self.num_classes[self.TARGET_IDX])

        train_dataset_N = len(train_dataset)
        self.train_dataset = train_dataset
        self.gen0_max_steps = gen0_max_steps

        self.register_buffer('generalization_gap', torch.tensor([0.]),
            persistent=True)

        # Calculate task prior (used for logging errors only)
        if len(self.num_classes) > 1:
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

        # Student refers to logit adjustment, teacher to bias network
        self.student_τ = student_temperature
        self.teacher_τ = teacher_temperature

        # Architecture related Hyperparameters
        assert backbone in UnsupervisedLogitAdjustment._BACKBONES
        self.base_model = self._BACKBONES[backbone]
        self.backbone_name = backbone
        self.backbone_args = backbone_args

        # Initialize generation and classifiers
        self.train_with_ula = len(self.generation_epochs) == 1
        self.register_buffer('generation',
            torch.tensor(0),
            persistent=True)
        self.confusion_mat_shape = torch.tensor([self.task_classes, self.task_classes])
        self.train_mode = train_mode
        uniform_matrix = torch.ones((self.task_classes, self.task_classes)) / self.task_classes**2
        self.confusion_mat_estimate = EMA(uniform_matrix, momentum=confusion_mat_ema)
        self.use_group_mixup = use_group_mixup

        # Get biased network
        self.biased_network_state = biased_network_state
        if biased_network_epochs:
            self.biased_network_state = self.biased_network_state % biased_network_epochs
        self.biased_network_args = biased_network_args

        if self.biased_network_args is not None:
            self.classifier = self.load_backbone(self.biased_network_args, self.biased_network_state,
                freeze_backbone=gen0_freeze_backbone)
        else:
            self.classifier = self.setup_classifier(
                freeze_backbone=gen0_freeze_backbone)

        self.biased_classifier = copy.deepcopy(self.classifier)

        # If only one generation, then we assume that we train with uLA.
        # In that case, we need to load the biased classifier from a previous uLA.py checkpoint
        if self.train_with_ula:
            # Get biased network
            if self.biased_network_state is not None:
                print('Restoring biased network from:', self.biased_network_state)
                assert(os.path.isfile(self.biased_network_state))
                proc_state = torch.load(self.biased_network_state)['state_dict']
                # Get only keys which start with 'classifier' from `proc_state`
                prefix = 'classifier.'
                biased_network_state = {k.removeprefix(prefix): v for (k, v) in proc_state.items() if k.startswith(prefix)}
                self.biased_classifier.load_state_dict(biased_network_state)
            else:
                print('Setting student temp to 0 (ERM), since no biased_network_state was given.')
                self.student_τ = 0.0

        self.classifier.train()
        self.biased_classifier.eval()
        self.biased_classifier.requires_grad_(False)

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

    def load_backbone(self, args, state, freeze_backbone=False) -> nn.Module:
        print('Restoring biased network from:', state)
        assert(os.path.isfile(args) and os.path.isfile(state))
        with open(args) as f:
            args = dict(json.load(f))
        method = args['method']
        pre_backbone_name = args['backbone']

        from ula.methods import METHODS
        biased_model = METHODS[method](train_dataset=self.train_dataset, **args)
        state = torch.load(state)
        biased_model.load_state_dict(state['state_dict'])
        biased_model.requires_grad_(False)
        biased_model.eval()

        # Initialize classifiers
        assert(hasattr(biased_model, 'backbone'))
        backbone = copy.deepcopy(biased_model.backbone)
        backbone.requires_grad_(False)
        backbone.eval()
        if pre_backbone_name == 'mlp':
            self.features_dim = 100
        elif pre_backbone_name.startswith('resnet'):
            self.features_dim = backbone.inplanes
        else:
            self.features_dim = backbone.num_features
        classifier = nn.Sequential(
            backbone,
            nn.Linear(self.features_dim, self.task_classes),
            )
        if not freeze_backbone:
            classifier.requires_grad_(True)
            classifier.train()

        # can provide up to ~20% speed up
        if not self.no_channel_last:
            classifier = classifier.to(memory_format=torch.channels_last)
        classifier = classifier.to(self.device)

        return classifier

    def setup_classifier(self, freeze_backbone=False) -> nn.Module:
        kwargs = self.backbone_args.copy()
        img_size = kwargs.pop('img_size', 224)
        img_channels = kwargs.pop('img_channels', 3)

        dataset = self.extra_args.get("dataset", None)
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

        else:
            raise ValueError(f'Unsupported backbone: {self.backbone_name}')

        backbone.requires_grad_(freeze_backbone is False)

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

    @torch.no_grad()
    def compute_confusion_matrix(self, pred_y, y):
        y = y.view(-1, 1, 1)  # batch, nc, nc
        pred_y = pred_y.view(-1, self.task_classes, 1)
        nc = torch.arange(self.task_classes, device=y.device)[None, None, :]
        running_estimate = torch.where(y == nc, pred_y, 0.).mean(0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(running_estimate)
            running_estimate /= dist.get_world_size()
        return running_estimate  # y_pred, y

    def on_train_epoch_start(self):
        self.biased_classifier.eval()
        generations_to_process = (self.current_epoch < self.generation_epochs).nonzero()
        next_generation = int(generations_to_process[0, 0])

        self.train_with_ula = self.train_with_ula or int(self.generation) > 0
        # If it is time to train next generation, or if we early stop current one
        condition = next_generation > int(self.generation)
        if not condition:
            return

        self.train_with_ula = True
        if self.gen0_max_steps < 0:
            self.gen0_max_steps = self.global_step
        # Bump up generation number
        self.generation.add_(1)
        print(f"Starting generation {int(self.generation)}.")

        # Make ERM network the bias network, and reset classifier
        with torch.no_grad():
            self.biased_classifier.load_state_dict(self.classifier.state_dict())
            self.biased_classifier.requires_grad_(False)
            self.biased_classifier.eval()
            if self.train_mode in ['finetune', 'head']:
                self.classifier = self.load_backbone(self.biased_network_args, self.biased_network_state,
                    freeze_backbone=(self.train_mode == 'head'))
            elif self.train_mode == 'scratch':
                self.classifier = self.setup_classifier(
                    freeze_backbone=False)
            else:
                raise NotImplementedError

        self.trainer.strategy.setup(self.trainer)
        print('Setup done')

    def on_train_batch_start(self, batch, batch_idx):
        early_stop = False
        if int(self.generation) == 0 and \
                self.gen0_max_steps >= 0 and \
                self.global_step >= self.gen0_max_steps:
            print('max number of gen0 steps have been reached:', self.global_step)
            early_stop = True

        if early_stop:
            return -1

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
        BACKBONES = UnsupervisedLogitAdjustment._BACKBONES

        parser.add_argument("--backbone", choices=BACKBONES, type=str)

        # if we want to finetune the backbone
        parser.add_argument("--task", type=int, nargs='+', default=0)

        parser.add_argument("--biased_network_state", type=str, default=None,
            help="Path to checkpoint with pretrained biased network to bias logits with.")
        parser.add_argument("--biased_network_args", type=str, default=None)
        parser.add_argument("--biased_network_epochs", type=int, default=None)
        parser.add_argument("--train_mode",
            type=str, default='finetune', choices=['scratch', 'finetune', 'head'])

        parser.add_argument("--confusion_mat_ema", type=float, default=0.999)
        parser.add_argument("--use_group_mixup", type=eval, default=False, choices=[True, False])
        parser.add_argument("--student_temperature", type=float, default=1.)
        parser.add_argument("--teacher_temperature", type=float, default=1.)

        parser.add_argument("--gen0_max_epochs", type=int, default=None)
        parser.add_argument("--gen0_max_steps", type=int, default=-1,
            help="Number of training steps to stop training of bias network. -1 is off")
        # Bias-(un)supervised model selection configuration
        parser.add_argument("--gen0_freeze_backbone", type=eval, default=True,
            choices=[True, False])

        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--optimizer", choices=UnsupervisedLogitAdjustment._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--lr", type=_str_to_list_float, default=[0.3])
        parser.add_argument("--weight_decay", type=_str_to_list_float, default=[0.0001])
        # lars args
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=0.002, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")
        # adamw args
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=UnsupervisedLogitAdjustment._SCHEDULERS, type=str, default="none"
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

        generation = int(self.generation)
        wd = self.weight_decay[generation]
        learnable_params = [
            {
                "name": "classifier",
                "params": filter(lambda p: p.requires_grad, self.classifier.parameters()),
                "weight_decay": wd,
            },
        ]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr[generation],
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none" or generation == 0:
            return optimizer

        if "warmup" in self.scheduler:
            batches_per_epoch = self.trainer.estimated_stepping_batches / self.max_epochs
            max_warmup_steps = (
                self.warmup_epochs * batches_per_epoch
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_epochs = self.generation_epochs[self.generation] - self.current_epoch
            max_scheduler_steps = (
                max_epochs * batches_per_epoch
                if self.scheduler_interval == "step"
                else max_epochs
            )
            if self.scheduler == "warmup_cosine":
                scheduler = {
                    "scheduler": LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=max_warmup_steps,
                        max_epochs=max_scheduler_steps,
                        warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr[generation],
                        eta_min=self.min_lr,
                    ),
                    "interval": self.scheduler_interval,
                    "frequency": 1,
                }
            else:
                scheduler = {
                    "scheduler": LinearLR(
                        optimizer,
                        start_factor=self.warmup_start_lr / self.lr[generation],
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

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        if not self.no_channel_last and X.ndim == 4:
            X = X.to(memory_format=torch.channels_last)

        generation = int(self.generation)
        biased_preds = None
        if self.train_with_ula:
            # A. Get conditional for current biased predictions (soft or hard way)
            with torch.no_grad():
                biased_logits = self.biased_classifier(X)
                biased_pyx = F.softmax(biased_logits / self.teacher_τ, dim=-1)
                biased_preds = biased_logits.argmax(-1)

            if self.training:
                assert(Y is not None)
                run_estimate = self.compute_confusion_matrix(biased_pyx, Y)
                confusion_mat = self.confusion_mat_estimate(run_estimate)

                # B. Estimate conditional of true target given biased predictions value
                p = confusion_mat[biased_preds]

                if self.use_group_mixup:
                    # Cosine annealing of tau
                    batches_per_epoch = self.trainer.estimated_stepping_batches / self.max_epochs
                    max_steps = batches_per_epoch * self.generation_epochs[-1] - self.gen0_max_steps
                    tau = min(1.0, 1 - (math.cos(math.pi * (self.global_step - self.gen0_max_steps) / max_steps) + 1) / 2)
                    # further augmenting samples from minority groups
                    X, Y, p = group_mixup(X, Y, biased_preds, p, tau=tau)

                cond = p / p.sum(dim=-1, keepdim=True)
                cond = cond.detach()
                logit_bias = self.student_τ * torch.log(cond) if self.student_τ > 0 else 0.
        else:
            logit_bias = 0.

        logits = self.classifier(X)
        # C. Bias logits of debiasing network
        if self.training:
            logits = logits + logit_bias

        return dict(logits=logits,
            biased_preds=biased_preds,
            Y=Y)

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
        index, X, *targets = batch
        targets = [targets[i] for i in self.task]
        Y = targets[self.TARGET_IDX]

        outs = self(X, Y)
        logits = outs['logits']

        out = self.get_metrics(logits, outs['Y'])

        generation = int(self.generation)
        loss = out['loss']

        metrics = {'train/' + k: v for k, v in out.items()}

        if len(self.num_classes) > 1 and torch.all(self.num_classes == self.confusion_mat_shape) and hasattr(self, 'task_prior'):
            error = self.confusion_mat_estimate.value.sub(self.task_prior).abs()
            diag_error = error.diagonal()
            metrics['train/total_bias_est_error'] = error.sum()
            metrics['train/diag_bias_est_error'] = diag_error.sum()
            metrics['train/off_bias_est_error'] = error.sum() - diag_error.sum()

        metrics['generation'] = float(self.generation)
        with torch.no_grad():
            metrics['train/HYgX'] = - F.softmax(logits, dim=-1).mul(F.log_softmax(logits, dim=-1)).sum(-1).mean()

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
        Y = targets[self.TARGET_IDX]

        outs = self(X)
        logits = outs['logits']
        biased_preds = outs['biased_preds']

        # Get (unbalanced) validation metrics
        out = self.get_metrics(logits, Y)

        # Prepare data for bias-supervised balanced validation
        correct = (logits.argmax(-1) == targets[self.TARGET_IDX]).float().squeeze()
        out['unbiased/correct'] = correct
        out['biased/preds'] = biased_preds

        metrics = {f'val_{dataloader_idx}/' + k: v for k, v in out.items()}
        metrics['targets'] = torch.stack(targets, dim=-1)
        metrics['batch_size'] = X.size(0)

        return metrics

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int=0
    ) -> Dict[str, Any]:
        _, X, *targets = batch
        targets = [targets[i] for i in self.task]
        Y = targets[self.TARGET_IDX]

        outs = self(X)
        logits = outs['logits']

        out = self.get_metrics(logits, Y)

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

            if len(self.num_classes) > 1:
                dataset = self.extra_args.get("dataset", None)
                s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                    unbiased_correct_preds, targets)
                metrics[f"val_{i}/s_balanced/acc1"] = 100. * s_balanced_acc
                metrics[f"val_{i}/s_worst/acc1"] = 100. * s_worst_acc

            if self.train_with_ula:
                # bias-unsupervised balanced iid validation
                biased_preds = torch.cat([outs__[f'val_{i}/biased/preds'] for outs__ in outs_])
                u_balanced_acc, u_worst_acc = bias_unsuper_balanced_accuracy(
                    unbiased_correct_preds, biased_preds, targets[:, self.TARGET_IDX], num_classes=self.confusion_mat_shape)
                metrics[f"val_{i}/u_balanced/acc1"] = 100. * u_balanced_acc
                metrics[f"val_{i}/u_worst/acc1"] = 100. * u_worst_acc
            else:
                # HACK: Even if this metric is not available during generation 0,
                # we need it to report an increasing value so that training
                # does not early stop during generation 0.
                # fake_metric below is always non-positive during generation 0,
                # so that all possible values during generation 1 are greater
                # and simultaneously increasing per epoch so that early stopping
                # does not trigger
                fake_metric = float(self.current_epoch - self.generation_epochs[0]) / self.generation_epochs[0]
                metrics[f"val_{i}/u_balanced/acc1"] = fake_metric
                metrics[f"val_{i}/u_worst/acc1"] = fake_metric

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

        for i, outs_ in enumerate(outs):  # For all validation datasets
            for key in outs_[0].keys():
                if skip_key(key):
                    continue
                metrics[key] = weighted_mean(outs_, key, "batch_size")

            # bias-supervised balanced iid validation
            if len(self.num_classes) > 1:
                targets = torch.cat([outs__['targets'] for outs__ in outs_])
                unbiased_correct_preds = torch.cat([outs__[f'test_{i}/unbiased/correct'] for outs__ in outs_])
                metrics[f"test_{i}/acc"] = 100. * unbiased_correct_preds.mean()

                dataset = self.extra_args.get("dataset", None)
                s_balanced_acc, s_worst_acc = bias_super_balanced_accuracy(
                    unbiased_correct_preds, targets)
                metrics[f"test_{i}/s_balanced/acc1"] = 100. * s_balanced_acc
                metrics[f"test_{i}/s_worst/acc1"] = 100. * s_worst_acc

        self.log_dict(metrics, sync_dist=True)
