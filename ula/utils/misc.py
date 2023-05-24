from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.callbacks import Callback


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


def make_contiguous(module):
    """Make the model contigous in order to comply with some distributed strategies.
    https://github.com/lucidrains/DALLE-pytorch/issues/330
    """

    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


class EMA(nn.Module):
    def __init__(self, init_value, momentum=0.998):
        super().__init__()
        self.register_buffer('value',
            init_value,
            persistent=True)
        assert momentum <= 1. and momentum >= 0.
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.first = True

    def forward(self, x):
        if self.first:
            self.value.copy_(x.detach())
            self.first = False
            return self.value

        m = self.momentum
        self.value.mul_(m).add_((1 - m) * x.clone().detach())
        return self.value


class Accumulator(nn.Module):
    def __init__(self, shape=(1,)):
        super().__init__()
        self.register_buffer('avg',
            torch.empty(shape),
            persistent=True)
        self.register_buffer('counts',
            torch.empty(shape),
            persistent=True)
        self.reset()

    def reset(self):
        self.avg.fill_(0.)
        self.counts.fill_(0.)

    def forward(self, x):
        self.counts.add_(1.)
        x = x.clone().detach()
        self.avg.mul_((self.counts - 1) / self.counts).add_(x / self.counts)
        return self.avg


class EMACallback(Callback):
    def __init__(self, metric, momentum=0.2):
        super().__init__()
        self.momentum = momentum
        self.metric = metric
        self.ema_metric = None

    @property
    def state_key(self):
        return self._generate_state_key(metric=self.metric)

    def load_state_dict(self, state_dict):
        self.ema_metric = float(state_dict['ema_metric'])

    def state_dict(self):
        return {'ema_metric': self.ema_metric}

    @property
    def value(self):
        return float(self.ema_metric)

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = float(trainer.callback_metrics[self.metric])
        if self.ema_metric is None:
            self.ema_metric = metric
        else:
            self.ema_metric = self.momentum * self.ema_metric + (1 - self.momentum) * metric


def calculate_empirical_prior(
        attributes: torch.Tensor,
        num_classes: Optional[torch.Tensor]=None) -> torch.Tensor:
    if num_classes is None:
        num_classes = attributes.max(dim=0)[0] + 1
    base = torch.cat([torch.tensor([1]), num_classes.flip((0,))[:-1]]).cumprod(0).flip((0,))
    base = base.to(attributes.device)
    idx, counts = attributes.mul(base).sum(-1).unique(sorted=False, return_counts=True)
    prior = attributes.new_zeros(list(num_classes), dtype=torch.float32).flatten()
    prior.index_add_(dim=0,
        index=idx, source=counts.float(),
        alpha=1. / attributes.size(0))
    prior = prior.reshape(*num_classes)
    return prior
