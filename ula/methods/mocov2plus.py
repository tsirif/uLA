import argparse
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ula.losses.mocov2plus import mocov2plus_loss_func
from ula.methods.base import BaseMomentumMethod
from ula.utils.misc import gather


class MoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """
        self.proj_output_dim = proj_output_dim
        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = deepcopy(self.projector)
        self.momentum_projector.requires_grad_(False)

        # create the queue
        self.register_buffer("queue", torch.randn(queue_size, 2, proj_output_dim))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def build_classifiers(self) -> Dict[str, List[nn.Module]]:
        classifiers = super().build_classifiers()
        classifier = nn.ModuleList([nn.Linear(self.proj_output_dim, nc) for nc in self.num_classes])
        classifiers['norm_z'] = classifier
        return classifiers

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MoCoV2Plus, MoCoV2Plus).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: List[torch.Tensor]):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, :, :] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and query.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z, "norm_z": F.normalize(z, dim=-1)})
        z_ = out['norm_z'].detach()
        logits = [classifier(z_) for classifier in self.classifiers['norm_z']]
        out['norm_z_logits'] = logits
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the key.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z, "norm_z": F.normalize(z, dim=-1)})
        if self.momentum_classifiers is not None:
            z_ = out['norm_z'].detach()
            logits = [classifier(z_) for classifier in self.momentum_classifiers['norm_z']]
            out['norm_z_logits'] = logits
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo V2+ reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MoCo loss and classification loss.

        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Q = out["norm_z"]
        K = out["momentum/norm_z"]
        B = K[0].shape[0]

        # ------- contrastive loss -------
        contrastive_loss = 0.
        queue = self.queue.clone().detach()
        for i, (k, q) in enumerate(zip(K, Q[::-1])):
            loss = mocov2plus_loss_func(q, k,
                                        queue[:, i].contiguous(),
                                        self.temperature)
            contrastive_loss = contrastive_loss + 0.5 * loss

        # ------- update queue -------
        with torch.no_grad():
            K = gather(torch.stack(K, dim=1))
            self._dequeue_and_enqueue(K)

            q = F.normalize(q, dim=-1)
            q_mean = q.mean(0).pow(2).sum()

        metrics = {
            "train/nce_loss": contrastive_loss,
            "train/q_mean": q_mean,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        loss = contrastive_loss + class_loss
        return loss
