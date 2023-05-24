from typing import Optional, Dict, List, Sequence, Tuple

import torch
from torch import functional as F


@torch.no_grad()
def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[float]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """
    maxk = max(top_k)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=-1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        if key in out:
            value += out[batch_size_key] * out[key]
            n += out[batch_size_key]
    value = value / n
    try:
        return value.squeeze(0)
    except AttributeError:
        return value


@torch.no_grad()
def bias_super_balanced_accuracy(
        correct: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_classes is None:
        num_classes = torch.max(targets, dim=0)[0]
        num_classes = num_classes.add(1).cpu()
    base = torch.cat([torch.tensor([1]), num_classes[:-1]]).cumprod(0)
    collapsed_targets = targets.cpu() @ base
    collapsed_targets.squeeze_()

    strata = []
    num_strata = 0
    worst_accuracy = float('inf')
    for i in range(int(num_classes.prod())):
        idxs = collapsed_targets == i
        # need to safe guard from funny bug if a combination is not represented at all
        if idxs.any():
            num_strata += 1
            mean = correct[idxs].mean()
            if float(mean) < float(worst_accuracy):
                worst_accuracy = float(mean)
            strata.append(mean)
    balanced_accuracy = sum(strata, 0.) / num_strata

    return balanced_accuracy, worst_accuracy


@torch.no_grad()
def bias_unsuper_balanced_accuracy(
        correct: torch.Tensor,
        biased_preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #  majority_mean = 0.5 * unbiased_correct_preds[biased_correct_preds > 0.5].mean()
    #  minority_mean = 0.5 * unbiased_correct_preds[biased_correct_preds <= 0.5].mean()
    #  balanced_accuracy = majority_mean + minority_mean
    #  return balanced_accuracy
    groups = torch.stack([biased_preds, targets], dim=-1)
    if num_classes is None:
        num_classes = torch.max(targets, dim=0)[0]
        num_classes = num_classes.add(1).cpu()
        num_classes = torch.stack([num_classes, num_classes], dim=0)
    return bias_super_balanced_accuracy(
        correct,
        groups,
        num_classes=num_classes
        )
