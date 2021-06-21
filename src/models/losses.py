import torch
import torch.nn.functional as F


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = 'none',
):
    pt = input.softmax(dim=-1)
    log_pt = input.log_softmax(dim=-1)
    loss = F.nll_loss(alpha * (1 - pt).pow(gamma) * log_pt, target, reduction='none')

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
