import torch
import torch.nn.functional as F


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = 'none',
):
    pt = F.sigmoid(input)
    log_pt = F.logsigmoid(input)
    loss = F.nll_loss(alpha * (1 - pt).pow(gamma) * log_pt, target, reduction=reduction)

    return loss
