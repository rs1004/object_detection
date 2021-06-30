import torch
import torch.nn.functional as F


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = 'none',
):
    pt = torch.where(target == 1, input.sigmoid(), 1 - input.sigmoid())
    logpt = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    alpha = torch.where(target == 1, alpha, 1 - alpha)
    loss = alpha * (1 - pt) ** gamma * logpt

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss
