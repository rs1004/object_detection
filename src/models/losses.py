import torch
import torch.nn.functional as F


def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2, reduction: str = 'none'):
    pt = F.softmax(input, dim=-1)
    log_pt = F.log_softmax(input, dim=-1)
    loss = F.nll_loss(alpha * (1 - pt).pow(gamma) * log_pt, target, reduction=reduction)
    return loss


def iou_loss_with_distance(input: torch.Tensor, target: torch.Tensor, reduction: str = 'none'):
    eps = 1e-8

    def _calc_area(t):
        return (t[:, 1] + t[:, 0]) * (t[:, 3] + t[:, 2])

    inter = _calc_area(torch.minimum(input, target))
    union = _calc_area(input) + _calc_area(target) - inter
    iou = inter / union.clamp(min=eps)

    loss = -iou.log()

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss
