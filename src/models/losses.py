import torch
import torch.nn.functional as F


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = 'none',
):
    eps = 1e-7
    yt = F.one_hot(target, num_classes=input.size(-1))
    pt = F.softmax(input, dim=-1).clamp(eps, 1 - eps)
    loss = (-alpha * yt * (1 - pt) ** gamma * torch.log(pt)).sum(dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
