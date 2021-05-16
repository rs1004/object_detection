import torch
import torch.nn.functional as F


def focal_loss(input: torch.Tensor, target: torch.Tensor, reduction='mean') -> torch.Tensor:
    alpha = 0.25
    gamma = 2

    target = target.view(-1, 1)

    logpt = F.log_softmax(input, dim=1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.exp()

    loss = -alpha * (1 - pt)**gamma * logpt

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss
