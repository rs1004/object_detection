from torch.optim import *  # noqa


def Optimizer(params, cfg):
    optimizer = eval(cfg.pop('type'))
    return optimizer(params=params, **cfg)


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    p = [nn.Parameter(torch.rand(2, 2))]
    optimizer = Optimizer(params=p, cfg={'type': 'SGD', 'lr': 0.001})
    print(optimizer)
