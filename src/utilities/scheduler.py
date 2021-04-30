from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import *  # noqa
from collections import Counter
import math


class MultiStepLRWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma, eta_min=0.1, T_up=0, last_epoch=-1):
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        if T_up > min(milestones):
            raise ValueError("Expected T_up < milestones")
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.eta_min = eta_min
        self.T_up = T_up
        super(MultiStepLRWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.T_up:
            return [(base_lr - self.eta_min) * self.last_epoch / self.T_up + self.eta_min for base_lr in self.base_lrs]
        elif self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


class ExponentialLRWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, gamma, eta_min=0.1, T_up=0, last_epoch=-1):
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.gamma = gamma
        self.eta_min = eta_min
        self.T_up = T_up
        super(ExponentialLRWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.T_up:
            return [(base_lr - self.eta_min) * self.last_epoch / self.T_up + self.eta_min for base_lr in self.base_lrs]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_min = eta_min
        self.eta_min = eta_min
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(base_lr - self.eta_min) * self.T_cur / self.T_up + self.eta_min for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        epoch = self.last_epoch + 1
        self.T_cur = self.T_cur + 1
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
            self.base_lrs = [base_lr * (self.gamma ** self.cycle) for base_lr in self.base_lrs]

        self.eta_min = self.base_eta_min * (self.gamma**self.cycle)
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def Scheduler(optimizer, cfg):
    scheduler = eval(cfg.pop('type'))
    return scheduler(optimizer=optimizer, **cfg)


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torch.optim import SGD

    optimizer = SGD([nn.Parameter(torch.rand(2, 2))], lr=0.1)

    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.0, T_up=5, gamma=0.5)

    epochs = 100
    lrs = []
    for _ in range(epochs):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    fig = plt.figure()
    plt.plot(range(epochs), lrs)
    plt.ylim(0, 0.11)
    fig.savefig('./demo/debug_schedulers/CosineAnnealingWarmUpRestarts.png')
