import torch
import torch.nn as nn


class Concatenate(nn.Module):
    def __init__(self, keys: list):
        super(Concatenate, self).__init__()
        self.keys = keys

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return x
