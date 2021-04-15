import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=1, padding=0, is_bn=True, args=None):
        super(ConvBlock, self).__init__()
        if args is not None:
            self.conv = args.get('conv')
            self.bn = args.get('bn', nn.Identity())
            self.act = args.get('act')
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels) if is_bn else nn.Identity()
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.reshape(1, -1, 1, 1).expand_as(x) * x
        return out


if __name__ == '__main__':
    m = ConvBlock(3, 10, 10)
    x = torch.rand(5, 3, 20, 20)
    x = m(x)
    print(x.shape)

    m = L2Norm(10, 20)
    x = torch.rand(5, 10, 20, 20)
    x = m(x)
    print(x.shape)
