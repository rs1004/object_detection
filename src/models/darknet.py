import torch
import torch.nn as nn
import numpy as np
import urllib
from models.layers import ConvBlock
from functools import partial
from pathlib import Path


class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DarkResidualBlock, self).__init__()
        self.layer1 = ConvBlock(in_channels, mid_channels, kernel_size=1, act='leaky')
        self.layer2 = ConvBlock(mid_channels, out_channels, kernel_size=3, padding=1, act='leaky')

    def forward(self, x):
        res = x
        out = self.layer2(self.layer1(x))
        out += res

        return out


class Darknet53(nn.Module):
    def __init__(self, pretrained=True):
        super(Darknet53, self).__init__()
        self.features = nn.ModuleDict([
            ('conv1_1', ConvBlock(3, 32, kernel_size=3, padding=1, act='leaky')),
            ('conv1_2', ConvBlock(32, 64, kernel_size=3, stride=2, padding=1, act='leaky')),

            ('darkres2_1', DarkResidualBlock(64, 32, 64)),
            ('conv2_1', ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, act='leaky')),

            ('darkres3_1', DarkResidualBlock(128, 64, 128)),
            ('darkres3_2', DarkResidualBlock(128, 64, 128)),
            ('conv3_1', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, act='leaky')),

            ('darkres4_1', DarkResidualBlock(256, 128, 256)),
            ('darkres4_2', DarkResidualBlock(256, 128, 256)),
            ('darkres4_3', DarkResidualBlock(256, 128, 256)),
            ('darkres4_4', DarkResidualBlock(256, 128, 256)),
            ('darkres4_5', DarkResidualBlock(256, 128, 256)),
            ('darkres4_6', DarkResidualBlock(256, 128, 256)),
            ('darkres4_7', DarkResidualBlock(256, 128, 256)),
            ('darkres4_8', DarkResidualBlock(256, 128, 256)),
            ('conv4_1', ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, act='leaky')),

            ('darkres5_1', DarkResidualBlock(512, 256, 512)),
            ('darkres5_2', DarkResidualBlock(512, 256, 512)),
            ('darkres5_3', DarkResidualBlock(512, 256, 512)),
            ('darkres5_4', DarkResidualBlock(512, 256, 512)),
            ('darkres5_5', DarkResidualBlock(512, 256, 512)),
            ('darkres5_6', DarkResidualBlock(512, 256, 512)),
            ('darkres5_7', DarkResidualBlock(512, 256, 512)),
            ('darkres5_8', DarkResidualBlock(512, 256, 512)),
            ('conv5_1', ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1, act='leaky')),

            ('darkres6_1', DarkResidualBlock(1024, 512, 1024)),
            ('darkres6_2', DarkResidualBlock(1024, 512, 1024)),
            ('darkres6_3', DarkResidualBlock(1024, 512, 1024)),
            ('darkres6_4', DarkResidualBlock(1024, 512, 1024)),
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(1024, 1000)

        if pretrained:
            self._load()
        else:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    # He の初期化
                    # [memo] sigmoid, tanh を使う場合はXavierの初期値, Relu を使用する場合は He の初期値を使用する
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _load(self):
        save_path = torch.hub.get_dir() + '/checkpoints/darknet53.pth'

        if Path(save_path).exists():
            self.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        else:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            url = 'https://pjreddie.com/media/files/yolov3.weights'
            req = urllib.request.Request(url)
            print('Downloading: "{}" to {}\n'.format(url, save_path))
            with urllib.request.urlopen(req) as res:
                get_parameter = partial(self._get_parameter, res=res)

                # read head
                res.read(16)

                # read body
                for name, m in self.features.items():
                    if isinstance(m, ConvBlock):
                        m.bn.bias = get_parameter((m.bn.num_features, ))
                        m.bn.weight = get_parameter((m.bn.num_features, ))
                        m.bn.running_mean = get_parameter((m.bn.num_features, ), return_tensor=True)
                        m.bn.running_var = get_parameter((m.bn.num_features, ), return_tensor=True)
                        m.conv.weight = get_parameter((m.conv.out_channels, m.conv.in_channels, m.conv.kernel_size[0], m.conv.kernel_size[0]))
                    elif isinstance(m, DarkResidualBlock):
                        m.layer1.bn.bias = get_parameter((m.layer1.bn.num_features, ))
                        m.layer1.bn.weight = get_parameter((m.layer1.bn.num_features, ))
                        m.layer1.bn.running_mean = get_parameter((m.layer1.bn.num_features, ), return_tensor=True)
                        m.layer1.bn.running_var = get_parameter((m.layer1.bn.num_features, ), return_tensor=True)
                        m.layer1.conv.weight = get_parameter(
                            (m.layer1.conv.out_channels, m.layer1.conv.in_channels, m.layer1.conv.kernel_size[0], m.layer1.conv.kernel_size[0])
                        )

                        m.layer2.bn.bias = get_parameter((m.layer2.bn.num_features, ))
                        m.layer2.bn.weight = get_parameter((m.layer2.bn.num_features, ))
                        m.layer2.bn.running_mean = get_parameter((m.layer2.bn.num_features, ), return_tensor=True)
                        m.layer2.bn.running_var = get_parameter((m.layer2.bn.num_features, ), return_tensor=True)
                        m.layer2.conv.weight = get_parameter(
                            (m.layer2.conv.out_channels, m.layer2.conv.in_channels, m.layer2.conv.kernel_size[0], m.layer2.conv.kernel_size[0])
                        )

            torch.save(self.state_dict(), save_path)

    def _get_parameter(self, shape, res, return_tensor=False):
        param = np.ndarray(
            shape=shape,
            dtype='float32',
            buffer=res.read(np.product(shape) * 4))
        t = torch.as_tensor(param.copy())
        if return_tensor:
            return t
        else:
            return nn.Parameter(t)

    def forward(self, x):
        for name, m in self.features.items():
            x = m(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = Darknet53(pretrained=True)
    summary(model, (3, 256, 256))
