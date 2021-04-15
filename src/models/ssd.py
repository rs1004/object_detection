import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from collections import Counter
try:
    from .layers import ConvBlock, L2Norm
except ImportError:
    from layers import ConvBlock, L2Norm


class SSD(nn.Module):
    def __init__(self, num_classes, pretrained=True, pretrained_weights=None):
        super(SSD, self).__init__()
        self.nc = num_classes + 1  # add background class

        if pretrained_weights:
            bb = vgg16_bn(pretrained=False)
            bb.load_state_dict(torch.load(pretrained_weights))
        else:
            bb = vgg16_bn(pretrained=pretrained)

        self.features = self._parse_features(bb.features[:-1])

        self.extras = nn.ModuleDict([
            ('conv6_1', ConvBlock(512, 1024, kernel_size=3, padding=1)),
            ('conv7_1', ConvBlock(1024, 1024, kernel_size=1)),

            ('conv8_1', ConvBlock(1024, 256, kernel_size=1)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)),

            ('conv9_1', ConvBlock(512, 128, kernel_size=1)),
            ('conv9_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)),

            ('conv10_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3)),

            ('conv11_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv11_2', ConvBlock(128, 256, kernel_size=3)),
        ])

        self.localizers = {
            'conv4_3': nn.Sequential(
                L2Norm(512),
                nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1)
            ),
            'conv7_1': nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        }

        self.classifiers = {
            'conv4_3': nn.Sequential(
                L2Norm(512),
                nn.Conv2d(in_channels=512, out_channels=4 * self.nc, kernel_size=3, padding=1)
            ),
            'conv7_1': nn.Conv2d(in_channels=1024, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
        }

        self.dboxes = self._get_dboxes()

    def forward(self, x):
        batch_size = x.size(0)
        out_ls = []
        out_cs = []
        for name, m in self.features.items():
            x = m(x)
            if name in self.localizers:
                out_ls.append(self.localizers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
                out_cs.append(self.classifiers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))

        for name, m in self.extras.items():
            x = m(x)
            if name in self.localizers:
                out_ls.append(self.localizers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
                out_cs.append(self.classifiers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))

        out_ls, out_cs = torch.cat(out_ls, dim=1), torch.cat(out_cs, dim=1)
        return out_ls, out_cs

    def _parse_features(self, vgg_features: nn.Sequential) -> nn.ModuleDict:
        """ torchvision の VGG16 モデルの特徴抽出層を ConvBlock にパースする

        Args:
            vgg_features (nn.Sequential): features of vgg16

        Returns:
            nn.ModuleDict: ConvBlock の集合. conv1_1 ~ conv5_3
        """
        for m in vgg_features:
            if isinstance(m, nn.MaxPool2d):
                m.ceil_mode = True

        l_counter = Counter({'layer': 1})
        m_counter = Counter()
        features = nn.ModuleDict()
        args = {}
        for m in vgg_features:
            if isinstance(m, nn.Conv2d):
                args['conv'] = m
            elif isinstance(m, nn.BatchNorm2d):
                args['bn'] = m
            elif isinstance(m, nn.ReLU):
                args['act'] = m
                m_counter['block'] += 1
                features[f"conv{l_counter['layer']}_{m_counter['block']}"] = ConvBlock(args=args)
                args.clear()
            elif isinstance(m, nn.MaxPool2d):
                features[f"pool{l_counter['layer']}"] = m
                l_counter['layer'] += 1
                m_counter.clear()
                args.clear()

        return features

    def _get_dboxes(self) -> torch.Tensor:
        """ Default Box を生成する

        Returns:
            torch.Tensor (8732, 4): Default Box (fmt: [cx, cy, w, h])
        """
        def s_(k, m=6, s_min=0.2, s_max=0.9):
            return s_min + (s_max - s_min) * (k - 1) / (m - 1)

        dboxes = []
        cfg = [[38, 38, 4], [19, 19, 6], [10, 10, 6], [5, 5, 6], [3, 3, 4], [1, 1, 4]]

        for k, (H, W, num_aspects) in enumerate(cfg, start=1):
            aspects = [1, 2, 1 / 2, 'add'] if num_aspects == 4 else [1, 2, 3, 1 / 2, 1 / 3, 'add']
            for i in range(H):
                for j in range(W):
                    cx = (j + 0.5) / W
                    cy = (i + 0.5) / H
                    for a in aspects:
                        if a == 'add':
                            w = h = pow(s_(k) * s_(k + 1), 0.5)
                        else:
                            w = s_(k) * pow(a, 0.5)
                            h = s_(k) * pow(1 / a, 0.5)
                        dboxes.append([cx, cy, w, h])

        dboxes = torch.tensor(dboxes)
        return dboxes

    def loss(self, out_ls, out_cs, bboxes, labels):
        # dboxes = self.dboxes.to(bboxes.device)
        pass


if __name__ == '__main__':
    import torch
    x = torch.rand(2, 3, 300, 300)

    model = SSD(num_classes=20, pretrained=False)
    out_ls, out_cs = model(x)
    print(out_ls.shape, out_cs.shape)
    for coord in model.dboxes:
        print(coord)
