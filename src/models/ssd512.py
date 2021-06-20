import torch
import torch.nn as nn
from itertools import product
from models.ssd300 import ConvRelu, ConvBNRelu, SSD300


class SSD512(SSD300):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(SSD512, self).__init__(num_classes=num_classes, backbone=backbone)

        if any(isinstance(m, nn.BatchNorm2d) for m in backbone.features):
            ConvBlock = ConvBNRelu
        else:
            ConvBlock = ConvRelu

        self.extras = nn.ModuleDict([
            ('conv8_1', ConvBlock(1024, 256, kernel_size=1)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)),
            ('conv9_1', ConvBlock(512, 128, kernel_size=1)),
            ('conv9_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)),
            ('conv10_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)),
            ('conv11_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv11_2', ConvBlock(128, 256, kernel_size=3)),
            ('conv12_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv12_2', ConvBlock(128, 256, kernel_size=2)),
        ])

        self.localizers = nn.ModuleDict({
            'conv4_3': nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            'conv7_1': nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            'conv12_2': nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        })

        self.classifiers = nn.ModuleDict({
            'conv4_3': nn.Conv2d(512, 4 * self.nc, kernel_size=3, padding=1),
            'conv7_1': nn.Conv2d(1024, 6 * self.nc, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(512, 6 * self.nc, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(256, 6 * self.nc, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(256, 6 * self.nc, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(256, 4 * self.nc, kernel_size=3, padding=1),
            'conv12_2': nn.Conv2d(256, 4 * self.nc, kernel_size=3, padding=1),
        })

        self.init_weights(blocks=[self.extras, self.localizers, self.classifiers])

    def _get_default_boxes(self) -> torch.Tensor:
        """ Default Box を生成する

        Returns:
            torch.Tensor (24564, 4): Default Box (coord fmt: [cx, cy, w, h])
        """
        def s_(k, m=7, s_min=0.07, s_max=0.88):
            return s_min + (s_max - s_min) * (k - 1) / (m - 1)

        dboxes = []
        cfg = [[64, 4], [32, 6], [16, 6], [8, 6], [4, 6], [2, 4], [1, 4]]

        for k, (f_k, num_aspects) in enumerate(cfg, start=1):
            aspects = [1, 'add', 2, 1 / 2] if num_aspects == 4 else [1, 'add', 2, 1 / 2, 3, 1 / 3]
            for i, j in product(range(f_k), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for a in aspects:
                    if a == 'add':
                        w = h = pow(s_(k) * s_(k + 1), 0.5)
                    else:
                        w = s_(k) * pow(a, 0.5)
                        h = s_(k) * pow(1 / a, 0.5)
                    dboxes.append([cx, cy, w, h])

        dboxes = torch.tensor(dboxes).clamp(min=0, max=1)
        return dboxes


if __name__ == '__main__':
    import torch
    from torchvision.models import vgg16
    x = torch.rand(2, 3, 512, 512)

    backbone = vgg16(pretrained=True)
    model = SSD512(num_classes=20, backbone=backbone)
    print(model)

    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)

    out_locs = torch.rand(4, 24564, 4)
    out_confs = torch.rand(4, 24564, 21)
    outputs = (out_locs, out_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 21, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))
