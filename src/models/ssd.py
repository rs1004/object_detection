import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from collections import Counter
from torchvision.ops import box_iou, box_convert
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

    def loss(self, out_ls: torch.Tensor, out_cs: torch.Tensor, batch_bboxes: list, batch_labels: list,
             iou_thresh: float = 0.5, alpha: float = 1.0) -> tuple:
        """ 損失関数

        Args:
            out_ls (torch.Tensor): 予測オフセット (B, D, 4) (fmt: [Δcx, Δcy, Δw, Δh])
                                   D: DBoxの数. D = 8732 の想定.
            out_cs (torch.Tensor): 予測信頼度 (B, D, 21)
            batch_bboxes (list): 正解BBOX座標 [(G1, 4), (G2, 4), ...] (fmt: [cx, cy, w, h])
            batch_labels (list): 正解ラベル [(G1,), (G2,)]
            iou_thresh (float): Potitive / Negative を判定する際の iou の閾値

        Returns:
            tuple: (loss, localization loss, classification loss)
        """

        device = out_ls.device
        dboxes = self.dboxes.to(device)
        N = out_ls.size(0)
        loss_all = loss_l = loss_c = 0
        for out_l, out_c, bboxes, labels in zip(out_ls, out_cs, batch_bboxes, batch_labels):
            # to GPU
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            # [Step 1]
            #   各 Default Box を bbox に対応させ、Positive, Negative の判定を行う
            #   * max_iou >= 0.5 の場合、Positive Box とみなし、最大 iou の bbox を対応させる
            #   * max_iou <  0.5 の場合、Negative Box とみなす
            bboxes_xyxy = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            dboxes_xyxy = box_convert(dboxes, in_fmt='cxcywh', out_fmt='xyxy')
            max_ious, indices = box_iou(dboxes_xyxy, bboxes_xyxy).max(dim=1)
            pos_ids, neg_ids = (max_ious >= iou_thresh).nonzero().reshape(-1), (max_ious < iou_thresh).nonzero().reshape(-1)

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            bboxes_pos = bboxes[indices[pos_ids]]
            dboxes_pos = dboxes[pos_ids]
            dbboxes_pos = self._calc_delta(bboxes=bboxes_pos, dboxes=dboxes_pos)
            loss_l += self._smooth_l1(out_l[pos_ids] - dbboxes_pos).sum()

            # [Step 3]
            #   Positive / Negative Box に対して、Confidence Loss を計算する
            #   * Negative Box の label は 0 とする (Positive Box の label を 1 ずらす)
            #   * Negative Box は Loss の上位 len(pos_ids) * 3 個のみを計算に使用する (Hard Negative Mining)
            labels = labels[indices] + 1
            labels[neg_ids] = 0
            sce = self._softmax_cross_entropy(out_c, labels)
            loss_c += sce[pos_ids].sum() + sce[neg_ids].topk(k=len(pos_ids) * 3).values.sum()

        # [Step 4]
        #   損失の和を計算する
        loss_l /= N
        loss_c /= N
        loss_all = loss_c + alpha * loss_l

        return loss_all, loss_l, loss_c

    def _calc_delta(self, bboxes: torch.Tensor, dboxes: torch.Tensor, variance: list = [0.1, 0.2]) -> torch.Tensor:
        """ Δg を算出する

        Args:
            bboxes (torch.Tensor, [X, 4]): BBox
            dboxes (torch.Tensor, [X, 4]): Default Box
            variance (list, optional): 係数. Defaults to [0.1, 0.2].

        Returns:
            torch.Tensor: [X, 4]
        """
        db_cx = (1 / variance[0]) * (bboxes[:, 0] - dboxes[:, 0]) / dboxes[:, 2]
        db_cy = (1 / variance[0]) * (bboxes[:, 1] - dboxes[:, 1]) / dboxes[:, 3]
        db_w = (1 / variance[1]) * (bboxes[:, 2] / dboxes[:, 2]).log()
        db_h = (1 / variance[1]) * (bboxes[:, 3] / dboxes[:, 3]).log()

        dbboxes = torch.stack([db_cx, db_cy, db_w, db_h], dim=1)
        return dbboxes

    def _smooth_l1(self, x: torch.Tensor) -> torch.Tensor:
        """ smooth l1 を計算する
        Args:
            x (torch.Tensor): any tensor
        Returns:
            torch.Tensor : calculated result
        """
        return torch.where(x.abs() < 1, 0.5 * x * x, x.abs() - 0.5)

    def _softmax_cross_entropy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ softmax cross entropy を計算する
        Args:
            x (torch.Tensor): any tensor
        Returns:
            torch.Tensor : calculated result
        """
        return -nn.functional.log_softmax(pred, dim=-1)[range(len(target)), target]


if __name__ == '__main__':
    import torch
    x = torch.rand(2, 3, 300, 300)

    model = SSD(num_classes=20, pretrained=False)
    out_ls, out_cs = model(x)
    print(out_ls.shape, out_cs.shape)
    for coord in model.dboxes:
        print(coord)

    out_ls = torch.rand(4, 8732, 4)
    out_cs = torch.rand(4, 8732, 21)
    batch_bboxes = [torch.rand(5, 4) for _ in range(4)]
    batch_labels = [torch.randint(0, 20, (5,)) for _ in range(4)]

    print(model.loss(out_ls, out_cs, batch_bboxes, batch_labels))
