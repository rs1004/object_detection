import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.base import DetectionNet
from models.losses import focal_loss


class UpAdd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpAdd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, cx, fx):
        _, _, h, w = cx.size()
        cx = self.conv(cx)
        fx = F.interpolate(fx, size=(h, w), mode='bilinear', align_corners=True)
        return cx + fx


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.outc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        out = self.outc(x)
        return out


class RetinaNet(DetectionNet):
    def __init__(self, num_classes: int, backborn: nn.Module):
        super(RetinaNet, self).__init__()

        self.nc = num_classes + 1  # add background class

        fpn_in_channels = [
            backborn.layer2[-1].conv3.out_channels,
            backborn.layer3[-1].conv3.out_channels,
            backborn.layer4[-1].conv3.out_channels
        ]
        fpn_out_channels = 256

        self.c1 = nn.Sequential(
            backborn.conv1,
            backborn.bn1,
            backborn.relu,
            backborn.maxpool
        )

        self.c2 = backborn.layer1
        self.c3 = backborn.layer2
        self.c4 = backborn.layer3
        self.c5 = backborn.layer4

        self.p3 = UpAdd(fpn_in_channels[0], fpn_out_channels)
        self.p4 = UpAdd(fpn_in_channels[1], fpn_out_channels)
        self.p5 = nn.Conv2d(fpn_in_channels[2], fpn_out_channels, kernel_size=1)
        self.p6 = nn.Conv2d(fpn_in_channels[2], fpn_out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.p3_avgpool = nn.AdaptiveAvgPool2d(output_size=32)
        self.p4_avgpool = nn.AdaptiveAvgPool2d(output_size=16)
        self.p5_avgpool = nn.AdaptiveAvgPool2d(output_size=8)
        self.p6_avgpool = nn.AdaptiveAvgPool2d(output_size=4)
        self.p7_avgpool = nn.AdaptiveAvgPool2d(output_size=2)

        self.regressor = Head(fpn_out_channels, 4 * 9)

        self.classifier = Head(fpn_out_channels, self.nc * 9)

        self.pboxes = self._get_prior_boxes()

        self.init_weights(blocks=[self.p3, self.p4, self.p5, self.p6, self.p7, self.regressor, self.classifier])

    def forward(self, x):
        batch_size = x.size(0)

        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.p5_avgpool(self.p5(c5))
        p6 = self.p6_avgpool(self.p6(c5))
        p7 = self.p7_avgpool(self.p7(p6))

        p4 = self.p4_avgpool(self.p4(c4, p5))
        p3 = self.p3_avgpool(self.p3(c3, p4))

        out_locs = []
        out_confs = []
        for p in [p3, p4, p5, p6, p7]:
            out_locs.append(self.regressor(p).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))

        for p in [p3, p4, p5, p6, p7]:
            out_confs.append(self.classifier(p).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))

        out_locs = torch.cat(out_locs, dim=1)
        out_confs = torch.cat(out_confs, dim=1)
        return out_locs, out_confs

    def _get_prior_boxes(self):
        """ Prior Box を生成する

        Returns:
            torch.Tensor (32526, 4): Prior Box (coord fmt: [cx, cy, w, h])
        """
        pboxes = []

        for f_k in [32, 16, 8, 4, 2]:
            for i, j in product(range(f_k), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for aspect in [1/2, 1, 2]:
                    for size in [2 ** 0, 2 ** (1/3), 2 ** (2/3)]:
                        w = size * pow(1 / aspect, 0.5) / f_k
                        h = size * pow(aspect, 0.5) / f_k
                        pboxes.append([cx, cy, w, h])

        pboxes = torch.tensor(pboxes)
        return pboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_thresholds: tuple = (0.5, 0.4)) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測信頼度)
                            * 予測オフセット : (B, P, 4) (coord fmt: [Δcx, Δcy, Δw, Δh])
                                    (P: PBoxの数. P = 32526 の想定.)
                            * 予測信頼度     : (B, P, num_classes + 1)
            gt_bboxes (list): 正解BBOX座標 [(G1, 4), (G2, 4), ...] (coord fmt: [cx, cy, w, h])
            gt_labels (list): 正解ラベル [(G1,), (G2,)]
            iou_thresholds (float): Potitive / Negative を判定する際の iou の閾値. (Potitive の閾値, Negative の閾値)

        Returns:
            dict: {
                loss: xxx,
                loss_loc: xxx,
                loss_conf: xxx
            }
        """
        out_locs, out_confs = outputs

        device = out_locs.device
        pboxes = self.pboxes.to(device)
        B = out_locs.size(0)
        pos_thresh, neg_thresh = iou_thresholds
        loss_loc = torch.tensor(0.).to(device)
        loss_conf = torch.tensor(0.).to(device)
        for locs, confs, bboxes, labels in zip(out_locs, out_confs, gt_bboxes, gt_labels):
            # to GPU
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            # [Step 1]
            #   各 Default Box を BBox に対応させ、Positive, Negative の判定を行う
            #   - max_iou >= 0.5 の場合、Positive Box とみなし、最大 iou の BBox を対応させる
            #   - max_iou <  0.5 の場合、Negative Box とみなす
            #   - N := Positive Box の個数。N = 0 ならば Loss = 0 とする（skip する）
            bboxes_xyxy = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            pboxes_xyxy = box_convert(pboxes, in_fmt='cxcywh', out_fmt='xyxy')
            max_ious, bbox_ids = box_iou(pboxes_xyxy, bboxes_xyxy).max(dim=1)
            pos_ids, neg_ids = (max_ious >= pos_thresh).nonzero().reshape(-1), (max_ious < neg_thresh).nonzero().reshape(-1)
            N = len(pos_ids)
            if N == 0:
                continue

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            loss_loc += (1 / N) * F.smooth_l1_loss(
                locs[pos_ids],
                self._calc_delta(bboxes=bboxes[bbox_ids[pos_ids]], pboxes=pboxes[pos_ids]),
                reduction='sum'
            )

            # [Step 3]
            #   Positive / Negative Box に対して、Confidence Loss を計算する
            #   - Negative Box の labels は 0 とする
            #   - Negative Box は Loss の上位 len(pos_ids) * 3 個のみを計算に使用する (Hard Negative Mining)
            loss_conf += (1 / N) * (
                focal_loss(confs[pos_ids], labels[bbox_ids[pos_ids]], reduction='sum') +
                focal_loss(confs[neg_ids], torch.zeros_like(labels[bbox_ids[neg_ids]]), reduction='sum')
            )

        # [Step 4]
        #   損失の和を計算する
        loss = loss_conf + loss_loc

        return {
            'loss': (1 / B) * loss,
            'loss_loc': (1 / B) * loss_loc,
            'loss_conf': (1 / B) * loss_conf
        }

    def _calc_delta(self, bboxes: torch.Tensor, pboxes: torch.Tensor, std: list = [0.1, 0.2]) -> torch.Tensor:
        """ Δg を算出する

        Args:
            bboxes (torch.Tensor, [X, 4]): GT BBox
            pboxes (torch.Tensor, [X, 4]): Prior Box
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Δcx, Δcy, Δw, Δh が標準正規分布に従うようにしている.
                                    第1項が Δcx, Δcy に対する値. 第2項が Δw, Δh に対する値.
                                    Defaults to [0.1, 0.2]. (TODO: 使用するデータに対し調査して設定する必要がある)

        Returns:
            torch.Tensor: [X, 4]
        """
        db_cx = (1 / std[0]) * (bboxes[:, 0] - pboxes[:, 0]) / pboxes[:, 2]
        db_cy = (1 / std[0]) * (bboxes[:, 1] - pboxes[:, 1]) / pboxes[:, 3]
        db_w = (1 / std[1]) * (bboxes[:, 2] / pboxes[:, 2]).log()
        db_h = (1 / std[1]) * (bboxes[:, 3] / pboxes[:, 3]).log()

        dbboxes = torch.stack([db_cx, db_cy, db_w, db_h], dim=1).contiguous()
        return dbboxes

    def _calc_coord(self, locs: torch.Tensor, dboxes: torch.Tensor, std: list = [0.1, 0.2]) -> torch.Tensor:
        """ g を算出する

        Args:
            locs (torch.Tensor, [X, 4]): Offset Prediction
            dboxes (torch.Tensor, [X, 4]): Default Box
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Defaults to [0.1, 0.2].

        Returns:
            torch.Tensor: [X, 4]
        """
        b_cx = dboxes[:, 0] + std[0] * locs[:, 0] * dboxes[:, 2]
        b_cy = dboxes[:, 1] + std[0] * locs[:, 1] * dboxes[:, 3]
        b_w = dboxes[:, 2] * (std[1] * locs[:, 2]).exp()
        b_h = dboxes[:, 3] * (std[1] * locs[:, 3]).exp()

        bboxes = torch.stack([b_cx, b_cy, b_w, b_h], dim=1).contiguous()
        return bboxes

    def pre_predict(self, outputs: tuple, conf_thresh: float = 0.4) -> tuple:
        """ モデルの出力結果を予測データに変換する

        Args:
            outputs (tuple): モデルの出力. (予測オフセット, 予測信頼度)
            conf_thresh (float): 信頼度の閾値. Defaults to 0.4.

        Returns:
            tuple: (予測BBox, 予測信頼度, 予測クラス)
                    - 予測BBox   : [N, 32526, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, 32526]
                    - 予測クラス : [N, 32526]
        """
        out_locs, out_confs = outputs
        out_confs = F.softmax(out_confs, dim=-1)

        # to CPU
        out_locs = out_locs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        pred_bboxes = []
        pred_confs = []
        pred_class_ids = []

        for locs, confs in zip(out_locs, out_confs):
            confs, class_ids = confs.max(dim=-1)
            pos_ids = ((class_ids != 0) * (confs >= conf_thresh)).nonzero().reshape(-1)  # 0 is background class
            confs, class_ids = confs[pos_ids], class_ids[pos_ids]
            bboxes = self._calc_coord(locs[pos_ids], self.pboxes[pos_ids])
            bboxes = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')

            pred_bboxes.append(bboxes)
            pred_confs.append(confs)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_confs, pred_class_ids


if __name__ == '__main__':
    import torch
    from torchvision.models import resnet50
    size = 416
    x = torch.rand(2, 3, size, size)

    backborn = resnet50()
    model = RetinaNet(num_classes=20, backborn=backborn)
    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)
    print(len(model.pboxes))

    num_pboxes = sum([(2 ** i) ** 2 * 9 for i in range(1, 6)])
    out_locs = torch.rand(4, num_pboxes, 4)
    out_confs = torch.rand(4, num_pboxes, 21)
    outputs = (out_locs, out_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 20, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))
