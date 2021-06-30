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
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(RetinaNet, self).__init__()

        self.nc = num_classes + 1  # add background class

        fpn_in_channels = [
            backbone.layer2[-1].conv3.out_channels,
            backbone.layer3[-1].conv3.out_channels,
            backbone.layer4[-1].conv3.out_channels
        ]
        fpn_out_channels = 256

        self.c1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.c2 = backbone.layer1
        self.c3 = backbone.layer2
        self.c4 = backbone.layer3
        self.c5 = backbone.layer4

        self.p3 = UpAdd(fpn_in_channels[0], fpn_out_channels)
        self.p4 = UpAdd(fpn_in_channels[1], fpn_out_channels)
        self.p5 = nn.Conv2d(fpn_in_channels[2], fpn_out_channels, kernel_size=1)
        self.p6 = nn.Conv2d(fpn_in_channels[2], fpn_out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.BatchNorm2d(fpn_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1)
        )

        self.regressor = Head(fpn_out_channels, 9 * 4)
        self.classifier = Head(fpn_out_channels, 9 * self.nc)

        self.init_weights(blocks=[self.p3, self.p4, self.p5, self.p6, self.p7, self.regressor, self.classifier])

    def forward(self, x):
        batch_size = x.size(0)

        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.p5(c5)
        p6 = self.p6(c5)
        p7 = self.p7(p6)

        p4 = self.p4(c4, p5)
        p3 = self.p3(c3, p4)

        out_locs = []
        out_confs = []
        f_k_list = []
        for p in [p3, p4, p5, p6, p7]:
            f_k_list.append(p.size(-1))
            out_locs.append(self.regressor(p).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
            out_confs.append(self.classifier(p).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))

        out_locs = torch.cat(out_locs, dim=1)
        out_confs = torch.cat(out_confs, dim=1)
        if not hasattr(self, 'pboxes'):
            self.pboxes = self._get_prior_boxes(f_k_list)
        return out_locs, out_confs

    def _get_prior_boxes(self, f_k_list: list):
        """ Prior Box を生成する

        Returns:
            torch.Tensor (32526, 4): Prior Box (coord fmt: [cx, cy, w, h])
        """
        pboxes = []

        for k, f_k in enumerate(f_k_list, start=1):
            for i, j in product(range(f_k), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for aspect in [0.5, 1, 2]:
                    for scale in [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]:
                        w = 4 * scale / f_k * pow(1 / aspect, 0.5)
                        h = 4 * scale / f_k * pow(aspect, 0.5)
                        pboxes.append([cx, cy, w, h])

        pboxes = torch.tensor(pboxes).clamp(min=0, max=1)
        return pboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_threshs: tuple = (0.4, 0.5)) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測信頼度)
                            * 予測オフセット : (B, P, 4) (coord fmt: [Δcx, Δcy, Δw, Δh])
                                    (P: PBoxの数. P = 32526 の想定.)
                            * 予測信頼度     : (B, P, num_classes + 1)
            gt_bboxes (list): 正解BBOX座標 [(G1, 4), (G2, 4), ...] (coord fmt: [cx, cy, w, h])
            gt_labels (list): 正解ラベル [(G1,), (G2,)]
            iou_threshs (float): Potitive / Negative を判定する際の iou の閾値

        Returns:
            dict: {
                loss: xxx,
                loss_loc: xxx,
                loss_conf: xxx
            }
        """
        out_locs, out_confs = outputs
        device = out_locs.device

        # [Step 1]
        #   target を作成する
        #   - Pred を GT に対応させる
        #     - Pred の Default Box との IoU が最大となる BBox, Label
        #     - BBox との IoU が最大となる Default Box -> その BBox に割り当てる
        #   - 最大 IoU が 0.4 未満の場合、Label を 0 に設定する
        #   - 最大 IoU が 0.5 未満の場合、Label を -1 に設定する (void)

        B, P, C = out_confs.size()
        target_locs = torch.zeros(B, P, 4, device=device)
        target_labels = torch.zeros(B, P, dtype=torch.long, device=device)

        pboxes = self.pboxes.to(device)
        for i in range(B):
            bboxes = gt_bboxes[i].to(device)
            labels = gt_labels[i].to(device)

            bboxes_xyxy = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            pboxes_xyxy = box_convert(pboxes, in_fmt='cxcywh', out_fmt='xyxy')
            ious = box_iou(pboxes_xyxy, bboxes_xyxy)
            best_ious, best_pbox_ids = ious.max(dim=0)
            max_ious, matched_bbox_ids = ious.max(dim=1)

            # 各 BBox に対し最大 IoU を取る Prior Box を選ぶ -> その BBox に割り当てる
            for i in range(len(best_pbox_ids)):
                matched_bbox_ids[best_pbox_ids][i] = i
            max_ious[best_pbox_ids] = iou_threshs[1]

            bboxes = bboxes[matched_bbox_ids]
            locs = self._calc_delta(bboxes, pboxes)
            labels = labels[matched_bbox_ids]
            labels[max_ious.less(iou_threshs[1])] = -1  # void クラス. 計算に含めない.
            labels[max_ious.less(iou_threshs[0])] = 0  # 0 が背景クラス. Positive Class は 1 ~

            target_locs[i] = locs
            target_labels[i] = labels

        # [Step 2]
        #   pos_mask, neg_mask を作成する
        #   - pos_mask: Label が 0 でないもの
        #   - neg_mask: Label が 0 のもの

        pos_mask = target_labels > 0
        neg_mask = target_labels == 0

        N = pos_mask.sum()
        # [Step 3]
        #   Positive に対して、 Localization Loss を計算する
        loss_loc = F.smooth_l1_loss(out_locs[pos_mask], target_locs[pos_mask], reduction='sum') / N

        # [Step 4]
        #   Positive & Negative に対して、Confidence Loss を計算する
        loss_conf = focal_loss(out_confs[pos_mask + neg_mask], target_labels[pos_mask + neg_mask], reduction='sum') / N

        # [Step 5]
        #   損失の和を計算する
        loss = loss_conf + loss_loc

        return {
            'loss': loss,
            'loss_loc': loss_loc,
            'loss_conf': loss_conf
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

    def _calc_coord(self, locs: torch.Tensor, pboxes: torch.Tensor, std: list = [0.1, 0.2]) -> torch.Tensor:
        """ g を算出する

        Args:
            locs (torch.Tensor, [X, 4]): Offset Prediction
            pboxes (torch.Tensor, [X, 4]): Prior Box
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Defaults to [0.1, 0.2].

        Returns:
            torch.Tensor: [X, 4]
        """
        b_cx = pboxes[:, 0] + std[0] * locs[:, 0] * pboxes[:, 2]
        b_cy = pboxes[:, 1] + std[0] * locs[:, 1] * pboxes[:, 3]
        b_w = pboxes[:, 2] * (std[1] * locs[:, 2]).exp()
        b_h = pboxes[:, 3] * (std[1] * locs[:, 3]).exp()

        bboxes = torch.stack([b_cx, b_cy, b_w, b_h], dim=1).contiguous()
        return bboxes

    def pre_predict(self, outputs: tuple, conf_thresh: float = 0.01, top_k: int = 200) -> tuple:
        """ モデルの出力結果を予測データに変換する

        Args:
            outputs (tuple): モデルの出力. (予測オフセット, 予測信頼度)
            conf_thresh (float): 信頼度の閾値
            top_k (int): 検出数

        Returns:
            tuple: (予測BBox, 予測信頼度, 予測クラス)
                    - 予測BBox   : [N, P, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, P]
                    - 予測クラス : [N, P]
        """
        out_locs, out_confs = outputs
        out_confs = F.softmax(out_confs, dim=-1)

        # to CPU
        out_locs = out_locs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        pred_bboxes = []
        pred_scores = []
        pred_class_ids = []

        for locs, confs in zip(out_locs, out_confs):
            bboxes = []
            scores = []
            class_ids = []

            for class_id in range(1, confs.size(1)):  # 0 is background class
                pos_mask = (confs[:, class_id] > conf_thresh) * (confs[:, class_id].argsort(descending=True).argsort() < top_k)
                scores_ = confs[pos_mask, class_id]
                class_ids_ = torch.full_like(scores_, class_id, dtype=torch.long)
                bboxes_ = self._calc_coord(locs[pos_mask], self.pboxes[pos_mask])
                bboxes_ = box_convert(bboxes_, in_fmt='cxcywh', out_fmt='xyxy')

                bboxes.append(bboxes_)
                scores.append(scores_)
                class_ids.append(class_ids_)

            pred_bboxes.append(torch.cat(bboxes))
            pred_scores.append(torch.cat(scores))
            pred_class_ids.append(torch.cat(class_ids))

        return pred_bboxes, pred_scores, pred_class_ids


if __name__ == '__main__':
    import torch
    from torchvision.models import resnet50
    size = 416
    x = torch.rand(2, 3, size, size)

    backbone = resnet50()
    model = RetinaNet(num_classes=20, backbone=backbone)
    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)
    print(len(model.pboxes))

    out_locs = torch.rand(4, 32526, 4)
    out_confs = torch.rand(4, 32526, 21)
    outputs = (out_locs, out_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 20, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))
