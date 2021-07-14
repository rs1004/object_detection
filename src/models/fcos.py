import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_convert
from models.base import DetectionNet
from models.losses import focal_loss, iou_loss_with_distance

INF = 1e8


class UpAdd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpAdd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, cx, px):
        cx = self.conv(cx)
        px = self.up(px)
        return cx + px


class Head(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4):
        super(Head, self).__init__()
        head_list = []
        for _ in range(num_blocks):
            head_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            head_list.append(nn.ReLU(inplace=True))
        self.headc = nn.Sequential(*head_list)
        self.outc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.headc(x)
        out = self.outc(x)
        return out


class FCOS(DetectionNet):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(FCOS, self).__init__()

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

        self.p3_1 = UpAdd(fpn_in_channels[0], fpn_out_channels)
        self.p4_1 = UpAdd(fpn_in_channels[1], fpn_out_channels)
        self.p5_1 = nn.Conv2d(fpn_in_channels[2], fpn_out_channels, kernel_size=1)

        self.p3_2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.p4_2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.p5_2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.p6_1 = nn.Conv2d(fpn_in_channels[2], fpn_out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_1 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.regressor = Head(fpn_out_channels, 4)
        self.classifier = Head(fpn_out_channels, self.nc)
        self.centerness = nn.Sequential(
            self.classifier.headc,
            nn.Conv2d(fpn_out_channels, 1, kernel_size=3, padding=1)
        )

        self.init_weights(blocks=[
            self.p3_1, self.p4_1, self.p5_1, self.p3_2, self.p4_2, self.p5_2,
            self.p6_1, self.p7_1, self.regressor, self.classifier, self.centerness]
        )

    def forward(self, x):
        batch_size = x.size(0)

        # backbone
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        # connection
        p5 = self.p5_1(c5)
        p4 = self.p4_1(c4, p5)
        p3 = self.p3_1(c3, p4)

        # extract feature
        p3 = self.p3_2(p3)
        p4 = self.p4_2(p4)
        p5 = self.p5_2(p5)
        p6 = self.p6_1(c5)
        p7 = self.p7_1(self.relu(p6))

        # detection head
        out_locs = []
        out_confs = []
        out_cents = []
        f_k_list = []
        for p in [p3, p4, p5, p6, p7]:
            f_k_list.append(p.size(-1))
            out_locs.append(self.regressor(p).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
            out_confs.append(self.classifier(p).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))
            out_cents.append(self.centerness(p).permute(0, 2, 3, 1).reshape(batch_size, -1))

        out_locs = torch.cat(out_locs, dim=1)
        out_confs = torch.cat(out_confs, dim=1)
        out_cents = torch.cat(out_cents, dim=1)
        if not hasattr(self, 'points'):
            self.points = self._get_points(f_k_list)
            self.regress_ranges = self._get_regress_ranges(f_k_list)
        return out_locs, out_confs, out_cents

    def _get_points(self, f_k_list: list):
        """ All Points を生成する

        Returns:
            torch.Tensor (5456, 2): Points (coord fmt: [cx, cy])
        """
        points = []
        for i in range(len(f_k_list)):
            f_k = f_k_list[i]
            for y, x in product(range(f_k), repeat=2):
                points.append([(x + 0.5) / f_k, (y + 0.5) / f_k])
        points = torch.tensor(points)

        return points

    def _get_regress_ranges(self, f_k_list: list):
        """ Regress Ranges を生成する

        Returns:
            torch.Tensor (5456, 2): Regress Ranges
        """
        regress_range_list = [[-1, 0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 1.], [1., INF]]

        regress_ranges = []
        for i in range(len(f_k_list)):
            f_k = f_k_list[i]
            low, upp = regress_range_list[i]
            for y, x in product(range(f_k), repeat=2):
                regress_ranges.append([low, upp])
        regress_ranges = torch.tensor(regress_ranges)

        return regress_ranges

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
        out_locs, out_confs, out_cents = outputs
        device = out_locs.device

        # [Step 1]
        #   target を作成する
        #   - Points を GT に対応させる
        #     - 条件 1: Points の点が BBox に含まれる
        #     - 条件 2: ray の長さの最大値が regress_range の範囲内である
        #     - 条件 3: Points が複数の BBox に対応する場合(ambiguous sample)、BBox の面積が一番小さいものに対応させる
        #   - 対応する GT が存在しない場合、Label を 0 にする

        B, P, C = out_confs.size()
        target_locs = torch.zeros(B, P, 4)
        target_cents = torch.zeros(B, P)
        target_labels = torch.zeros(B, P, dtype=torch.long)

        points = self.points
        regress_ranges = self.regress_ranges
        for i in range(B):
            bboxes = gt_bboxes[i]
            labels = gt_labels[i]

            bboxes_xyxy = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            areas = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]).repeat(len(points), 1)
            d_left = points[:, [0]] - bboxes_xyxy[:, 0]
            d_right = bboxes_xyxy[:, 2] - points[:, [0]]
            d_top = points[:, [1]] - bboxes_xyxy[:, 1]
            d_bottom = bboxes_xyxy[:, 3] - points[:, [1]]
            d = torch.stack([d_left, d_right, d_top, d_bottom], dim=-1)

            # 条件 1
            inside_bbox = (d_left > 0) * (d_right > 0) * (d_top > 0) * (d_bottom > 0)
            areas[~inside_bbox] = 1

            # 条件 2
            max_d = d.max(dim=-1).values
            inside_regress_range = (regress_ranges[:, [0]] < max_d) * (max_d < regress_ranges[:, [1]])
            areas[~inside_regress_range] = 1

            # 条件 3
            min_areas, matched_bbox_ids = areas.min(dim=1)
            locs = d[range(len(points)), matched_bbox_ids]
            cents = (
                locs[:, 0:2].min(dim=1).values / locs[:, 0:2].max(dim=1).values * locs[:, 2:4].min(dim=1).values / locs[:, 2:4].max(dim=1).values
            ).sqrt()
            labels = labels[matched_bbox_ids]
            labels[min_areas == 1] = 0  # 0 が背景クラス. Positive Class は 1 ~

            target_locs[i] = locs
            target_cents[i] = cents
            target_labels[i] = labels

        target_locs = target_locs.to(device)
        target_cents = target_cents.to(device)
        target_labels = target_labels.to(device)

        # [Step 2]
        #   pos_mask, neg_mask を作成する
        #   - pos_mask: Label が 0 でないもの
        #   - neg_mask: Label が 0 のもの

        pos_mask = target_labels > 0
        neg_mask = target_labels == 0

        N = pos_mask.sum()
        # [Step 3]
        #   Positive に対して、 Localization Loss を計算する
        loss_loc = iou_loss_with_distance(out_locs[pos_mask].exp(), target_locs[pos_mask], reduction='sum') / N

        # [Step 4]
        #   Positive に対して、 Centerness Loss を計算する
        loss_cent = F.binary_cross_entropy_with_logits(out_cents[pos_mask], target_cents[pos_mask], reduction='sum') / N

        # [Step 5]
        #   Positive & Negative に対して、Confidence Loss を計算する
        loss_conf = focal_loss(out_confs[pos_mask + neg_mask], target_labels[pos_mask + neg_mask], reduction='sum') / N

        # [Step 5]
        #   損失の和を計算する
        loss = loss_conf + loss_cent + loss_loc

        return {
            'loss': loss,
            'loss_loc': loss_loc,
            'loss_cent': loss_cent,
            'loss_conf': loss_conf
        }

    def _calc_coord(self, locs: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """ g を算出する

        Args:
            locs (torch.Tensor, [X, 4]): Ray Prediction
            points (torch.Tensor, [X, 4]): Points

        Returns:
            torch.Tensor: [X, 4]
        """
        b_xmin = points[:, 0] - locs[:, 0]
        b_ymin = points[:, 1] - locs[:, 1]
        b_xmax = points[:, 0] + locs[:, 2]
        b_ymax = points[:, 1] + locs[:, 3]

        bboxes = torch.stack([b_xmin, b_ymin, b_xmax, b_ymax], dim=1).contiguous()
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
        out_locs, out_confs, out_cents = outputs
        out_locs = out_locs.exp()
        out_confs = (F.softmax(out_confs, dim=-1) * out_cents.sigmoid()[..., None]).sqrt()

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
                bboxes_ = self._calc_coord(locs[pos_mask], self.points[pos_mask])

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
    size = 512
    x = torch.rand(2, 3, size, size)

    backbone = resnet50()
    model = FCOS(num_classes=20, backbone=backbone)
    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    print(len(model.points))

    out_locs = torch.randn(4, 5456, 4)
    out_confs = torch.randn(4, 5456, 21)
    out_cents = torch.randn(4, 5456)
    outputs = (out_locs, out_confs, out_cents)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(1, 21, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))
