import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.darknet import ConvBlock
from models.base import DetectionNet


class Concatenate(nn.Module):
    def __init__(self, keys: list):
        super(Concatenate, self).__init__()
        self.keys = keys

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return x


class YoloV3(DetectionNet):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(YoloV3, self).__init__()
        self.nc = num_classes

        self.backbone = backbone.features

        self.neck = nn.ModuleDict([
            ('conv6_1', ConvBlock(1024, 512, kernel_size=1)),
            ('conv6_2', ConvBlock(512, 1024, kernel_size=3, padding=1)),
            ('conv6_3', ConvBlock(1024, 512, kernel_size=1)),
            ('conv6_4', ConvBlock(512, 1024, kernel_size=3, padding=1)),
            ('conv6_5', ConvBlock(1024, 512, kernel_size=1)),
            ('conv6_6', ConvBlock(512, 1024, kernel_size=3, padding=1)),

            ('conv7_1', ConvBlock(1024, 256, kernel_size=1)),
            ('upsample7_1', nn.Upsample(scale_factor=2)),
            ('concat7_1', Concatenate(['darkres5_8', 'upsample7_1'])),

            ('conv8_1', ConvBlock(768, 256, kernel_size=1)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, padding=1)),
            ('conv8_3', ConvBlock(512, 256, kernel_size=1)),
            ('conv8_4', ConvBlock(256, 512, kernel_size=3, padding=1)),
            ('conv8_5', ConvBlock(512, 256, kernel_size=1)),
            ('conv8_6', ConvBlock(256, 512, kernel_size=3, padding=1)),

            ('conv9_1', ConvBlock(512, 128, kernel_size=1)),
            ('upsample9_1', nn.Upsample(scale_factor=2)),
            ('concat9_1', Concatenate(['darkres4_8', 'upsample9_1'])),

            ('conv10_1', ConvBlock(384, 128, kernel_size=1)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3, padding=1)),
            ('conv10_3', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_4', ConvBlock(128, 256, kernel_size=3, padding=1)),
            ('conv10_5', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_6', ConvBlock(128, 256, kernel_size=3, padding=1)),
        ])

        self.yolo_head = nn.ModuleDict({
            'conv6_6': nn.Conv2d(1024, 3 * (4 + 1 + self.nc), kernel_size=1),
            'conv8_6': nn.Conv2d(512, 3 * (4 + 1 + self.nc), kernel_size=1),
            'conv10_6': nn.Conv2d(256, 3 * (4 + 1 + self.nc), kernel_size=1),
        })

        self.concat_keys = ['darkres4_8', 'darkres5_8', 'upsample7_1', 'upsample9_1']

        self.pboxes = self._get_prior_boxes()

        self.init_weights(blocks=[self.neck, self.yolo_head])

    def forward(self, x):
        batch_size = x.size(0)

        srcs = dict.fromkeys(self.concat_keys)
        res = dict.fromkeys(self.yolo_head.keys())
        for name, m in self.backbone.items():
            x = m(x)
            if name in srcs:
                srcs[name] = x

        for name, m in self.neck.items():
            if isinstance(m, Concatenate):
                x = m(*[srcs.pop(key) for key in m.keys])
            else:
                x = m(x)
                if name in srcs:
                    srcs[name] = x
                elif name in res:
                    res[name] = x

        out_locs = []
        out_objs = []
        out_confs = []
        for name in self.yolo_head:
            outputs = self.yolo_head[name](res[name]).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4 + 1 + self.nc)
            out_locs.append(
                outputs[..., 0:4]
            )
            out_objs.append(
                outputs[..., 4]
            )
            out_confs.append(
                outputs[..., 5:]
            )
        out_locs, out_objs, out_confs = torch.cat(out_locs, dim=1), torch.cat(out_objs, dim=1), torch.cat(out_confs, dim=1)
        return out_locs, out_objs, out_confs

    def _get_prior_boxes(self) -> torch.Tensor:
        S = 416
        pboxes = []
        for f_k, aspects in [
            [13, [[116, 90], [156, 198], [373, 326]]],
            [26, [[30, 61],  [62, 45],  [59, 119]]],
            [52, [[10, 13],  [16, 30],  [33, 23]]]
        ]:
            for i, j in product(range(f_k), repeat=2):
                x = j / f_k
                y = i / f_k
                grid_length = 1 / f_k
                for w, h in aspects:
                    w /= S
                    h /= S
                    pboxes.append([x, y, w, h, grid_length])

        pboxes = torch.tensor(pboxes)
        return pboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_thresh: float = 0.5) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測存在率,  予測信頼度)
                            * 予測オフセット : (B, P, 4) (coord fmt: [Δx, Δy, Δw, Δh]) (P: PBoxの数. P = 10647 の想定.)
                            * 予測存在率     : (B, P)
                            * 予測信頼度     : (B, P, num_classes)
            gt_bboxes (list): 正解BBOX座標 [(G1, 4), (G2, 4), ...] (coord fmt: [x, y, w, h])
            gt_labels (list): 正解ラベル [(G1,), (G2,)]
            iou_thresh (float): Potitive / Negative を判定する際の iou の閾値

        Returns:
            dict: {
                loss: xxx,
                loss_loc: xxx,
                loss_obj: xxx,
                loss_conf: xxx
            }
        """
        out_locs, out_objs, out_confs = outputs
        device = out_locs.device

        # [Step 1]
        #   target を作成する
        #   - Pred を GT に対応させる
        #     - Grid 内に (x, y) が含まれ、BBox との IoU が最大となる Prior Box -> その BBox に割り当てる
        #   - 最大 IoU が 0.5 以上かつ GT に対応しない場合、 Label を -1 に設定する (ignore 対象とする)
        #   - 最大 IoU が 0.5 未満の場合、Label を 0 に設定する

        B, P, C = out_confs.size()
        target_locs = torch.zeros(B, P, 4)
        target_labels = torch.zeros(B, P, dtype=torch.long)

        pboxes, grid_length = self.pboxes.to(device).split(4, dim=1)
        for i in range(B):
            bboxes = gt_bboxes[i]
            labels = gt_labels[i]

            is_in_grid = (pboxes[:, [0]] <= bboxes[:, 0]) * (bboxes[:, 0] < pboxes[:, [0]] + grid_length) * \
                (pboxes[:, [1]] <= bboxes[:, 1]) * (bboxes[:, 1] < pboxes[:, [1]] + grid_length)
            bboxes_xyxy = box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
            pboxes_xyxy = box_convert(pboxes, in_fmt='xywh', out_fmt='xyxy')
            ious = box_iou(pboxes_xyxy, bboxes_xyxy)
            best_ious, best_pbox_ids = (ious * is_in_grid).max(dim=0)
            max_ious, matched_bbox_ids = ious.max(dim=1)

            # 各 BBox に対し最大 IoU を取る Prior Box を選ぶ -> その BBox に割り当てる
            matched_bbox_ids[best_pbox_ids] = torch.arange(len(best_pbox_ids))
            max_ious[best_pbox_ids] = 1.

            bboxes = bboxes[matched_bbox_ids]
            locs = self._calc_delta(bboxes, pboxes, grid_length)
            labels = labels[matched_bbox_ids]
            labels[max_ious < 1.] = -1  # void クラス
            labels[max_ious.less(iou_thresh)] = 0  # 0 が背景クラス. Positive Class は 1 ~

            target_locs[i] = locs
            target_labels[i] = labels

        target_locs = target_locs.to(device)
        target_labels = target_labels.to(device)

        # [Step 2]
        #   pos_mask, neg_mask を作成する
        #   - pos_mask: Label が > 0 のもの
        #   - neg_mask: label が = 0 のもの
        pos_mask = target_labels > 0
        neg_mask = target_labels == 0

        N = pos_mask.sum()
        # [Step 2]
        #   Positive に対して、 Localization Loss を計算する
        loss_loc = (
            F.mse_loss(
                out_locs[pos_mask][..., :2].sigmoid(),
                target_locs[pos_mask][..., :2],
                reduction='sum'
            ) + F.mse_loss(
                out_locs[pos_mask][..., 2:],
                target_locs[pos_mask][..., 2:],
                reduction='sum')
        ) / N

        # [Step 3]
        #   Positive に対して、Confidence Loss を計算する
        loss_conf = F.binary_cross_entropy_with_logits(
            out_confs[pos_mask],
            F.one_hot(target_labels[pos_mask] - 1, num_classes=self.nc).float(),
            reduction='sum'
        ) / N

        # [Step 4]
        #   Positive & Negative に対して、 Objectness Loss を計算する
        loss_obj = F.binary_cross_entropy_with_logits(
            out_objs[pos_mask + neg_mask],
            pos_mask[pos_mask + neg_mask].float(),
            reduction='sum'
        ) / N

        # [Step 5]
        #   損失の和を計算する
        loss = loss_loc + loss_obj + loss_conf

        return {
            'loss': loss,
            'loss_loc': loss_loc,
            'loss_conf': loss_conf,
            'loss_obj': loss_obj
        }

    def _calc_delta(self, bboxes: torch.Tensor, pboxes: torch.Tensor, grid_length: torch.Tensor,
                    std: list = [0.1, 0.2]) -> torch.Tensor:
        """ Δg を算出する

        Args:
            bboxes (torch.Tensor, [X, 4]): GT BBox
            pboxes (torch.Tensor, [X, 4]): Prior Box
            grid_length (torch.Tensor, [X, 1]): Grid Length
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Δcx, Δcy, Δw, Δh が標準正規分布に従うようにしている.
                                    第1項が Δcx, Δcy に対する値. 第2項が Δw, Δh に対する値.
                                    Defaults to [0.1, 0.2]. (TODO: 使用するデータに対し調査して設定する必要がある)

        Returns:
            torch.Tensor: [X, 4]
        """
        db_x = (1 / std[0]) * (bboxes[:, 0] - pboxes[:, 0]) / grid_length.squeeze()
        db_y = (1 / std[0]) * (bboxes[:, 1] - pboxes[:, 1]) / grid_length.squeeze()
        db_w = (1 / std[1]) * (bboxes[:, 2] / pboxes[:, 2]).log()
        db_h = (1 / std[1]) * (bboxes[:, 3] / pboxes[:, 3]).log()

        dbboxes = torch.stack([db_x, db_y, db_w, db_h], dim=1).contiguous()
        return dbboxes

    def _calc_coord(self, locs: torch.Tensor, pboxes: torch.Tensor, grid_length: torch.Tensor,
                    std: list = [0.1, 0.2]) -> torch.Tensor:
        """ g を算出する

        Args:
            locs (torch.Tensor, [X, 4]): Offset Prediction
            pboxes (torch.Tensor, [X, 4]): Prior Box
            grid_length (torch.Tensor, [X, 1]): Grid Length
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Δcx, Δcy, Δw, Δh が標準正規分布に従うようにしている.
                                    第1項が Δcx, Δcy に対する値. 第2項が Δw, Δh に対する値.
                                    Defaults to [0.1, 0.2]. (TODO: 使用するデータに対し調査して設定する必要がある)

        Returns:
            torch.Tensor: [X, 4]
        """
        b_x = pboxes[:, 0] + std[0] * locs[:, 0] * grid_length.squeeze()
        b_y = pboxes[:, 1] + std[0] * locs[:, 1] * grid_length.squeeze()
        b_w = pboxes[:, 2] * (std[1] * locs[:, 2]).exp()
        b_h = pboxes[:, 3] * (std[1] * locs[:, 3]).exp()

        bboxes = torch.stack([b_x, b_y, b_w, b_h], dim=1).contiguous()
        return bboxes

    def pre_predict(self, outputs: tuple, conf_thresh: float = 0.01, top_k: int = 200) -> tuple:
        """ モデルの出力結果を予測データに変換する

        Args:
            outputs (tuple): モデルの出力. (予測オフセット, 予測信頼度)
            conf_thresh (float): 信頼度の閾値
            top_k (int): 検出数

        Returns:
            tuple: (予測BBox, 予測信頼度, 予測クラス)
                    - 予測BBox   : [N, 8732, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, 8732]
                    - 予測クラス : [N, 8732]
        """
        out_locs, out_objs, out_confs = outputs
        out_locs[..., :2] = out_locs[..., :2].sigmoid()
        out_objs = out_objs.sigmoid()
        out_confs = out_confs.sigmoid()
        out_confs = out_confs * out_objs[..., None]

        # to CPU
        out_locs = out_locs.detach().cpu()
        out_objs = out_objs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        pred_bboxes = []
        pred_scores = []
        pred_class_ids = []

        for locs, objs, confs in zip(out_locs, out_objs, out_confs):
            bboxes = []
            scores = []
            class_ids = []

            for class_id in range(confs.size(1)):
                pos_mask = (confs[:, class_id] > conf_thresh) * (confs[:, class_id].argsort(descending=True).argsort() < top_k)
                scores_ = confs[pos_mask, class_id]
                class_ids_ = torch.full_like(scores_, class_id + 1, dtype=torch.long)
                bboxes_ = self._calc_coord(locs[pos_mask], self.dboxes[pos_mask])
                bboxes_ = box_convert(bboxes_, in_fmt='xywh', out_fmt='xyxy')

                bboxes.append(bboxes_)
                scores.append(scores_)
                class_ids.append(class_ids_)

            pred_bboxes.append(torch.cat(bboxes))
            pred_scores.append(torch.cat(scores))
            pred_class_ids.append(torch.cat(class_ids))
        return pred_bboxes, pred_scores, pred_class_ids


if __name__ == '__main__':
    from torchsummary import summary
    from models.darknet import Darknet53

    backbone = Darknet53()

    model = YoloV3(20, backbone)
    summary(model, (3, 416, 416))
    x = torch.rand(2, 3, 416, 416)
    out_locs, out_objs, out_confs = model(x)
    print(out_locs.shape, out_objs.shape, out_confs.shape)

    outputs = (out_locs, out_objs, out_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 21, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))
