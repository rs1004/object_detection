import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.layers import ConvBlock, Concatenate
from models.base import DetectionNet


class YoloV3(DetectionNet):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(YoloV3, self).__init__()
        self.nc = num_classes

        self.backbone = backbone.features

        self.neck = nn.ModuleDict([
            ('conv6_1', ConvBlock(1024, 512, kernel_size=1, act='leaky')),
            ('conv6_2', ConvBlock(512, 1024, kernel_size=3, padding=1, act='leaky')),
            ('conv6_3', ConvBlock(1024, 512, kernel_size=1, act='leaky')),
            ('conv6_4', ConvBlock(512, 1024, kernel_size=3, padding=1, act='leaky')),
            ('conv6_5', ConvBlock(1024, 512, kernel_size=1, act='leaky')),
            ('conv6_6', ConvBlock(512, 1024, kernel_size=3, padding=1, act='leaky')),

            ('conv7_1', ConvBlock(1024, 256, kernel_size=1, act='leaky')),
            ('upsample7_1', nn.Upsample(scale_factor=2)),
            ('concat7_1', Concatenate(['darkres5_8', 'upsample7_1'])),

            ('conv8_1', ConvBlock(768, 256, kernel_size=1, act='leaky')),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, padding=1, act='leaky')),
            ('conv8_3', ConvBlock(512, 256, kernel_size=1, act='leaky')),
            ('conv8_4', ConvBlock(256, 512, kernel_size=3, padding=1, act='leaky')),
            ('conv8_5', ConvBlock(512, 256, kernel_size=1, act='leaky')),
            ('conv8_6', ConvBlock(256, 512, kernel_size=3, padding=1, act='leaky')),

            ('conv9_1', ConvBlock(512, 128, kernel_size=1, act='leaky')),
            ('upsample9_1', nn.Upsample(scale_factor=2)),
            ('concat9_1', Concatenate(['darkres4_8', 'upsample9_1'])),

            ('conv10_1', ConvBlock(384, 128, kernel_size=1, act='leaky')),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3, padding=1, act='leaky')),
            ('conv10_3', ConvBlock(256, 128, kernel_size=1, act='leaky')),
            ('conv10_4', ConvBlock(128, 256, kernel_size=3, padding=1, act='leaky')),
            ('conv10_5', ConvBlock(256, 128, kernel_size=1, act='leaky')),
            ('conv10_6', ConvBlock(128, 256, kernel_size=3, padding=1, act='leaky')),
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
                x_max = (j + 1) / f_k
                y = i / f_k
                y_max = (i + 1) / f_k
                for w, h in aspects:
                    w /= S
                    h /= S
                    pboxes.append([x, y, w, h, x_max, y_max])

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
        pboxes = self.pboxes.to(device)
        B = out_locs.size(0)
        loss = loss_loc = loss_conf = loss_obj = 0
        for locs, objs, confs, bboxes, labels in zip(out_locs, out_objs, out_confs, gt_bboxes, gt_labels):
            # to GPU
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            # [Step 1]
            #   各 Prior Box を BBox に対応させ、Positive, Negative の判定を行う
            #   - 各 BBox の (x, y) がどの Grid に含まれるかを判定. 含まれる場合は True とする mask を作成.
            #   - 各 BBox に対し、Grid 内にあり最も IoU が高い Prior Box を取得する
            #       - その Prior Box との IoU が >= 0.5 の場合、 Positive Box とみなす
            #       - i.e. 各 BBox に対応する Prior Box はただ一つになる
            #   - max_iou <  0.5 の場合、Negative Box とみなす
            #   - Positive Box の個数 = 0 ならば Loss = 0 とする（skip する）
            bboxes_xyxy = box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
            pboxes_xyxy = box_convert(pboxes[:, :4], in_fmt='xywh', out_fmt='xyxy')
            ious = box_iou(pboxes_xyxy, bboxes_xyxy)
            bbox_ids = ious.max(dim=1).indices

            mask = torch.logical_and(
                torch.logical_and(
                    pboxes[:, [0]] <= bboxes_xyxy[:, 0],
                    bboxes_xyxy[:, 0] < pboxes[:, [4]]),
                torch.logical_and(
                    pboxes[:, [1]] <= bboxes_xyxy[:, 1],
                    bboxes_xyxy[:, 1] < pboxes[:, [5]]),
            )
            max_ious, pos_ids = (ious * mask).max(dim=0)
            neg_ids = (ious.max(dim=1).values < iou_thresh).nonzero().reshape(-1)
            neg_ids = neg_ids[(neg_ids[:, None] != pos_ids).all(dim=1)]  # pos_ids に含まれるものは除く

            if len(pos_ids) == 0:
                continue

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            dbboxes_pos = self._calc_delta(bboxes[bbox_ids[pos_ids]], pboxes[pos_ids])
            loss_loc += F.binary_cross_entropy_with_logits(
                locs[pos_ids, 0:2],
                dbboxes_pos[:, 0:2],
                reduction='sum'
            ) + F.mse_loss(
                locs[pos_ids, 2:4],
                dbboxes_pos[:, 2:4],
                reduction='sum'
            )

            # [Step 3]
            #   Positive Box に対して、Confidence Loss を計算する
            #   labels は 1 開始なので 0 開始に修正する
            labels = labels - 1
            loss_conf += F.binary_cross_entropy_with_logits(
                confs[pos_ids],
                F.one_hot(labels[bbox_ids[pos_ids]], num_classes=self.nc).float(),
                reduction='sum'
            )

            # [Step 4]
            #   Positive / Negative Box に対して、Objectness Loss を計算する
            objs_pos = objs[pos_ids]
            objs_neg = objs[neg_ids]
            loss_obj += F.binary_cross_entropy_with_logits(objs_pos, torch.zeros_like(objs_pos), reduction='sum') + \
                F.binary_cross_entropy_with_logits(objs_neg, torch.ones_like(objs_neg), reduction='sum')

        # [Step 4]
        #   損失の和を計算する
        loss = loss_loc + loss_conf + loss_obj

        return {
            'loss': (1 / B) * loss,
            'loss_loc': (1 / B) * loss_loc,
            'loss_conf': (1 / B) * loss_conf,
            'loss_obj': (1 / B) * loss_obj
        }

    def _calc_delta(self, bboxes: torch.Tensor, pboxes: torch.Tensor) -> torch.Tensor:
        """ Δg を算出する

        Args:
            bboxes (torch.Tensor, [X, 4]): GT BBox
            pboxes (torch.Tensor, [X, 4]): Prior Box

        Returns:
            torch.Tensor: [X, 4]
        """
        db_x = (bboxes[:, 0] - pboxes[:, 0]) / pboxes[:, 4]
        db_y = (bboxes[:, 1] - pboxes[:, 1]) / pboxes[:, 5]
        db_w = (bboxes[:, 2] / pboxes[:, 2]).log()
        db_h = (bboxes[:, 3] / pboxes[:, 3]).log()

        dbboxes = torch.stack([db_x, db_y, db_w, db_h], dim=1).contiguous()
        return dbboxes

    def _calc_coord(self, locs: torch.Tensor, pboxes: torch.Tensor) -> torch.Tensor:
        """ g を算出する

        Args:
            locs (torch.Tensor, [X, 4]): Offset Prediction
            pboxes (torch.Tensor, [X, 4]): Prior Box

        Returns:
            torch.Tensor: [X, 4]
        """
        b_x = pboxes[:, 0] + locs[:, 0].sigmoid() * pboxes[:, 4]
        b_y = pboxes[:, 1] + locs[:, 1].sigmoid() * pboxes[:, 5]
        b_w = pboxes[:, 2] * locs[:, 2].exp()
        b_h = pboxes[:, 3] * locs[:, 3].exp()

        bboxes = torch.stack([b_x, b_y, b_w, b_h], dim=1).contiguous()
        return bboxes

    def pre_predict(self, outputs: tuple, conf_thresh: float = 0.4):
        """ モデルの出力結果を予測データに変換する

        Args:
            outputs (tuple): モデルの出力. (予測オフセット, 予測信頼度)
            conf_thresh (float): 信頼度の閾値. Defaults to 0.4.

        Returns:
            tuple: (予測BBox, 予測信頼度, 予測クラス)
                    - 予測BBox   : [N, 8732, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, 8732]
                    - 予測クラス : [N, 8732]
        """
        out_locs, out_objs, out_confs = outputs
        out_objs = out_objs.sigmoid()
        out_confs = out_confs.sigmoid()
        out_confs = out_confs * out_objs[..., None]

        # to CPU
        out_locs = out_locs.detach().cpu()
        out_objs = out_objs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        pred_bboxes = []
        pred_confs = []
        pred_class_ids = []

        for locs, objs, confs in zip(out_locs, out_objs, out_confs):
            confs, class_ids = confs.max(dim=-1)
            class_ids = class_ids + 1
            pos_ids = (confs >= conf_thresh).nonzero().reshape(-1)
            confs, class_ids = confs[pos_ids], class_ids[pos_ids]
            bboxes = self._calc_coord(locs[pos_ids], self.pboxes[pos_ids])
            bboxes = box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')

            pred_bboxes.append(bboxes)
            pred_confs.append(confs)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_confs, pred_class_ids


if __name__ == '__main__':
    from torchsummary import summary
    from models.darknet import Darknet53

    backbone = Darknet53()

    model = YoloV3(10, backbone)
    summary(model, (3, 416, 416))
    x = torch.rand(2, 3, 416, 416)
    out_locs, out_objs, out_confs = model(x)
    print(out_locs.shape)
    for name, m in model.named_parameters():
        print(name, len(m.shape))
