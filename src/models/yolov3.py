import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.layers import ConvBlock, Concatenate
from models.base import DetectionNet


class YoloV3(DetectionNet):
    def __init__(self, num_classes: int, backborn: nn.Module):
        super(YoloV3, self).__init__()
        self.nc = num_classes

        self.backborn = backborn.features

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

        self.localizers = nn.ModuleDict({
            'conv6_6': nn.Conv2d(1024, 3 * 4, kernel_size=1),
            'conv8_6': nn.Conv2d(512, 3 * 4, kernel_size=1),
            'conv10_6': nn.Conv2d(256, 3 * 4, kernel_size=1),
        })

        self.objectnesses = nn.ModuleDict({
            'conv6_6': nn.Conv2d(1024, 3 * 1, kernel_size=1),
            'conv8_6': nn.Conv2d(512, 3 * 1, kernel_size=1),
            'conv10_6': nn.Conv2d(256, 3 * 1, kernel_size=1),
        })

        self.classifiers = nn.ModuleDict({
            'conv6_6': nn.Conv2d(1024, 3 * self.nc, kernel_size=1),
            'conv8_6': nn.Conv2d(512, 3 * self.nc, kernel_size=1),
            'conv10_6': nn.Conv2d(256, 3 * self.nc, kernel_size=1),
        })

        self.concat_keys = ['darkres4_8', 'darkres5_8', 'upsample7_1', 'upsample9_1']

        self.pboxes = self._get_prior_boxes()

        self.init_weights(blocks=[self.neck, self.localizers, self.objectnesses, self.classifiers])

    def forward(self, x):
        batch_size = x.size(0)

        srcs = dict.fromkeys(self.concat_keys)
        res = dict.fromkeys(self.localizers.keys())
        for name, m in self.backborn.items():
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
        for name in self.localizers:
            out_locs.append(
                self.localizers[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, 4)
            )
            out_objs.append(
                self.objectnesses[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, 1)
            )
            out_confs.append(
                self.classifiers[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, self.nc)
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
                for w, h in aspects:
                    w /= S
                    h /= S
                    pboxes.append([x, y, w, h])

        pboxes = torch.tensor(pboxes)
        return pboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_thresh: float = 0.5) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測存在率,  予測信頼度)
                            * 予測オフセット : (B, P, 4) (coord fmt: [Δx, Δy, Δw, Δh]) (P: PBoxの数. P = 10647 の想定.)
                            * 予測存在率     : (B, P, 1)
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
            #   - max_iou >= 0.5 の場合、Positive Box とみなし、最大 iou の BBox を対応させる
            #   - max_iou <  0.5 の場合、Negative Box とみなす
            #   - N := Positive Box の個数。N = 0 ならば Loss = 0 とする（skip する）
            bboxes_xyxy = box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
            pboxes_xyxy = box_convert(pboxes, in_fmt='xywh', out_fmt='xyxy')
            max_ious, bbox_ids = box_iou(pboxes_xyxy, bboxes_xyxy).max(dim=1)
            bboxes, labels = bboxes[bbox_ids], labels[bbox_ids]
            pos_ids, neg_ids = (max_ious >= iou_thresh).nonzero().reshape(-1), (max_ious < iou_thresh).nonzero().reshape(-1)
            N = len(pos_ids)
            M = len(neg_ids)
            if N == 0:
                continue

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            bboxes_pos = bboxes[pos_ids]
            pboxes_pos = pboxes[pos_ids]
            dbboxes_pos = self._calc_delta(bboxes=bboxes_pos, pboxes=pboxes_pos)
            loss_loc += (1 / N) * F.mse_loss(locs[pos_ids], dbboxes_pos, reduction='sum')

            # [Step 3]
            #   Positive Box に対して、Confidence Loss を計算する
            #   labels は 1 開始なので 0 開始に修正する
            labels.sub_(1)
            confs_pos = confs[pos_ids]
            labels_pos = labels[pos_ids]
            loss_conf += (1 / N) * F.cross_entropy(confs_pos, labels_pos, reduction='sum')

            # [Step 4]
            #   Positive / Negative Box に対して、Objectness Loss を計算する
            objs_pos = objs[pos_ids]
            objs_neg = objs[neg_ids]
            loss_obj += (1 / N) * F.binary_cross_entropy_with_logits(objs_pos, torch.ones_like(objs_pos), reduction='sum') + \
                (1 / M) * F.binary_cross_entropy_with_logits(objs_neg, torch.zeros_like(objs_neg), reduction='sum')

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
        db_cx = (bboxes[:, 0] - pboxes[:, 0]) / pboxes[:, 2]
        db_cy = (bboxes[:, 1] - pboxes[:, 1]) / pboxes[:, 3]
        db_w = (bboxes[:, 2] / pboxes[:, 2]).log()
        db_h = (bboxes[:, 3] / pboxes[:, 3]).log()

        dbboxes = torch.stack([db_cx, db_cy, db_w, db_h], dim=1).contiguous()
        return dbboxes

    def _calc_coord(self, dbboxes: torch.Tensor, dboxes: torch.Tensor) -> torch.Tensor:
        """ g を算出する

        Args:
            dbboxes (torch.Tensor, [X, 4]): Offset Prediction
            dboxes (torch.Tensor, [X, 4]): Default Box

        Returns:
            torch.Tensor: [X, 4]
        """
        b_cx = dboxes[:, 0] + dbboxes[:, 0] * dboxes[:, 2]
        b_cy = dboxes[:, 1] + dbboxes[:, 1] * dboxes[:, 3]
        b_w = dboxes[:, 2] * dbboxes[:, 2].exp()
        b_h = dboxes[:, 3] * dbboxes[:, 3].exp()

        bboxes = torch.stack([b_cx, b_cy, b_w, b_h], dim=1).contiguous()
        return bboxes

    def pre_predict(self, outputs: tuple):
        """ モデルの出力結果を予測データに変換する

        Args:
            outputs (tuple): モデルの出力. (予測オフセット, 予測信頼度)

        Returns:
            tuple: (予測BBox, 予測信頼度, 予測クラス)
                    - 予測BBox   : [N, 8732, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, 8732]
                    - 予測クラス : [N, 8732]
        """
        out_locs, out_objs, out_confs = outputs
        out_objs = out_objs.sigmoid()
        out_confs = F.softmax(out_confs, dim=-1)
        out_confs = out_confs.mul(out_objs)

        # to CPU
        out_locs = out_locs.detach().cpu()
        out_objs = out_objs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        pred_bboxes = []
        pred_confs = []
        pred_class_ids = []

        for locs, objs, confs in zip(out_locs, out_objs, out_confs):
            confs, class_ids = confs.max(dim=-1)
            class_ids.add_(1)
            pos_ids = objs.gt(0.5).reshape(-1).nonzero().reshape(-1)
            confs, class_ids = confs[pos_ids], class_ids[pos_ids]
            bboxes = self._calc_coord(locs[pos_ids], self.dboxes[pos_ids])
            bboxes = box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')

            pred_bboxes.append(bboxes)
            pred_confs.append(confs)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_confs, pred_class_ids


if __name__ == '__main__':
    from torchsummary import summary
    from models.darknet import Darknet53

    backborn = Darknet53()

    model = YoloV3(10, backborn)
    summary(model, (3, 416, 416))
    x = torch.rand(2, 3, 416, 416)
    out_locs, out_objs, out_confs = model(x)
    print(out_locs.shape)
