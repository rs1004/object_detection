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

        self.pboxes, self.grids = self._get_prior_boxes()

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
        grids = []
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
                    pboxes.append([x, y, w, h])
                    grids.append([x, y, x_max, y_max])

        pboxes = torch.tensor(pboxes).half()
        grids = torch.tensor(grids).half()
        return pboxes, grids

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
        grids = self.grids.to(device)
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
            pboxes_xyxy = box_convert(pboxes, in_fmt='xywh', out_fmt='xyxy')
            ious = box_iou(pboxes_xyxy, bboxes_xyxy)
            bbox_ids = ious.max(dim=1).indices
            bboxes, labels = bboxes[bbox_ids], labels[bbox_ids]

            mask = torch.logical_and(
                torch.logical_and(
                    grids[:, [0]] <= bboxes_xyxy[:, 0],
                    bboxes_xyxy[:, 0] < grids[:, [2]]),
                torch.logical_and(
                    grids[:, [1]] <= bboxes_xyxy[:, 1],
                    bboxes_xyxy[:, 1] < grids[:, [3]]),
            )
            max_ious, pbox_ids = (ious * mask).max(dim=0)
            pos_ids = pbox_ids[(max_ious >= iou_thresh).nonzero().reshape(-1)]
            neg_ids = (ious.max(dim=1).values < iou_thresh).nonzero().reshape(-1)

            if len(pos_ids) == 0:
                continue

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            bboxes_pos = bboxes[pos_ids]
            pboxes_pos = pboxes[pos_ids]
            grids_pos = grids[pos_ids]
            locs_pos = locs[pos_ids]
            locs_pos = self._calc_coord(locs_pos, pboxes_pos, grids_pos)
            loss_loc += F.mse_loss(locs_pos, bboxes_pos, reduction='sum')

            # [Step 3]
            #   Positive Box に対して、Confidence Loss を計算する
            #   labels は 1 開始なので 0 開始に修正する
            labels = labels - 1
            confs_pos = confs[pos_ids]
            labels_pos = F.one_hot(labels[pos_ids], num_classes=self.nc).float()
            loss_conf += F.binary_cross_entropy_with_logits(confs_pos, labels_pos, reduction='sum')

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

    def _calc_coord(self, offsets: torch.Tensor, pboxes: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        """ g を算出する

        Args:
            offsets (torch.Tensor, [X, 4]): Offset Prediction
            pboxes (torch.Tensor, [X, 4]): Prior Box
            grids (torch.Tensor, [X, 4]): Grid

        Returns:
            torch.Tensor: [X, 4]
        """
        b_cx = grids[:, 0] + offsets[:, 0].sigmoid()
        b_cy = grids[:, 1] + offsets[:, 1].sigmoid()
        b_w = pboxes[:, 2] * offsets[:, 2].exp()
        b_h = pboxes[:, 3] * offsets[:, 3].exp()

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
        out_confs = out_confs.sigmoid()
        out_confs = out_confs * out_objs

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
            bboxes = self._calc_coord(locs[pos_ids], self.pboxes[pos_ids])
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
    for name, m in model.named_parameters():
        print(name, len(m.shape))
