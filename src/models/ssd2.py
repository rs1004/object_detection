import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.layers import ConvBlock
from models.base import DetectionNet


class SSD2(DetectionNet):
    def __init__(self, num_classes: int, backborn: nn.Module):
        super(SSD2, self).__init__()
        self.nc = num_classes + 1  # add background class

        self.features = self._trace_features(backborn)

        self.extras = nn.ModuleDict([
            ('conv6_1', ConvBlock(2048, 512, kernel_size=1)),
            ('conv7_1', ConvBlock(512, 1024, kernel_size=1)),

            ('conv8_1', ConvBlock(1024, 256, kernel_size=1)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=1)),

            ('conv9_1', ConvBlock(512, 128, kernel_size=1)),
            ('conv9_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)),

            ('conv10_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3)),

            ('conv11_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv11_2', ConvBlock(128, 256, kernel_size=3)),

            ('conv12_1', ConvBlock(256, 256, kernel_size=1)),
            ('pool12', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv13_1', ConvBlock(256, 256, kernel_size=1)),
            ('pool13', nn.MaxPool2d(kernel_size=2, stride=2)),
        ])

        self.localizers = nn.ModuleDict({
            'layer2': nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
            'layer3': nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
            'pool12': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
            'pool13': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        })

        self.classifiers = nn.ModuleDict({
            'layer2': nn.Conv2d(in_channels=512, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'layer3': nn.Conv2d(in_channels=1024, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'pool12': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'pool13': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
        })

        self.dboxes = self._get_default_boxes()

        self.init_weights(blocks=[self.extras, self.localizers, self.classifiers])

    def forward(self, x):
        batch_size = x.size(0)
        res = {}
        for name, m in list(self.features.items()) + list(self.extras.items()):
            x = m(x)
            if name in self.localizers:
                res[name] = x

        out_locs = []
        out_confs = []
        for name in self.localizers:
            out_locs.append(
                self.localizers[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, 4)
            )
        for name in self.classifiers:
            out_confs.append(
                self.classifiers[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, self.nc)
            )

        out_locs, out_confs = torch.cat(out_locs, dim=1), torch.cat(out_confs, dim=1)
        return out_locs, out_confs

    def _trace_features(self, backborn: nn.Sequential) -> nn.ModuleDict:
        """ torchvision の ResNeXt モデルの特徴抽出層を ModuleDict 化する

        Args:
            backborn (nn.Sequential): feature layer of ResNeXt

        Returns:
            nn.ModuleDict: ResNeXt (conv1 ~ layer4)
        """
        features = nn.ModuleDict()
        for name, m in backborn.named_children():
            if name == 'avgpool':
                break
            features[name] = m

        return features

    def _get_default_boxes(self) -> torch.Tensor:
        """ Default Box を生成する

        Returns:
            torch.Tensor (24676, 4): Default Box (coord fmt: [cx, cy, w, h])
        """
        def s_(k, m=8, s_min=0.1, s_max=0.90):
            return s_min + (s_max - s_min) * (k - 1) / (m - 1)

        dboxes = []
        cfg = [[64, 4], [32, 6], [16, 6], [8, 6], [6, 4], [4, 4], [2, 4], [1, 4]]

        for k, (f_k, num_aspects) in enumerate(cfg, start=1):
            aspects = [1, 2, 1 / 2, 'add'] if num_aspects == 4 else [1, 2, 3, 1 / 2, 1 / 3, 'add']
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

        dboxes = torch.tensor(dboxes).clamp(min=0., max=1.)
        return dboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_thresh: float = 0.5, alpha: float = 1.0) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測信頼度)
                            * 予測オフセット : (B, D, 4) (coord fmt: [Δcx, Δcy, Δw, Δh])
                                    (D: DBoxの数. D = 24676 の想定.)
                            * 予測信頼度     : (B, D, num_classes + 1)
            gt_bboxes (list): 正解BBOX座標 [(G1, 4), (G2, 4), ...] (coord fmt: [cx, cy, w, h])
            gt_labels (list): 正解ラベル [(G1,), (G2,)]
            iou_thresh (float): Potitive / Negative を判定する際の iou の閾値
            alpha (float): loss = loss_conf + α * loss_loc の α

        Returns:
            dict: {
                loss: xxx,
                loss_loc: xxx,
                loss_conf: xxx
            }
        """
        out_locs, out_confs = outputs

        device = out_locs.device
        dboxes = self.dboxes.to(device)
        B = out_locs.size(0)
        loss = loss_loc = loss_conf = 0
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
            dboxes_xyxy = box_convert(dboxes, in_fmt='cxcywh', out_fmt='xyxy')
            max_ious, bbox_ids = box_iou(dboxes_xyxy, bboxes_xyxy).max(dim=1)
            pos_ids, neg_ids = (max_ious >= iou_thresh).nonzero().reshape(-1), (max_ious < iou_thresh).nonzero().reshape(-1)
            N = len(pos_ids)
            if N == 0:
                continue

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            loss_loc += (1 / N) * F.smooth_l1_loss(
                locs[pos_ids],
                self._calc_delta(bboxes=bboxes[bbox_ids[pos_ids]], dboxes=dboxes[pos_ids]),
                reduction='sum'
            )

            # [Step 3]
            #   Positive / Negative Box に対して、Confidence Loss を計算する
            #   - Negative Box の labels は 0 とする
            #   - Negative Box は Loss の上位 len(pos_ids) * 3 個のみを計算に使用する (Hard Negative Mining)
            loss_conf += (1 / N) * (
                F.cross_entropy(confs[pos_ids], labels[bbox_ids[pos_ids]], reduction='sum') +
                F.cross_entropy(confs[neg_ids], torch.zeros_like(labels[bbox_ids[neg_ids]]), reduction='sum')
            )

        # [Step 4]
        #   損失の和を計算する
        loss = loss_conf + alpha * loss_loc

        return {
            'loss': (1 / B) * loss,
            'loss_loc': (1 / B) * alpha * loss_loc,
            'loss_conf': (1 / B) * loss_conf
        }

    def _calc_delta(self, bboxes: torch.Tensor, dboxes: torch.Tensor, std: list = [0.1, 0.2]) -> torch.Tensor:
        """ Δg を算出する

        Args:
            bboxes (torch.Tensor, [X, 4]): GT BBox
            dboxes (torch.Tensor, [X, 4]): Default Box
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Δcx, Δcy, Δw, Δh が標準正規分布に従うようにしている.
                                    第1項が Δcx, Δcy に対する値. 第2項が Δw, Δh に対する値.
                                    Defaults to [0.1, 0.2]. (TODO: 使用するデータに対し調査して設定する必要がある)

        Returns:
            torch.Tensor: [X, 4]
        """
        db_cx = (1 / std[0]) * (bboxes[:, 0] - dboxes[:, 0]) / dboxes[:, 2]
        db_cy = (1 / std[0]) * (bboxes[:, 1] - dboxes[:, 1]) / dboxes[:, 3]
        db_w = (1 / std[1]) * (bboxes[:, 2] / dboxes[:, 2]).log()
        db_h = (1 / std[1]) * (bboxes[:, 3] / dboxes[:, 3]).log()

        dbboxes = torch.stack([db_cx, db_cy, db_w, db_h], dim=1).contiguous()
        return dbboxes

    def _calc_coord(self, dbboxes: torch.Tensor, dboxes: torch.Tensor, std: list = [0.1, 0.2]) -> torch.Tensor:
        """ g を算出する

        Args:
            dbboxes (torch.Tensor, [X, 4]): Offset Prediction
            dboxes (torch.Tensor, [X, 4]): Default Box
            std (list, optional): Δg を全データに対して計算して得られる標準偏差. Defaults to [0.1, 0.2].

        Returns:
            torch.Tensor: [X, 4]
        """
        b_cx = dboxes[:, 0] + std[0] * dbboxes[:, 0] * dboxes[:, 2]
        b_cy = dboxes[:, 1] + std[0] * dbboxes[:, 1] * dboxes[:, 3]
        b_w = dboxes[:, 2] * (std[1] * dbboxes[:, 2]).exp()
        b_h = dboxes[:, 3] * (std[1] * dbboxes[:, 3]).exp()

        bboxes = torch.stack([b_cx, b_cy, b_w, b_h], dim=1).contiguous()
        return bboxes

    def pre_predict(self, outputs: tuple, conf_thresh: float = 0.4) -> tuple:
        """ モデルの出力結果を予測データに変換する

        Args:
            outputs (tuple): モデルの出力. (予測オフセット, 予測信頼度)
            conf_thresh (float): 信頼度の閾値. Defaults to 0.4.

        Returns:
            tuple: (予測BBox, 予測信頼度, 予測クラス)
                    - 予測BBox   : [N, 24676, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, 24676]
                    - 予測クラス : [N, 24676]
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
            bboxes = self._calc_coord(locs[pos_ids], self.dboxes[pos_ids])
            bboxes = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')

            pred_bboxes.append(bboxes)
            pred_confs.append(confs)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_confs, pred_class_ids


if __name__ == '__main__':
    import torch
    from torchvision.models import resnext50_32x4d
    x = torch.rand(2, 3, 512, 512)

    backborn = resnext50_32x4d()
    model = SSD2(num_classes=20, backborn=backborn)
    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)
    print(len(model.dboxes))

    out_locs = torch.rand(4, 24676, 4)
    out_confs = torch.rand(4, 24676, 21)
    outputs = (out_locs, out_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 20, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))

    from PIL import Image, ImageDraw
    from tqdm import tqdm
    images = []
    c = 0
    for cx, cy, w, h in tqdm(model.dboxes * 512):
        c += 1
        if c % 10 == 0:
            image = Image.fromarray(torch.zeros((512, 512, 3)).numpy().astype('uint8'))
            draw = ImageDraw.Draw(image)
            draw.rectangle((int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)), outline=(255, 255, 255), width=2)
            images.append(image.copy())
    images[0].save('./demo/dboxes.gif', save_all=True, append_images=images[1:])
