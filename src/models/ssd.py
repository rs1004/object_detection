import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.layers import ConvBlock, L2Norm
from models.base import DetectionNet


class SSD(DetectionNet):
    def __init__(self, num_classes: int, backborn: nn.Module):
        super(SSD, self).__init__()
        self.nc = num_classes + 1  # add background class

        self.features = self._trace_features(backborn.features[:-1])

        self.extras = nn.ModuleDict([
            ('conv6_1', ConvBlock(512, 1024, kernel_size=3, padding=6, dilation=6, is_bn=False)),
            ('conv7_1', ConvBlock(1024, 1024, kernel_size=1, is_bn=False)),

            ('conv8_1', ConvBlock(1024, 256, kernel_size=1, is_bn=False)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, is_bn=False)),

            ('conv9_1', ConvBlock(512, 128, kernel_size=1, is_bn=False)),
            ('conv9_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, is_bn=False)),

            ('conv10_1', ConvBlock(256, 128, kernel_size=1, is_bn=False)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3, is_bn=False)),

            ('conv11_1', ConvBlock(256, 128, kernel_size=1, is_bn=False)),
            ('conv11_2', ConvBlock(128, 256, kernel_size=3, is_bn=False)),
        ])

        self.localizers = nn.ModuleDict({
            'conv4_3': nn.Sequential(
                L2Norm(n_channels=512),
                nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1)
            ),
            'conv7_1': nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        })

        self.classifiers = nn.ModuleDict({
            'conv4_3': nn.Sequential(
                L2Norm(n_channels=512),
                nn.Conv2d(in_channels=512, out_channels=4 * self.nc, kernel_size=3, padding=1)
            ),
            'conv7_1': nn.Conv2d(in_channels=1024, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
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

        pred_locs = []
        pred_confs = []
        for name in self.localizers:
            pred_locs.append(
                self.localizers[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, 4)
            )
            pred_confs.append(
                self.classifiers[name](res[name]).permute(0, 2, 3, 1).contiguous(
                ).view(batch_size, -1, self.nc)
            )

        pred_locs, pred_confs = torch.cat(pred_locs, dim=1), torch.cat(pred_confs, dim=1)
        return pred_locs, pred_confs

    def _trace_features(self, vgg_features: nn.Sequential) -> nn.ModuleDict:
        """ torchvision の VGG16 モデルの特徴抽出層を ConvBlock にトレースする

        Args:
            vgg_features (nn.Sequential): features of vgg16

        Returns:
            nn.ModuleDict: ConvBlock の集合. conv1_1 ~ conv5_3 + new pool5
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
        # change pool5 from 2 x 2 - s2 to 3 x 3 - s1
        features[f"pool{l_counter['layer']}"] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        return features

    def _get_default_boxes(self) -> torch.Tensor:
        """ Default Box を生成する

        Returns:
            torch.Tensor (8732, 4): Default Box (coord fmt: [cx, cy, w, h])
        """
        def s_(k, m=6, s_min=0.1, s_max=0.88):
            return s_min + (s_max - s_min) * (k - 1) / (m - 1)

        dboxes = []
        cfg = [[38, 4], [19, 6], [10, 6], [5, 6], [3, 4], [1, 4]]

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

        dboxes = torch.tensor(dboxes)
        return dboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_thresh: float = 0.5, alpha: float = 1.0) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測信頼度)
                            * 予測オフセット : (B, D, 4) (coord fmt: [Δcx, Δcy, Δw, Δh])
                                    (D: DBoxの数. D = 8732 の想定.)
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
        pred_locs, pred_confs = outputs
        device = pred_locs.device
        loss = loss_loc = loss_conf = 0

        # [Step 1]
        #   target を作成する
        #   - Pred を GT に対応させる
        #     - Pred の Default Box との IoU が最大となる BBox, Label
        #   - 最大 IoU が 0.5 未満の場合、Label を 0 に設定する

        B, P, C = pred_confs.size()
        target_locs = torch.zeros(B, P, 4)
        target_labels = torch.zeros(B, P, dtype=torch.long)

        dboxes = self.dboxes
        for i in range(B):
            bboxes = gt_bboxes[i]
            labels = gt_labels[i]

            bboxes_xyxy = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            dboxes_xyxy = box_convert(dboxes, in_fmt='cxcywh', out_fmt='xyxy')
            max_ious, bbox_ids = box_iou(dboxes_xyxy, bboxes_xyxy).max(dim=1)

            bboxes = bboxes[bbox_ids]
            locs = self._calc_delta(bboxes, dboxes)
            labels = labels[bbox_ids]
            labels[max_ious.less(0.5)] = 0  # TODO: gtに対するpriorをpositiveにする操作

            target_locs[i] = locs
            target_labels[i] = labels

        target_locs = target_locs.to(device)
        target_labels = target_labels.to(device)

        # [Step 2]
        #   pos_mask, neg_mask を作成する
        #   - pos_mask: Label が 0 でないもの
        #   - neg_mask: Positive でない、かつ、cross_entropy_loss の上位 3 * (Positive の件数) 以内のもの (Hard Negative Mining)

        pos_mask = target_labels > 0

        loss_neg = F.cross_entropy(pred_confs.view(-1, C), target_labels.view(-1), reduction='none').view(B, -1)
        loss_neg[pos_mask] = 0
        loss_neg_rank = loss_neg.argsort(descending=True).argsort()
        neg_mask = loss_neg_rank < 3 * pos_mask.sum(dim=1, keepdims=True)

        N = pos_mask.sum()
        if N > 0:
            # [Step 3]
            #   Positive に対して、 Localization Loss を計算する
            loss_loc = (1 / N) * F.smooth_l1_loss(pred_locs[pos_mask], target_locs[pos_mask], reduction='mean')

            # [Step 4]
            #   Positive & Negative に対して、Confidence Loss を計算する
            loss_conf = (1 / N) * F.cross_entropy(pred_confs[pos_mask + neg_mask], target_labels[pos_mask + neg_mask], reduction='mean')

            # [Step 5]
            #   損失の和を計算する
            loss = loss_conf + alpha * loss_loc

        return {
            'loss': loss,
            'loss_loc': loss_loc,
            'loss_conf': loss_conf
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
                    - 予測BBox   : [N, 8732, 4] (coord fmt: [xmin, ymin, xmax, ymax], 0 ~ 1)
                    - 予測信頼度 : [N, 8732]
                    - 予測クラス : [N, 8732]
        """
        pred_locs, pred_confs = outputs
        pred_confs = F.softmax(pred_confs, dim=-1)

        # to CPU
        pred_locs = pred_locs.detach().cpu()
        pred_confs = pred_confs.detach().cpu()

        pred_bboxes = []
        pred_confs = []
        pred_class_ids = []

        for locs, confs in zip(pred_locs, pred_confs):
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
    from torchvision.models import vgg16_bn
    x = torch.rand(2, 3, 300, 300)

    backborn = vgg16_bn()
    model = SSD(num_classes=20, backborn=backborn)
    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)
    for coord in model.dboxes:
        print(coord)

    pred_locs = torch.rand(4, 8732, 4)
    pred_confs = torch.rand(4, 8732, 21)
    outputs = (pred_locs, pred_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 20, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))

    from PIL import Image, ImageDraw
    from tqdm import tqdm
    images = []
    for cx, cy, w, h in tqdm(model.dboxes * 300):
        image = Image.fromarray(torch.zeros((300, 300, 3)).numpy().astype('uint8'))
        draw = ImageDraw.Draw(image)
        draw.rectangle((int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)), outline=(255, 255, 255), width=2)
        images.append(image.copy())
    images[0].save('./demo/dboxes.gif', save_all=True, append_images=images[1:])
    images[0].save('./demo/dboxes_fast.gif', save_all=True, append_images=images[::12])
