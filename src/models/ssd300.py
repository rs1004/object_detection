import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torchvision.ops import box_iou, box_convert
from models.base import DetectionNet


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        out = self.act(x)
        return out


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.act(x)
        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD300(DetectionNet):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(SSD300, self).__init__()
        self.nc = num_classes + 1  # add background class

        if any(isinstance(m, nn.BatchNorm2d) for m in backbone.features):
            ConvBlock = ConvBNRelu
        else:
            ConvBlock = ConvRelu

        self.features = self._build_features(backbone, ConvBlock)
        self.extras = nn.ModuleDict([
            ('conv8_1', ConvBlock(1024, 256, kernel_size=1)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)),
            ('conv9_1', ConvBlock(512, 128, kernel_size=1)),
            ('conv9_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)),
            ('conv10_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3)),
            ('conv11_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv11_2', ConvBlock(128, 256, kernel_size=3)),
        ])

        self.l2norm = L2Norm(512)

        self.localizers = nn.ModuleDict({
            'conv4_3': nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            'conv7_1': nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        })

        self.classifiers = nn.ModuleDict({
            'conv4_3': nn.Conv2d(512, 4 * self.nc, kernel_size=3, padding=1),
            'conv7_1': nn.Conv2d(1024, 6 * self.nc, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(512, 6 * self.nc, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(256, 6 * self.nc, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(256, 4 * self.nc, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(256, 4 * self.nc, kernel_size=3, padding=1),
        })

        self.dboxes = self._get_default_boxes()

        self.init_weights(blocks=[self.extras, self.localizers, self.classifiers])

    def forward(self, x):
        B = x.size(0)
        res = {}
        for name, m in list(self.features.items()) + list(self.extras.items()):
            x = m(x)
            if name in self.localizers:
                if name == 'conv4_3':
                    res[name] = self.l2norm(x)
                else:
                    res[name] = x

        out_locs = []
        out_confs = []
        for name in self.localizers:
            out_locs.append(
                self.localizers[name](res[name]).permute(0, 2, 3, 1).reshape(B, -1, 4)
            )
            out_confs.append(
                self.classifiers[name](res[name]).permute(0, 2, 3, 1).reshape(B, -1, self.nc)
            )

        out_locs, out_confs = torch.cat(out_locs, dim=1), torch.cat(out_confs, dim=1)
        return out_locs, out_confs

    def _build_features(self, backbone: nn.Module, ConvBlock: nn.Module) -> nn.ModuleDict:
        """ 特徴抽出層を作成

        Args:
            backbone (nn.Module): vgg16 or vgg16_bn
            ConvBlock (nn.Module): conv block

        Returns:
            nn.ModuleDict: ConvBlock の集合. conv1_1 ~ conv5_3 + new pool5 + conv6_1, conv7_1
        """
        layer_num = 1
        block_num = 1
        features = nn.ModuleDict()
        for m in backbone.features:
            if isinstance(m, nn.Conv2d):
                block = ConvBlock(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation
                )

                block.conv.load_state_dict(m.state_dict())

            elif isinstance(m, nn.BatchNorm2d):
                block.bn.load_state_dict(m.state_dict())

            elif isinstance(m, nn.ReLU):
                features[f'conv{layer_num}_{block_num}'] = block
                block_num += 1

            elif isinstance(m, nn.MaxPool2d):
                if layer_num < 5:
                    m.ceil_mode = True
                    features[f'pool{layer_num}'] = m
                else:
                    features[f'pool{layer_num}'] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
                layer_num += 1
                block_num = 1

        # linear layer -> conv layer (subsample weight)
        features['conv6_1'] = ConvBlock(512, 1024, kernel_size=3, padding=6, dilation=6)
        features['conv7_1'] = ConvBlock(1024, 1024, kernel_size=1)

        bc = backbone.classifier
        state_dict = {
            'conv6_1.conv.weight': bc[0].weight.reshape(4096, 512, 7, 7)[::4, :, ::3, ::3],
            'conv6_1.conv.bias': bc[0].bias[::4],
            'conv7_1.conv.weight': bc[3].weight.reshape(4096, 4096, 1, 1)[::4, ::4],
            'conv7_1.conv.bias':  bc[3].bias[::4]
        }

        features.load_state_dict(state_dict, strict=False)

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
            aspects = [1, 'add', 2, 1 / 2] if num_aspects == 4 else [1, 'add', 2, 1 / 2, 3, 1 / 3]
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

        dboxes = torch.tensor(dboxes).clamp(min=0, max=1)
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
        out_locs, out_confs = outputs
        device = out_locs.device

        # [Step 1]
        #   target を作成する
        #   - Pred を GT に対応させる
        #     - Pred の Default Box との IoU が最大となる BBox, Label
        #     - BBox との IoU が最大となる Default Box -> その BBox に割り当てる
        #   - 最大 IoU が 0.5 未満の場合、Label を 0 に設定する

        B, P, C = out_confs.size()
        target_locs = torch.zeros(B, P, 4, device=device)
        target_labels = torch.zeros(B, P, dtype=torch.long, device=device)

        dboxes = self.dboxes.to(device)
        for i in range(B):
            bboxes = gt_bboxes[i].to(device)
            labels = gt_labels[i].to(device)

            bboxes_xyxy = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            dboxes_xyxy = box_convert(dboxes, in_fmt='cxcywh', out_fmt='xyxy')
            ious = box_iou(dboxes_xyxy, bboxes_xyxy)
            best_ious, best_dbox_ids = ious.max(dim=0)
            max_ious, matched_bbox_ids = ious.max(dim=1)

            # 各 BBox に対し最大 IoU を取る Default Box を選ぶ -> その BBox に割り当てる
            for j in range(len(best_dbox_ids)):
                matched_bbox_ids[best_dbox_ids][j] = j
            max_ious[best_dbox_ids] = iou_thresh

            bboxes = bboxes[matched_bbox_ids]
            locs = self._calc_delta(bboxes, dboxes)
            labels = labels[matched_bbox_ids]
            labels[max_ious.less(iou_thresh)] = 0  # 0 が背景クラス. Positive Class は 1 ~

            target_locs[i] = locs
            target_labels[i] = labels

        # [Step 2]
        #   pos_mask, neg_mask を作成する
        #   - pos_mask: Label が 0 でないもの
        #   - neg_mask: Positive でない、かつ、cross_entropy_loss の上位 3 * (Positive の件数) 以内のもの (Hard Negative Mining)

        pos_mask = target_labels > 0

        loss_neg = F.cross_entropy(out_confs.view(-1, C), target_labels.view(-1), reduction='none').view(B, -1)
        loss_neg[pos_mask] = 0
        loss_neg_rank = loss_neg.argsort(descending=True).argsort()
        neg_mask = loss_neg_rank < 3 * pos_mask.sum(dim=1, keepdims=True)

        N = pos_mask.sum()
        # [Step 3]
        #   Positive に対して、 Localization Loss を計算する
        loss_loc = F.smooth_l1_loss(out_locs[pos_mask], target_locs[pos_mask], reduction='sum') / N

        # [Step 4]
        #   Positive & Negative に対して、Confidence Loss を計算する
        loss_conf = F.cross_entropy(out_confs[pos_mask + neg_mask], target_labels[pos_mask + neg_mask], reduction='sum') / N

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

        dbboxes = torch.stack([db_cx, db_cy, db_w, db_h], dim=1)
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

        bboxes = torch.stack([b_cx, b_cy, b_w, b_h], dim=1)
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
                bboxes_ = self._calc_coord(locs[pos_mask], self.dboxes[pos_mask])
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
    from torchvision.models import vgg16
    x = torch.rand(2, 3, 300, 300)

    backbone = vgg16(pretrained=True)
    model = SSD300(num_classes=20, backbone=backbone)
    print(model)

    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)

    out_locs = torch.rand(4, 8732, 4)
    out_confs = torch.rand(4, 8732, 21)
    outputs = (out_locs, out_confs)
    gt_bboxes = [torch.rand(5, 4) for _ in range(4)]
    gt_labels = [torch.randint(0, 21, (5,)) for _ in range(4)]

    print(model.loss(outputs, gt_bboxes, gt_labels))
