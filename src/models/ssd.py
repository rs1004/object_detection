import torch
import torch.nn as nn
import json
from torchvision.models import vgg16_bn
from collections import Counter
from itertools import product
from torchvision.ops import box_iou, box_convert, batched_nms
from models.layers import ConvBlock, L2Norm


class SSD(nn.Module):
    def __init__(self, num_classes, pretrained=True, pretrained_weights=None):
        super(SSD, self).__init__()
        self.nc = num_classes + 1  # add background class

        if pretrained_weights:
            bb = vgg16_bn(pretrained=False)
            bb.load_state_dict(torch.load(pretrained_weights))
        else:
            bb = vgg16_bn(pretrained=pretrained)

        self.features = self._parse_features(bb.features[:-1])

        self.extras = nn.ModuleDict([
            ('conv6_1', ConvBlock(512, 1024, kernel_size=3, padding=1)),
            ('conv7_1', ConvBlock(1024, 1024, kernel_size=1)),

            ('conv8_1', ConvBlock(1024, 256, kernel_size=1)),
            ('conv8_2', ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)),

            ('conv9_1', ConvBlock(512, 128, kernel_size=1)),
            ('conv9_2', ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)),

            ('conv10_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv10_2', ConvBlock(128, 256, kernel_size=3)),

            ('conv11_1', ConvBlock(256, 128, kernel_size=1)),
            ('conv11_2', ConvBlock(128, 256, kernel_size=3)),
        ])

        self.localizers = nn.ModuleDict({
            'conv4_3': nn.Sequential(
                L2Norm(512),
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
                L2Norm(512),
                nn.Conv2d(in_channels=512, out_channels=4 * self.nc, kernel_size=3, padding=1)
            ),
            'conv7_1': nn.Conv2d(in_channels=1024, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv8_2': nn.Conv2d(in_channels=512, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv9_2': nn.Conv2d(in_channels=256, out_channels=6 * self.nc, kernel_size=3, padding=1),
            'conv10_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
            'conv11_2': nn.Conv2d(in_channels=256, out_channels=4 * self.nc, kernel_size=3, padding=1),
        })

        self.dboxes = self._get_dboxes()

    def forward(self, x):
        batch_size = x.size(0)
        out_locs = []
        out_confs = []
        for name, m in self.features.items():
            x = m(x)
            if name in self.localizers:
                out_locs.append(self.localizers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
                out_confs.append(self.classifiers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))

        for name, m in self.extras.items():
            x = m(x)
            if name in self.localizers:
                out_locs.append(self.localizers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
                out_confs.append(self.classifiers[name](x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc))

        out_locs, out_confs = torch.cat(out_locs, dim=1), torch.cat(out_confs, dim=1)
        return out_locs, out_confs

    def _parse_features(self, vgg_features: nn.Sequential) -> nn.ModuleDict:
        """ torchvision の VGG16 モデルの特徴抽出層を ConvBlock にパースする

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

    def _get_dboxes(self) -> torch.Tensor:
        """ Default Box を生成する

        Returns:
            torch.Tensor (8732, 4): Default Box (fmt: [cx, cy, w, h])
        """
        def s_(k, m=6, s_min=0.2, s_max=0.9):
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

        dboxes = torch.tensor(dboxes).clamp(min=0.0, max=1.0)
        return dboxes

    def loss(self, outputs: tuple, gt_bboxes: list, gt_labels: list, iou_thresh: float = 0.5, alpha: float = 1.0) -> dict:
        """ 損失関数

        Args:
            outputs (tuple): (予測オフセット, 予測信頼度)
                            * 予測オフセット : (B, D, 4) (fmt: [Δcx, Δcy, Δw, Δh])
                                    (D: DBoxの数. D = 8732 の想定.)
                            * 予測信頼度     : (B, D, num_classes + 1)
            gt_bboxes (list): 正解BBOX座標 [(G1, 4), (G2, 4), ...] (fmt: [cx, cy, w, h])
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
        for out_loc, out_conf, bboxes, labels in zip(out_locs, out_confs, gt_bboxes, gt_labels):
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
            max_ious, indices = box_iou(dboxes_xyxy, bboxes_xyxy).max(dim=1)
            pos_ids, neg_ids = (max_ious >= iou_thresh).nonzero().reshape(-1), (max_ious < iou_thresh).nonzero().reshape(-1)
            N = len(pos_ids)
            if N == 0:
                continue

            # [Step 2]
            #   Positive Box に対して、 Localization Loss を計算する
            bboxes_pos = bboxes[indices[pos_ids]]
            dboxes_pos = dboxes[pos_ids]
            dbboxes_pos = self._calc_delta(bboxes=bboxes_pos, dboxes=dboxes_pos)
            loss_loc += (1 / N) * self._smooth_l1(out_loc[pos_ids] - dbboxes_pos)

            # [Step 3]
            #   Positive / Negative Box に対して、Confidence Loss を計算する
            #   - Negative Box の labels は 0 とする
            #   - Negative Box は Loss の上位 len(pos_ids) * 3 個のみを計算に使用する (Hard Negative Mining)
            labels = labels[indices]
            labels[neg_ids] = 0
            sce = self._softmax_cross_entropy(out_conf, labels)
            loss_conf += (1 / N) * (sce[pos_ids].sum() + sce[neg_ids].topk(k=int(N * 3)).values.sum())

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

    def _smooth_l1(self, x: torch.Tensor) -> torch.Tensor:
        if x.abs().sum() < 1:
            return (0.5 * x ** 2).sum()
        else:
            return (x.abs() - 0.5).sum()

    def _softmax_cross_entropy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -nn.functional.log_softmax(pred, dim=-1)[range(len(target)), target]

    def get_parameters(self, lrs: dict = {'features': 0.0001, '_': 0.001}) -> list:
        """ 学習パラメータと学習率の一覧を取得する

        Args:
            lrs (dict, optional): 学習率の一覧. Defaults to {'features': 0.0001, '_': 0.001}.

        Returns:
            list: 学習パラメータと学習率の一覧
        """
        params_to_update = {key: [] for key in lrs.keys()}

        for name, param in self.named_parameters():
            for key in sorted(lrs.keys(), reverse=True):
                if key in name or key == '_':
                    if lrs[key] > 0:
                        params_to_update[key].append(param)
                    else:
                        param.requires_grad = False
                    break

        if lrs['_'] == 0:
            del lrs['_']
        params = [{'params': params_to_update[key], 'lr': lrs[key]} for key in lrs.keys()]

        return params

    def inference(self, images: torch.Tensor, outputs: tuple, num_done: int, norm_cfg: dict,
                  bbox_painter, top_k: int = 30, iou_thresh: float = 0.45) -> int:
        out_locs, out_confs = outputs
        out_confs = nn.functional.softmax(out_confs, dim=-1)
        H, W = images.shape[2:]

        # De Normalize
        device = images.device
        mean = torch.tensor(norm_cfg['mean']).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor(norm_cfg['std']).reshape(1, 3, 1, 1).to(device)
        images = images * std + mean

        # to CPU
        images = images.detach().cpu()
        out_locs = out_locs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        for image, out_loc, out_conf in zip(images, out_locs, out_confs):

            # 座標・クラスの復元
            confs, class_ids = out_conf[:, 1:].max(dim=-1)  # 0 is background class
            valid_ids = confs.topk(k=top_k).indices
            bboxes_valid = self._calc_coord(dbboxes=out_loc[valid_ids], dboxes=self.dboxes[valid_ids])
            bboxes_valid = box_convert(bboxes_valid, in_fmt='cxcywh', out_fmt='xyxy') * torch.tensor([W, H, W, H])
            class_ids_valid = class_ids[valid_ids]
            confs_valid = confs[valid_ids]

            # 重複の除去（non-maximum supression）
            keep = batched_nms(bboxes_valid, confs_valid, class_ids_valid, iou_threshold=iou_thresh)
            bboxes_valid = bboxes_valid[keep]
            confs_valid = confs_valid[keep]
            class_ids_valid = class_ids_valid[keep]

            for bbox, class_id, conf in zip(bboxes_valid, class_ids_valid, confs_valid):
                image = bbox_painter.draw_bbox(
                    image=image,
                    coord=bbox,
                    class_id=class_id,
                    conf=conf
                )

            bbox_painter.save(image, file_name=f'{num_done:06}.png')
            num_done += 1

        return num_done

    def detect(self, outputs: tuple, image_metas: list, output_dir, conf_thresh: float = 0.4, iou_thresh: float = 0.45):
        out_locs, out_confs = outputs
        out_confs = nn.functional.softmax(out_confs, dim=-1)

        # to CPU
        out_locs = out_locs.detach().cpu()
        out_confs = out_confs.detach().cpu()

        for image_meta, out_loc, out_conf in zip(image_metas, out_locs, out_confs):

            # 座標・クラスの復元
            H, W = image_meta['height'], image_meta['width']
            confs, class_ids = out_conf[:, 1:].max(dim=-1)  # 0 is background class
            valid_ids = confs.gt(conf_thresh).nonzero().reshape(-1)
            bboxes_valid = self._calc_coord(dbboxes=out_loc[valid_ids], dboxes=self.dboxes[valid_ids])
            bboxes_valid = box_convert(bboxes_valid, in_fmt='cxcywh', out_fmt='xyxy') * torch.tensor([W, H, W, H])
            class_ids_valid = class_ids[valid_ids]
            confs_valid = confs[valid_ids]

            # 重複の除去（non-maximum supression）
            keep = batched_nms(bboxes_valid, confs_valid, class_ids_valid, iou_threshold=iou_thresh)
            bboxes_valid = box_convert(bboxes_valid[keep], in_fmt='xyxy', out_fmt='xywh')
            confs_valid = confs_valid[keep]
            class_ids_valid = class_ids_valid[keep]

            result = []
            for bbox, class_id, conf in zip(bboxes_valid, class_ids_valid, confs_valid):
                res = {
                    'image_id': image_meta['image_id'],
                    'category_id': int(class_id),
                    'bbox': bbox.numpy().tolist(),
                    'score': float(conf),
                }
                result.append(res)

            with open(output_dir / f"{image_meta['image_id']:08}.json", 'w') as f:
                json.dump(result, f)

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


if __name__ == '__main__':
    import torch
    x = torch.rand(2, 3, 300, 300)

    model = SSD(num_classes=20, pretrained=False)
    outputs = model(x)
    print(outputs[0].shape, outputs[1].shape)
    for coord in model.dboxes:
        print(coord)

    out_locs = torch.rand(4, 8732, 4)
    out_confs = torch.rand(4, 8732, 21)
    outputs = (out_locs, out_confs)
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
