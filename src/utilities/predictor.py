import torch
import seaborn as sns
from torchvision.ops import box_convert, batched_nms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class Predictor:
    """ 予測を行うクラス

    Args:
        iou_thresh (float, optional): NMS における iou の閾値. Defaults to 0.45.
        classes (dict, optional): クラス ID と名前の辞書. Defaults to None.
        out_dir (str, optional): BBox 描画済み画像の出力先. Defaults to None.

    Methods:
        run: BBox, 信頼度, クラスIDの予測結果から条件を満たすものを抽出し、Coco形式の辞書のリストを作成
             out_dir を指定した場合、BBox 描画済み画像を作成する

    Outputs:
        result (list): 予測結果の辞書のリスト
            [{
                'image_id': 画像ID
                'category_id': クラスID
                'bbox': BBox 座標 (coord fmt: [x, y, w, h])
                'score': 信頼度
            }, {...}, ...]
    """

    def __init__(self, iou_thresh: float = 0.45, classes: dict = None, out_dir: str = None):
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.out_dir = out_dir
        if self.classes:
            colors = [tuple([int(i * 255) for i in c]) for c in sns.color_palette('hls', n_colors=len(classes))]
            self.palette = dict(zip(classes.keys(), colors))

    def run(self, images: torch.Tensor, image_metas: list, pred_bboxes: torch.Tensor, pred_scores: torch.Tensor, pred_class_ids: torch.Tensor) -> list:
        """ 予測結果から条件を満たすものを抽出し、結果の辞書のリストを作成
            予測結果 -> 信頼度でフィルタ -> NMS でフィルタ -> 最終予測結果

        Args:
            images (torch.Tensor): 画像データ [N, 3, H, W]
            image_metas (list): 画像メタデータ
            pred_bboxes (torch.Tensor): 予測 BBox [N, num_preds, 4] (coord fmt: [xmin, ymin, xmax, ymax])
            pred_scores (torch.Tensor): 予測信頼度 [N, num_preds]
            pred_class_ids (torch.Tensor): 予測クラス ID [N, num_preds]

        Returns:
            list: 最終予測結果
        """
        result = []
        for image, image_meta, bboxes, scores, class_ids in zip(images, image_metas, pred_bboxes, pred_scores, pred_class_ids):

            # 重複の除去（non-maximum supression）
            keep = batched_nms(bboxes, scores, class_ids, iou_threshold=self.iou_thresh)
            bboxes = box_convert(bboxes[keep], in_fmt='xyxy', out_fmt='xywh')
            scores = scores[keep]
            class_ids = class_ids[keep]

            H, W = image_meta['height'], image_meta['width']
            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                bbox[[0, 2]] *= W
                bbox[[1, 3]] *= H
                res = {
                    'image_id': image_meta['image_id'],
                    'category_id': int(class_id),
                    'bbox': bbox.numpy().tolist(),
                    'score': float(score),
                }
                result.append(res)

            if self.out_dir:
                mean = torch.tensor(image_meta['norm_mean']).reshape(3, 1, 1)
                std = torch.tensor(image_meta['norm_std']).reshape(3, 1, 1)
                image = image * std + mean
                image = self._to_pil_image(image, size=(W, H))
                image = self._draw_bbox(image, result)
                self._save(image, image_meta['image_id'])

        return result

    def _to_pil_image(self, image_tensor, size=(300, 300)):
        return Image.fromarray((image_tensor.permute(1, 2, 0) * 255).numpy().astype('uint8')).resize(size)

    def _draw_bbox(self, image: Image, result: list) -> Image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype((Path(__file__).parent / 'Roboto-Regular.ttf').as_posix(), size=20)

        for res in result:
            x, y, w, h = res['bbox']
            class_id = res['category_id']
            score = res['score']
            if class_id not in self.palette:
                continue
            color = self.palette[class_id]
            text = f'{self.classes[class_id]}: {round(score, 3)}'

            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
            draw.text((x, y), text, fill=color, font=font)

        return image

    def _save(self, image: Image, image_id: int):
        image.save(f'{self.out_dir}/{image_id:06}.png')
