import torch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


class BBoxPainter:
    def __init__(self, classes, save_dir):
        self.classes = classes
        self.save_dir = save_dir
        self.palette = [tuple([int(i * 255) for i in c]) for c in sns.color_palette('hls', n_colors=len(classes))]

    def draw_bbox(self, image: Image, result: list) -> Image:
        """ BBox を描画する

        Args:
            image (Image): 画像データ
            result (list): 予測結果の辞書のリスト
                [{
                    'image_id': 画像ID
                    'category_id': クラスID
                    'bbox': BBox 座標 (fmt: [x, y, w, h])
                    'score': 信頼度
                }]

        Returns:
            Image: BBox 描画済み画像
        """
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('arial.ttf')

        for res in result:
            x, y, w, h = res['bbox']
            class_id = res['category_id']
            score = res['score']
            color = self.palette[class_id]
            text = f'{self.classes[class_id]}: {round(score, 3)}'

            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
            draw.text((x, y), text, fill=color, font=font)

        return image

    def save(self, image: torch.Tensor or Image, file_name: str, imsize: tuple = (600, 400)):
        """ 画像の保存を行う

        Args:
            image (torch.Tensor or Image): 画像データ
            file_name (str): ファイル名
            imsize (tuple, optional): リサイズして保存する. Defaults to (600, 400).
        """
        if isinstance(image, torch.Tensor):
            image = self.to_pil_image(image)

        image = image.resize(imsize)
        image.save(f'{self.save_dir}/{file_name}')

    def to_pil_image(self, image_tensor, size=(300, 300)):
        return Image.fromarray((image_tensor.permute(1, 2, 0) * 255).numpy().astype('uint8')).resize(size)
