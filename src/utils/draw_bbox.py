import torch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class BBoxPainter:
    def __init__(self, classes, save_dir):
        self.classes = classes
        self.save_dir = save_dir
        self.palette = [tuple([int(i * 255) for i in c]) for c in sns.color_palette('hls', n_colors=len(classes))]

    def draw_bbox(self, image: torch.Tensor or Image, coord: torch.Tensor, class_id: torch.Tensor, conf: torch.Tensor) -> Image:
        """ BBox を描画する

        Args:
            image (torch.Tensor or Image): 画像データ
            coord (torch.Tensor): BBox 座標データ（fmt: [xmin, ymin, xmax, ymax], pixel 座標）
            class_id (torch.Tensor): BBox クラスデータ
            conf (torch.Tensor): BBox 信頼度

        Returns:
            Image: [description]
        """
        if isinstance(image, torch.Tensor):
            image = self._to_pil_image(image)

        xmin, ymin, xmax, ymax = coord
        color = self.palette[int(class_id)]
        text = f'{self.classes[int(class_id)]}: {round(float(conf), 3)}'
        font = ImageFont.truetype((Path(__file__).parent / 'Gargi.ttf').as_posix())

        draw = ImageDraw.Draw(image)
        draw.rectangle((int(xmin), int(ymin), int(xmax), int(ymax)), outline=color, width=2)
        draw.text((xmin, ymin), text, fill=color, font=font)

        return image

    def save(self, image: torch.Tensor or Image, file_name: str, imsize: tuple = (600, 400)):
        """ 画像の保存を行う

        Args:
            image (torch.Tensor or Image): 画像データ
            file_name (str): ファイル名
            imsize (tuple, optional): リサイズして保存する. Defaults to (600, 400).
        """
        if isinstance(image, torch.Tensor):
            image = self._to_pil_image(image)

        image = image.resize(imsize)
        image.save(Path(self.save_dir) / file_name)

    def _to_pil_image(self, image_tensor):
        return Image.fromarray((image_tensor.permute(1, 2, 0) * 255).numpy().astype('uint8'))
