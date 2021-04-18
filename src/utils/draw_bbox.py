import torch
import seaborn as sns
from PIL import Image, ImageDraw
from pathlib import Path


class BBoxPainter:
    def __init__(self, classes, save_dir):
        self.classes = classes
        self.save_dir = save_dir
        self.palette = [tuple([int(i * 255) for i in c]) for c in sns.color_palette('hls', n_colors=len(classes))]

    def draw_bbox(self, image: torch.Tensor or Image, coord: torch.Tensor, class_id: torch.Tensor, conf: torch.Tensor) -> Image:
        """[summary]

        Args:
            image (torch.TensororImage): [description]
            coord (torch.Tensor): [description]
            class_id (torch.Tensor): [description]
            conf (torch.Tensor): [description]

        Returns:
            Image: [description]
        """
        if isinstance(image, torch.Tensor):
            image = self._to_pil_image(image)

        xmin, ymin, xmax, ymax = coord
        color = self.palette[int(class_id)]
        text = f'{self.classes[int(class_id)]}: {round(float(conf), 3)}'

        draw = ImageDraw.Draw(image)
        draw.rectangle((int(xmin), int(ymin), int(xmax), int(ymax)), outline=color, width=2)
        draw.text((xmin, ymin), text, fill=color)

        return image

    def save(self, image: Image, file_name: str, imsize: tuple = (600, 400)):
        if isinstance(image, torch.Tensor):
            image = self._to_pil_image(image)

        image = image.resize(imsize)
        image.save(Path(self.save_dir) / file_name)

    def _to_pil_image(self, image_tensor):
        return Image.fromarray((image_tensor.permute(1, 2, 0) * 255).numpy().astype('uint8'))
