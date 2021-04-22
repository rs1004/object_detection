from torch.utils.data import Dataset
from torchvision.ops import box_convert
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
import torch
import numpy as np
from datasets.pipeline import Pipeline


class DetectionDataset(Dataset):
    """ 物体検出のデータセット
    Args:
        data_dir (str): 画像データのディレクトリ
        input_size (int): モデルへの画像の入力サイズ (データ拡張で使用)
        norm_cfg (dict): 画像の標準化設定値（データ拡張で使用）
        fmt (str): bbox のフォーマット. 'xyxy' or 'cxcywh'
        phase (str): 'train' or 'val'
    Returns:
        (image, bbox, label): image: torch.tensor (3, input_size, input_size)
                              bbox : torch.tensor (k, 4) (fmt: [cx, cy, w, h])
                              label: torch.tensor (k,)
    """

    def __init__(self, data_dir: str, input_size: int, norm_cfg: dict, fmt: str = 'cxcywh', phase: str = 'train'):
        super(DetectionDataset, self).__init__()
        self.data_list = self._get_data_list(data_dir, phase)
        self.input_size = input_size
        self.fmt = fmt
        self.transform = Pipeline(
            input_size=input_size,
            mean=norm_cfg['mean'],
            std=norm_cfg['std'],
            phase=phase
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        image_path, image_meta, annotation = self.data_list[i]

        # read image
        image = np.array(Image.open(image_path))

        # read meta
        image_meta = {
            'file_path': image_path,
            'height': image_meta['height'],
            'width': image_meta['width']
        }

        # read bboxes & labels
        bboxes = []
        labels = []
        for anno in annotation:
            bboxes.append(anno['bbox'])
            labels.append(anno['category_id'])

        # transform
        augmented = self.transform(image=image, image_meta=image_meta, bboxes=bboxes, labels=labels)
        image, image_meta, bboxes, labels = augmented['image'], augmented['image_meta'], augmented['bboxes'], augmented['labels']
        bboxes = box_convert(torch.tensor(bboxes), in_fmt='xywh', out_fmt=self.fmt)
        bboxes /= self.input_size
        labels = torch.tensor(labels)

        return image, image_meta, bboxes, labels

    def _get_data_list(self, data_dir: str, phase: str) -> list:
        """ データリスト（画像のパスリスト）を作成する

        Args:
            data_dir (str): 画像データのディレクトリ
            phase (str): 'train' or 'val'

        Returns:
            list: (画像のパス, 画像メタ情報, アノテーション）のリスト
        """
        anno_path = Path(data_dir) / 'annotations' / f'instances_{phase}.json'
        cocoGt = COCO(anno_path)

        data_list = [[
            (Path(data_dir) / phase / cocoGt.loadImgs(ids=image_id)[0]['file_name']).as_posix(),
            cocoGt.loadImgs(ids=image_id)[0],
            cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=image_id))
        ] for image_id in cocoGt.getImgIds()]

        return data_list

    def collate_fn(self, batch: tuple):
        images = []
        bboxes = []
        labels = []
        for image, bbox, label in batch:
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)
        images = torch.stack(images, dim=0)

        return images, bboxes, labels


if __name__ == '__main__':
    norm_cfg = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    ds = DetectionDataset('/home/sato/work/object_detection/data/voc', 100, norm_cfg)
    image, image_meta, bboxes, labels = ds.__getitem__(0)
    print(image)
    print(image_meta)
    print(bboxes)
    print(labels)
