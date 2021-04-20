from torch.utils.data import Dataset
from torchvision.ops import box_convert
from PIL import Image
from pathlib import Path
import torch
import json
from datasets.transform import DataTransform


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
        self.annotations = self._get_annotations(data_dir, phase)
        self.fmt = fmt
        self.transform = DataTransform(
            input_size=input_size,
            mean=norm_cfg['mean'],
            std=norm_cfg['std'],
            phase=phase
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        file_name, image_path = self.data_list[i]

        # read image
        image = Image.open(image_path)

        # read label
        anno = self.annotations[file_name]
        H, W = anno['image']['height'], anno['image']['width']
        label = []
        bbox = []
        for a in anno['annotation']:
            label.append(a['category_id'])
            x, y, w, h = a['bbox']
            bbox.append([x / W, y / H, (x + w) / W, (y + h) / H])
        label, bbox = torch.tensor(label), torch.tensor(bbox)

        # transform
        image, bbox, label = self.transform(image, bbox, label)
        bbox = box_convert(bbox, in_fmt='xyxy', out_fmt=self.fmt)

        return image, bbox, label

    def _get_data_list(self, data_dir: str, phase: str) -> list:
        """ データリスト（画像のパスリスト）を作成する

        Args:
            data_dir (str): 画像データのディレクトリ
            phase (str): 'train' or 'val'

        Returns:
            list: 画像のパスリスト
        """
        data_dir = Path(data_dir) / phase
        data_list = [(p.name, p.resolve()) for p in data_dir.glob('**/*.jpg')]

        return data_list

    def _get_annotations(self, data_dir: str, phase: str) -> dict:
        """ アノテーションファイルを読み込み、辞書を作成する

        Args:
            data_dir (str): 画像データのディレクトリ
            phase (str): 'train' or 'val'

        Returns:
            dict: アノテーションの辞書
        """
        anno_dir = Path(data_dir) / 'annotations'
        with open(anno_dir / f'instances_{phase}.json', 'r') as f:
            instances = json.load(f)

        annotations = {}
        id2filename = {}
        for image in instances['images']:
            annotations[image['file_name']] = {
                'image': {
                    'width': image['width'],
                    'height': image['height']
                }
            }
            id2filename[image['id']] = image['file_name']
        for annotation in instances['annotations']:
            file_name = id2filename[annotation['image_id']]
            if 'annotation' not in annotations[file_name]:
                annotations[file_name]['annotation'] = []
            annotations[file_name]['annotation'].append({
                'category_id': annotation['category_id'],
                'bbox': annotation['bbox'],
            })

        return annotations

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
    image, bbox, label = ds.__getitem__(0)
    print(image)
    print(bbox)
    print(label)
