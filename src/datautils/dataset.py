from torch.utils.data import Dataset
from torchvision.ops import box_convert
from PIL import Image
from pathlib import Path
import torch
try:
    from .transform import DataTransform
except ImportError:
    from transform import DataTransform


class DetectionDataset(Dataset):
    """ 物体検出のデータセット
    Args:
        data_dir (str): 画像データのディレクトリ
        input_size (int): モデルへの画像の入力サイズ (データ拡張で使用)
        phase (str): 'train' or 'val'
    Returns:
        (image, label): image: torch.tensor (3, input_size, input_size)
                        label: torch.tensor (k, [class_id, x, y, w, h])
    """

    def __init__(self, data_dir: str, input_size: int, fmt: str = 'xyxy', phase: str = 'train'):
        super(DetectionDataset, self).__init__()
        self.data_list = self._get_data_list(data_dir, phase)
        self.fmt = fmt
        self.transform = DataTransform(
            input_size=input_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            phase=phase
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        image_path, label_path = self.data_list[i]

        # read image
        image = Image.open(image_path)

        # read label
        with open(label_path, 'r') as f:
            data = torch.tensor([self._parse(line) for line in f.read().split('\n')])
            label, bbox = data[:, 0:1].long(), data[:, 1:]

        # transform
        image, label, bbox = self.transform(image, label, bbox)
        bbox = box_convert(bbox, in_fmt='xyxy', out_fmt=self.fmt)

        return image, label, bbox

    def _get_data_list(self, data_dir: str, phase: str) -> list:
        data_dir = Path(data_dir) / phase
        image_dir = data_dir / 'images'
        label_dir = data_dir / 'labels'
        data_list = [
            (image_dir / f'{p.stem}.jpg', label_dir / f'{p.stem}.txt')
            for p in data_dir.glob('**/*.jpg')]

        return data_list

    def _parse(self, line):
        class_id, x, y, w, h = line.split(' ')
        return int(class_id), float(x), float(y), float(w), float(h)


if __name__ == '__main__':
    ds = DetectionDataset('/home/sato/work/object_detection/data/voc', 100)
    image, label, bbox = ds.__getitem__(0)
    print(image)
    print(label)
    print(bbox)
