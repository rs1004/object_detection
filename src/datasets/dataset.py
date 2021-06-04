from torch.utils.data import Dataset
from torchvision.ops import box_convert
from PIL import Image
from pycocotools.coco import COCO
import torch
import numpy as np
from datasets.pipeline import Pipeline


class DetectionDataset(Dataset):
    """ 物体検出のデータセット

    Args:
        data_dir (str): 画像データのディレクトリ
        pipeline (dict): Augmentation の定義辞書
        fmt (str): bbox のフォーマット. 'xyxy' or 'cxcywh'
        phase (str): 'train' or 'val'

    Returns:
        (image, image_meta, bbox, label): image      : torch.tensor (3, input_size, input_size)
                                          image_meta : dict
                                          bbox       : torch.tensor (k, 4) (coord fmt: [cx, cy, w, h])
                                          label      : torch.tensor (k,)
    """

    def __init__(self, data_dir: str, pipeline: dict, fmt: str = 'cxcywh', phase: str = 'train'):
        super(DetectionDataset, self).__init__()
        self.classes = None
        self.data_list = self._get_data_list(data_dir, phase)
        self.fmt = fmt
        self.transform = Pipeline(pipeline)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        image_path, image_meta, annotation = self.data_list[i]

        # read image
        image = np.array(Image.open(image_path)).astype(np.float32)

        # read meta
        image_meta = {
            'image_id': image_meta['id'],
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
        image, image_meta, bboxes, labels = self.transform(image=image, image_meta=image_meta, bboxes=bboxes, labels=labels)
        bboxes = box_convert(torch.tensor(bboxes), in_fmt='xywh', out_fmt=self.fmt)
        bboxes = bboxes.div(image.size(-1)).float()
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
        anno_path = f'{data_dir}/annotations/instances_{phase}.json'
        cocoGt = COCO(anno_path)
        self.classes = {cat['id']: cat['name'] for cat in cocoGt.loadCats(cocoGt.getCatIds())}

        data_list = [[
            f'{data_dir}/{phase}/{cocoGt.loadImgs(ids=image_id)[0]["file_name"]}',
            cocoGt.loadImgs(ids=image_id)[0],
            cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=image_id))
        ] for image_id in cocoGt.getImgIds()]

        return data_list

    def collate_fn(self, batch: tuple):
        images = []
        image_metas = []
        gt_bboxes = []
        gt_labels = []
        for image, image_meta, bboxes, labels in batch:
            images.append(image)
            image_metas.append(image_meta)
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        images = torch.stack(images, dim=0)

        return images, image_metas, gt_bboxes, gt_labels


if __name__ == '__main__':
    from PIL import ImageDraw

    size = 300
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    pipeline = dict(
        albu=[
            dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
            dict(type='Expand', mean=(0.485*255, 0.456*255, 0.406*255), ratio_range=(1, 4)),
            dict(type='MinIoURandomCrop'),
            dict(type='Resize', height=size, width=size),
            dict(type='HorizontalFlip'),
        ],
        torch=[
            dict(type='ToTensor'),
            dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='Dropout', p=(0.0, 0.1))
        ]
    )
    ds = DetectionDataset('/home/sato/work/object_detection/data/voc07', pipeline)

    images = []
    for _ in range(40):
        image, image_meta, bboxes, labels = ds.__getitem__(17)
        image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
        draw = ImageDraw.Draw(image)
        for (cx, cy, w, h), label in zip(bboxes, labels):
            xmin = cx - w/2
            ymin = cy - h/2
            xmax = cx + w/2
            ymax = cy + h/2
            draw.rectangle((int(xmin * size), int(ymin * size), int(xmax * size), int(ymax * size)), outline=(255, 255, 255), width=3)
            draw.text((int(xmin * size), int(ymin * size)), classes[int(label)-1])

        images.append(image.copy())
    images[0].save('./demo/transformed.gif', save_all=True, append_images=images[1:], duration=1000)
