import albumentations as A
import albumentations.pytorch.transforms as T


class Pipeline:
    """ データ変換（データ拡張）
    Args:
        input_size: モデルへの入力画像のサイズ
        mean: データセットの RGB 画素値の平均（標準化に使用）
        std: データセットの RGB 画素値の標準偏差（標準化に使用）
        phase: 'train' or 'val'
    Example:
        >>> transform = Pipeline(224, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), 'train')
        >>>_,  image, labels = transform_, (image, labels)
    """

    def __init__(self, input_size: int, mean: tuple, std: tuple, phase: str = 'train'):
        if phase == 'train':
            self.data_pipeline = A.Compose([
                A.RandomSizedBBoxSafeCrop(height=input_size, width=input_size, erosion_rate=0.1),
                A.HorizontalFlip(),
                A.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
                A.Normalize(mean=mean, std=std),
                T.ToTensor()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        elif phase == 'val':
            self.data_pipeline = A.Compose([
                A.Resize(height=input_size, width=input_size),
                A.Normalize(mean=mean, std=std),
                T.ToTensor()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        else:
            raise NotImplementedError(f'phase "{phase}" is invalid')

    def __call__(self, **kargs):
        return self.data_pipeline(**kargs)


if __name__ == '__main__':
    from datasets.dataset import DetectionDataset
    from PIL import Image, ImageDraw

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    size = 300
    norm_cfg = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    ds = DetectionDataset('/home/sato/work/object_detection/data/voc', input_size=size, norm_cfg=norm_cfg, fmt='xyxy', phase='train')
    image, _, bboxes, labels = ds.__getitem__(0)
    image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
    draw = ImageDraw.Draw(image)
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, labels):
        draw.rectangle((int(xmin * size), int(ymin * size), int(xmax * size), int(ymax * size)), outline=(255, 255, 255), width=3)
        draw.text((int(xmin * size), int(ymin * size)), classes[int(label)-1])

    image.save('./demo/transformed.png')
