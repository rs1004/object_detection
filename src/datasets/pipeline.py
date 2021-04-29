import albumentations as A
import torchvision.transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = T.Normalize(mean, std)

    def __call__(self, image, image_meta):
        image = self.normalize(image)
        image_meta['norm_mean'] = self.mean
        image_meta['norm_std'] = self.std

        return image, image_meta


class Pipeline:
    """ データ変換（データ拡張）

    Args:
        pipeline: Augmentation の定義辞書

    Example:
        >>> transform = Pipeline(pipeline)
        >>> image, image_meta, bboxes, labels = transform(image, image_meta, bboxes, labels)
    """

    def __init__(self, pipeline: dict):
        self.albu_pipeline = self._build_albu(pipeline['albu'])
        self.torch_pipeline = self._build_torch(pipeline['torch'])
        self.addmeta_pipeline = self._build_addmeta(pipeline['addmeta'])

    def __call__(self, image, image_meta, bboxes, labels):
        # albumentations process
        image, image_meta, bboxes, labels = self.albu_pipeline(
            image=image,
            image_meta=image_meta,
            bboxes=bboxes,
            labels=labels
        ).values()

        # torchvision process
        image = self.torch_pipeline(image)

        # original transform process
        image, image_meta = self.addmeta_pipeline(image, image_meta)

        return image, image_meta, bboxes, labels

    def _build_albu(self, pipe_cfg):
        transforms = []
        for cfg in pipe_cfg:
            transforms.append(eval('A.' + cfg.pop('type'))(**cfg))
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
        )

    def _build_torch(self, pipe_cfg):
        transforms = []
        for cfg in pipe_cfg:
            transforms.append(eval('T.' + cfg.pop('type'))(**cfg))
        return T.Compose(transforms)

    def _build_addmeta(self, pipe_cfg):
        transforms = []
        for cfg in pipe_cfg:
            transforms.append(eval(cfg.pop('type'))(**cfg))
        return Compose(transforms)


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
