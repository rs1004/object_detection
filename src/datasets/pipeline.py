import albumentations as A
import torchvision.transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, image_meta):
        for t in self.transforms:
            if isinstance(t, T.Normalize):
                image_meta['norm_mean'] = t.mean
                image_meta['norm_std'] = t.std
            image = t(image)
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

    def __call__(self, image, image_meta, bboxes, labels):
        # albumentations process
        image, image_meta, bboxes, labels = self.albu_pipeline(
            image=image,
            image_meta=image_meta,
            bboxes=bboxes,
            labels=labels
        ).values()

        # torchvision process
        image, image_meta = self.torch_pipeline(image, image_meta)

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
        return Compose(transforms)
