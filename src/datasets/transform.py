import datasets.augmentation as A


class DataTransform:
    """ データ変換（データ拡張）
    Args:
        input_size: モデルへの入力画像のサイズ
        mean: データセットの RGB 画素値の平均（標準化に使用）
        std: データセットの RGB 画素値の標準偏差（標準化に使用）
        phase: 'train' or 'val'
    Example:
        >>> transform = DataTransform(224, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), 'train')
        >>> image, label = transform(image, label)
    """

    def __init__(self, input_size: int, mean: tuple, std: tuple, phase: str = 'train'):
        r = 0.5
        p = 0.5
        if phase == 'train':
            self.data_transform = A.Compose([
                A.RandomColorJitter(b_ratio=r, c_ratio=r, s_ratio=r, h_ratio=r),
                A.RandomSampleCrop(p=p),
                A.Resize(input_size=(input_size, input_size)),
                A.RandomMirror(p=p),
                A.ToTensor(),
                A.Normalize(mean, std)
            ])
        elif phase == 'val':
            self.data_transform = A.Compose([
                A.Resize(input_size=(input_size, input_size)),
                A.ToTensor(),
                A.Normalize(mean, std)
            ])
        else:
            raise NotImplementedError(f'phase "{phase}" is invalid')

    def __call__(self, image, bbox, label):
        return self.data_transform(image, bbox, label)


if __name__ == '__main__':
    from datasets.dataset import DetectionDataset
    from PIL import Image, ImageDraw

    size = 300
    norm_cfg = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    ds = DetectionDataset('/home/sato/work/object_detection/data/voc', input_size=size, norm_cfg=norm_cfg, fmt='xyxy', phase='train')
    image, bbox, label = ds.__getitem__(0)
    image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
    draw = ImageDraw.Draw(image)
    for xmin, ymin, xmax, ymax in bbox:
        draw.rectangle((int(xmin * size), int(ymin * size), int(xmax * size), int(ymax * size)), outline=(255, 255, 255), width=3)

    image.save('./demo/transformed.png')
