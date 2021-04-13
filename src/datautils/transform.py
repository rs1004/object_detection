import augmentation as A


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
                A.Resize(input_size=(input_size, input_size)),
                A.RandomMirror(p=p),
                A.RandomRotate(degree=5.0, p=p),
                A.ToTensor(),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif phase == 'val':
            self.data_transform = A.Compose([
                A.Resize(input_size=(input_size, input_size)),
                A.ToTensor(),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError(f'phase "{phase}" is invalid')

    def __call__(self, image, label):
        return self.data_transform(image, label)


if __name__ == '__main__':
    from datautils.dataset import DetectionDataset
    from PIL import Image, ImageDraw

    size = 300
    ds = DetectionDataset('/home/sato/work/object_detection/data/voc', input_size=size, phase='train')
    image, label = ds.__getitem__(0)
    image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
    draw = ImageDraw.Draw(image)
    for _, x, y, w, h in label:
        draw.rectangle((int(x * size), int(y * size), int((x + w) * size), int((y + h) * size)), outline=(255, 255, 255), width=3)

    image.save('a.png')
