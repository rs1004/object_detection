import torch
import torchvision.transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, bbox):
        for t in self.transforms:
            image, label, bbox = t(image, label, bbox)

        return image, label, bbox


class RandomColorJitter:
    def __init__(self, b_ratio=0, c_ratio=0, s_ratio=0, h_ratio=0):
        self.color_jitter = T.ColorJitter(
            brightness=b_ratio,
            contrast=c_ratio,
            saturation=s_ratio,
            hue=h_ratio
        )

    def __call__(self, image, label, bbox):
        image = self.color_jitter(image)
        return image, label, bbox


class Resize:
    def __init__(self, input_size):
        self.resize = T.Resize(input_size)

    def __call__(self, image, label, bbox):
        image = self.resize(image)
        return image, label, bbox


class RandomRotate:
    def __init__(self, degree, p=0.5):
        self.degree = self._get_uniformally(degree)
        self.p = p

    def _get_uniformally(self, x):
        return (torch.rand(1) * 2 - 1) * x

    def __call__(self, image, label, bbox):
        if torch.rand(1) < self.p:
            image = T.functional.rotate(image, float(self.degree), interpolation=T.functional.InterpolationMode.BICUBIC)
            bbox_ = []
            for xmin, ymin, xmax, ymax in bbox:
                corner_points = torch.tensor(
                    [[xmin, xmin, xmax, xmax],
                     [ymin, ymax, ymin, ymax]]
                ) - 0.5  # lt, lb, rt, rb
                rcp = self._rotate_p(corner_points, self.degree) + 0.5
                xmin, xmax, ymin, ymax = rcp[0].min(), rcp[0].max(), rcp[1].min(), rcp[1].max()
                bbox_.append([xmin, ymin, xmax, ymax])
            bbox = torch.tensor(bbox_)

        return image, label, bbox

    def _rotate_p(self, p, degree):
        theta = torch.deg2rad(degree)
        rot_mat = torch.tensor([
            [torch.cos(theta), torch.sin(theta)],
            [-torch.sin(theta), torch.cos(theta)],
        ])

        return torch.mm(rot_mat, p)


class RandomMirror:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, bbox):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            bbox[:, 0], bbox[:, 2] = 1 - bbox[:, 2], 1 - bbox[:, 0]

        return image, label, bbox


class ToTensor:
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, image, label, bbox):
        image = self.to_tensor(image)
        return image, label, bbox


class Normalize:
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image, label, bbox):
        image = self.normalize(image)
        return image, label, bbox
