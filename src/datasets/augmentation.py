import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import box_area


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox, label):
        for t in self.transforms:
            image, bbox, label = t(image, bbox, label)

        return image, bbox, label


class RandomColorJitter:
    def __init__(self, b_ratio=0, c_ratio=0, s_ratio=0, h_ratio=0):
        self.color_jitter = T.ColorJitter(
            brightness=b_ratio,
            contrast=c_ratio,
            saturation=s_ratio,
            hue=h_ratio
        )

    def __call__(self, image, bbox, label):
        image = self.color_jitter(image)
        return image, bbox, label


class Resize:
    def __init__(self, input_size):
        self.resize = T.Resize(input_size)

    def __call__(self, image, bbox, label):
        image = self.resize(image)
        return image, bbox, label


class RandomRotate:
    def __init__(self, degree=5.0, p=0.5):
        self.degree = self._get_uniformally(degree)
        self.p = p

    def _get_uniformally(self, x):
        return (torch.rand(1) * 2 - 1) * x

    def __call__(self, image, bbox, label):
        if torch.rand(1) < self.p:
            image = T.functional.rotate(image, float(self.degree))
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

        return image, bbox, label

    def _rotate_p(self, p, degree):
        theta = torch.deg2rad(degree)
        rot_mat = torch.tensor([
            [torch.cos(theta), torch.sin(theta)],
            [-torch.sin(theta), torch.cos(theta)],
        ])

        return torch.mm(rot_mat, p)


class RandomSampleCrop:
    def __init__(self, p=0.5, upper_ratio=0.1, reduction_thresh=0.8):
        self.p = p
        self.upper_ratio = upper_ratio
        self.reduction_thresh = reduction_thresh

    def __call__(self, image, bbox, label):
        if torch.rand(1) < self.p:
            cropped_corner = torch.rand(4) * self.upper_ratio
            cropped_corner[2:] = 1 - cropped_corner[2:]
            intersect = torch.cat(
                [torch.max(cropped_corner[0:2], bbox[:, 0:2]),
                 torch.min(cropped_corner[2:4], bbox[:, 2:4])],
                dim=1)
            if (box_area(intersect) / (box_area(bbox) + 1e-10) > self.reduction_thresh).all():
                W, H = image.size
                xmin, ymin, xmax, ymax = (cropped_corner * torch.tensor([W, H, W, H]))
                image = T.functional.crop(image, int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin))
                bbox = (bbox - cropped_corner[0:2].repeat(2)) / (cropped_corner[2:4].repeat(2) - cropped_corner[0:2].repeat(2))

        return image, bbox, label


class RandomMirror:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bbox, label):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            bbox[:, 0], bbox[:, 2] = 1 - bbox[:, 2], 1 - bbox[:, 0]

        return image, bbox, label


class ToTensor:
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, image, bbox, label):
        image = self.to_tensor(image)
        return image, bbox, label


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.normalize = T.Normalize(mean=mean, std=std)

    def forward(self, image, bbox, label):
        image = self.normalize(image)
        return image, bbox, label
