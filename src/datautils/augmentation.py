import torch
import torchvision.transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)

        return image, label


class RandomColorJitter:
    def __init__(self, b_ratio=0, c_ratio=0, s_ratio=0, h_ratio=0):
        self.color_jitter = T.ColorJitter(
            brightness=b_ratio,
            contrast=c_ratio,
            saturation=s_ratio,
            hue=h_ratio
        )

    def __call__(self, image, label):
        image = self.color_jitter(image)
        return image, label


class Resize:
    def __init__(self, input_size):
        self.resize = T.Resize(input_size)

    def __call__(self, image, label):
        image = self.resize(image)
        return image, label


class RandomRotate:
    def __init__(self, degree, p=0.5):
        self.degree = self._get_uniformally(degree)
        self.p = p

    def _get_uniformally(self, x):
        return (torch.rand(1) * 2 - 1) * x

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            image = T.functional.rotate(image, float(self.degree), interpolation=T.functional.InterpolationMode.BICUBIC)
            label_ = []
            for class_id, x, y, w, h in label:
                corner_points = torch.tensor(
                    [[x, x,     x + w, x + w],
                     [y, y + h, y,     y + h]]
                ) - 0.5  # lt, lb, rt, rb
                rcp = self._rotate_p(corner_points, self.degree) + 0.5
                xmin, xmax, ymin, ymax = rcp[0].min(), rcp[0].max(), rcp[1].min(), rcp[1].max()
                label_.append([class_id, xmin, ymin, xmax - xmin, ymax - ymin])
            label = torch.tensor(label_)

        return image, label

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

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            label[:, 1] = 1 - label[:, 1] - label[:, 3]

        return image, label


class ToTensor:
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, image, label):
        image = self.to_tensor(image)
        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image, label):
        image = self.normalize(image)
        return image, label
