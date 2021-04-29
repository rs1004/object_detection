from .ssd import SSD  # noqa
from torchvision.models import vgg16, vgg16_bn
import torch

BACKBORNS = {
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn
}

MODELS = {
    'ssd': SSD
}


def Model(type: str, num_classes: int, backborn: str, backborn_weight: str = None):
    m = BACKBORNS[backborn]
    if backborn_weight:
        backborn = m(pretrained=False)
        backborn.load_state_dict(torch.load(backborn_weight), strict=False)
    else:
        backborn = m(pretrained=True)

    model = MODELS[type](num_classes=num_classes, backborn=backborn)

    return model
