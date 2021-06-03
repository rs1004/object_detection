from .ssd import SSD
from .retinanet import RetinaNet
from torchvision.models import *  # noqa
import torch


MODELS = {
    'ssd': SSD,
    'retinanet': RetinaNet
}


def Model(type: str, num_classes: int, backborn: str, backborn_weight: str = None):
    m = eval(backborn)
    if backborn_weight:
        backborn = m(pretrained=False)
        backborn.load_state_dict(torch.load(backborn_weight), strict=False)
    else:
        backborn = m(pretrained=True)

    model = MODELS[type](num_classes=num_classes, backborn=backborn)

    return model
