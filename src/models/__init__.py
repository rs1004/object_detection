from .ssd import SSD
from .ssd2 import SSD2
from .yolov3 import YoloV3
from .darknet import Darknet53  # noqa
from torchvision.models import *  # noqa
import torch


MODELS = {
    'ssd': SSD,
    'ssd2': SSD2,
    'yolov3': YoloV3
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
