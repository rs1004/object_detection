from .ssd300 import SSD300
from .ssd512 import SSD512
from .darknet import Darknet53  # noqa
from .yolov3 import YoloV3
from .retinanet import RetinaNet
from .fcos import FCOS
from torchvision.models import *  # noqa
import torch


MODELS = {
    'ssd300': SSD300,
    'ssd512': SSD512,
    'yolov3': YoloV3,
    'retinanet': RetinaNet,
    'fcos': FCOS
}


def Model(type: str, num_classes: int, backbone: str, backbone_weight: str = None):
    m = eval(backbone)
    if backbone_weight:
        backbone = m(pretrained=False)
        backbone.load_state_dict(torch.load(backbone_weight), strict=False)
    else:
        backbone = m(pretrained=True)

    model = MODELS[type](num_classes=num_classes, backbone=backbone)

    return model
