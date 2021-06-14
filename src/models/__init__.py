from .ssd import SSD
from .retinanet import RetinaNet
from torchvision.models import *  # noqa
import torch


MODELS = {
    'ssd': SSD,
    'retinanet': RetinaNet
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
