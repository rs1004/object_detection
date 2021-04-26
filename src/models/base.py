import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class DetectionNet(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(DetectionNet, self).__init__()

    def init_weights(self, blocks: list):
        """ 重み初期化を行う

        Args:
            blocks (list): 初期化を行うレイヤーブロックのリスト
        """
        for block in blocks:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    # He の初期化
                    # [memo] sigmoid, tanh を使う場合はXavierの初期値, Relu を使用する場合は He の初期値を使用する
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def get_parameters(self, lrs: dict = {'features': 0.0001, '_': 0.002}) -> list:
        """ 学習パラメータと学習率の一覧を取得する

        Args:
            lrs (dict, optional): 学習率の一覧. Defaults to {'features': 0.0001, '_': 0.001}.

        Returns:
            list: 学習パラメータと学習率の一覧
        """
        params_no_decay = []
        params_else = []

        for name, param in self.named_parameters():
            if len(param.shape) == 1:
                params_no_decay.append(param)
            else:
                params_else.append(param)

        params = [
            {'params': params_no_decay, 'weight_decay': 0.0},
            {'params': params_else},
        ]

        return params

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def loss(self, outputs, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def predict(self, images: torch.Tensor, image_metas: list, outputs,
                conf_thresh: float = 0.4, iou_thresh: float = 0.45):
        pass
