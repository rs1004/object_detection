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

    def get_parameters(self, train_conditions: list) -> list:
        """ 学習パラメータを取得する

        Args:
            train_conditions (list): 学習パラメータの振り分けの条件

        Returns:
            list: 学習パラメータ一覧
        """
        for name, param in self.named_parameters():
            for i in range(len(train_conditions)):
                if any(k in name for k in train_conditions[i]['keys']):
                    if train_conditions[i].get('lr', None) == 0:
                        param.requires_grad = False
                    else:
                        if 'params' not in train_conditions[i]:
                            train_conditions[i]['params'] = []
                        train_conditions[i]['params'].append(param.shape)
                    break

        params_to_update = []
        for d in train_conditions:
            if 'params' not in d:
                continue
            d.pop('keys')
            params_to_update.append(d)

        return params_to_update

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def loss(self, outputs, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def pre_predict(self, outputs: tuple) -> tuple:
        pass
