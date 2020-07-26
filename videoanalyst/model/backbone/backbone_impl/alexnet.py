# -*- coding: utf-8 -*
from loguru import logger

import torch
import torch.nn as nn

from videoanalyst.model.backbone.backbone_base import CLS_BACKBONES
from videoanalyst.model.module_base import ModuleBase


# __all__ = ['AlexNet', 'alexnet']


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }


@CLS_BACKBONES.register
class AlexNetOrigin(ModuleBase):
    r""" Original version of AlexNet
           used for training alignement w/t official example
           from: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    default_hyper_params = {"pretrain_model_path": "",
                            "output_width": 4096,}

    def __init__(self, num_classes=1000):   
        super(AlexNetOrigin, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.unsqueeze(2).unsqueeze(3)
        return x


# def alexnet(pretrained=False, progress=True, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = AlexNet(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
