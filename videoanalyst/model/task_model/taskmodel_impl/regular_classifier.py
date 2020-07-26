# -*- coding: utf-8 -*

import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from videoanalyst.model.common_opr.common_block import conv_bn_relu

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import CLS_TASKMODELS

torch.set_printoptions(precision=8)


@CLS_TASKMODELS.register
class RegularClassifier(ModuleBase):
    r"""
    Regular classifier for classification task (e.g. ImageNet pretraining)
    """
    default_hyper_params = dict(pretrain_model_path="",
                                nr_classes=1000,
                                )

    def __init__(self, backbone, head=None, loss=None):
        super(RegularClassifier, self).__init__()
        self.basemodel = backbone
        self.head = head
        self.loss = loss
        self.classifier = None
    
    def update_params(self,):
        super(RegularClassifier, self).update_params()
        feature_width = self.basemodel._hyper_params["output_width"]
        nr_classes = self._hyper_params["nr_classes"]
        self.classifier = nn.Linear(feature_width, nr_classes)

    def forward(self, training_data):
        im = training_data["image"]
        x = self.basemodel(im)
        x = x.mean(dim=3).mean(dim=2)
        x = self.classifier(x)

        predict_data = dict(logits=x)

        return predict_data

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
