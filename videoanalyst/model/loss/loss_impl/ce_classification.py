# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from ...module_base import ModuleBase
from ..loss_base import CLS_LOSSES
from .utils import SafeLog

eps = np.finfo(np.float32).tiny


@CLS_LOSSES.register
class CrossEntropyClassification(ModuleBase):
    r""" A wrapper for Cross Entropy Loss for Classificaiton
    """
    default_hyper_params = dict(
        name="cls_ce",  # will be displayed in monitor
        weight=1.0,
    )

    def __init__(self, ):
        super(CrossEntropyClassification, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_data, target_data):
        logits = pred_data["logits"]
        gt_class = target_data["anno"]

        loss = self.criterion(logits, gt_class)

        extra = dict()
        accu_dict = _accuracy(logits, gt_class, topk=(1, 5))
        extra.update(accu_dict)

        return loss, extra

def _accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the k top predictions for the specified values of k
        Borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = dict()
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_k = correct_k.mul_(100.0 / batch_size)
            res["top{}".format(k)] = acc_k
        return res
