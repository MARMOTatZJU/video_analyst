# -*- coding: utf-8 -*-
import copy
import os.path as osp
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger
from yacs.config import CfgNode
import torchvision.datasets as datasets

from videoanalyst.data.dataset.dataset_base import CLS_DATASETS, DatasetBase
from videoanalyst.utils import load_image

_current_dir = osp.dirname(osp.realpath(__file__))


@CLS_DATASETS.register
class ImageNetDataset(DatasetBase):
    r"""
    ILCVRC-CLS (ImageNet) dataset helper
    """
    default_hyper_params = dict(
        dataset_root="datasets/ILSVRC2015",
        subset="train",
    )

    def __init__(self) -> None:
        r"""
        Create dataset with config

        Arguments
        ---------
        cfg: CfgNode
            dataset config
        """
        super(ImageNetDataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = osp.realpath(self._hyper_params["dataset_root"])
        subset = self._hyper_params["subset"]
        subset_dir = osp.join(dataset_root, "Data/CLS-LOC", subset)
        
        self._state["dataset"] = datasets.ImageFolder(
            subset_dir, 
            # only return the path
            loader=_identity_loader,
            # transforms.Compose([
            #     transforms.RandomResizedCrop(224),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize,
            #     ])
            )

    def __getitem__(self, item: int) -> Dict:
        # img_file: image file path
        # class_idx: integer
        img_file, class_idx = self._state["dataset"][item]
        sequence_data = dict(image=img_file, anno=class_idx)

        return sequence_data

    def __len__(self):
        return len(self._state["dataset"])

def _identity_loader(p):
    r""" Return path to sampler
         Image loading performed in sampler in this framework
    """
    return p
