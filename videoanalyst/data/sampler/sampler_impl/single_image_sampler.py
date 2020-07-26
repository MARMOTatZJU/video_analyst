# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from yacs.config import CfgNode

from videoanalyst.data.dataset.dataset_base import DatasetBase
from videoanalyst.utils import load_image

from ..sampler_base import CLS_SAMPLERS, SamplerBase


@CLS_SAMPLERS.register
class SingleImageSampler(SamplerBase):
    r"""
    Single image sampler
    Sample procedure: just load the image

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        nr_image_per_epoch=1,
    )

    def __init__(self,
                 datasets: List[DatasetBase] = [],  # only use the first dataset in the list
                 seed: int = 0,
                 data_filter=None) -> None:
        super().__init__(datasets, seed=seed)
        self._state["index_list"] = tuple(range(len(self.datasets[0])))
        if data_filter is None:
            self.data_filter = lambda x: (x is None)
        else:
            self.data_filter = data_filter

    def __getitem__(self, item) -> dict:
        r""" TODO: Add random sampler to sample full epoch with arg *item* passed in in deterministic order
        """
        data = None
        sample_try_num = 0

        # TODO: ensure full epoch sampling
        item = self._state["rng"].choice(self._state["index_list"])

        while self.data_filter(data):
            data = self.datasets[0][item]
            data["image"] = load_image(data["image"])
            sample_try_num += 1
        sampled_data = data

        return sampled_data
