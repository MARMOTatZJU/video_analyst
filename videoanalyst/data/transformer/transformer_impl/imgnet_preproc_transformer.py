from typing import Dict

from PIL import Image
import cv2
import numpy as np
from yacs.config import CfgNode

import torch
import torchvision.transforms as transforms

from ..transformer_base import CLS_TRANSFORMERS, TransformerBase


@CLS_TRANSFORMERS.register
class ImagenetPreprocessTransformer(TransformerBase):
    r"""
    Cropping training pair with data augmentation (random shift / random scaling)
        from: https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Hyper-parameters
    ----------------
    input_size : int
        input image size
    """
    default_hyper_params = dict(
        input_size=224,
    )

    def __init__(self, seed: int = 0) -> None:
        super(ImagenetPreprocessTransformer, self).__init__(seed=seed)


    def update_params(self, seed: int = 0) -> None:
        super(ImagenetPreprocessTransformer, self).update_params(seed)
        input_size = self._hyper_params["input_size"]
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
        ])

    def __call__(self, sampled_data: Dict) -> Dict:
        r"""
        sampled_data: Dict()
            input data
            Dict(data=Dict(image, anno))
        """
        data = sampled_data
        # TODO: investigate if PIO conversion is too slow
        image = data["image"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        anno = data["anno"]

        image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose(2, 0, 1)  # HxWxC -> CxHxW

        # sampled_data["data"] = dict(image=image, anno=anno)
        sampled_data = dict(image=image, anno=anno)

        return sampled_data
