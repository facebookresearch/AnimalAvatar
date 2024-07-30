# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras


class CustomSingleVideo(Dataset):
    """
    An example custom dataset class
    To train Animal-Avatar on a custom scene, complete this Dataset (except get_sparse_keypoints()) with your data
    """

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        raise NotImplementedError

    def get_masks(self, list_items: list[int]) -> np.ndarray:
        """
        Return:
            - masks [len(list_items),H,W,1] NPFLOAT32 [0,1](foreground probability)
        """
        raise NotImplementedError

    def get_imgs_rgb(self, list_items: list[int]) -> np.ndarray:
        """
        Return:
            - imgs_rgb [len(list_items),H,W,3] NPFLOAT32 [0,1] (rgb frames)
        """
        raise NotImplementedError

    def get_cameras(self, list_items: list[int]) -> PerspectiveCameras:
        """
        Return:
            - cameras (len(list_items),) PerspectiveCameras in ndc_space
        """
        raise NotImplementedError

    def get_sparse_keypoints(self, list_items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Optional!
        Returns:
            - sparse_keypoints (N, N_KPTS, 2) NPFLOAT32
            - scores (N, N_KPTS, 1) NPFLOAT32
        """
        assert False, ("No GT sparse-keypoints are available for custom scenes.", "Please set exp.l_optim_sparse_kp=0 in config/config.yaml", "This will affect performance quality.")
