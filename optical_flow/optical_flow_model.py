# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pk
import torch
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import utils.img_utils as imutil
from optical_flow.flow_checker import forward_backward_flow_check


def load_refined_cse_maps(sequence_index: str, cache_path: str) -> torch.Tensor:
    computed_file_path = os.path.join(cache_path, f"{sequence_index}_cse_maps_refined.pk")
    assert os.path.isfile(computed_file_path)
    with open(computed_file_path, "rb") as f:
        return torch.Tensor(pk.load(f))


def partition_index(totalsize: int, chunksize: int) -> list[tuple[int, int]]:
    a, b = 0, 0
    if totalsize <= chunksize:
        return [(a, totalsize)]
    list_tuples = []
    b = chunksize
    while b < totalsize:
        list_tuples.append((a, b))
        a = b
        b += chunksize
    list_tuples.append((a, min(totalsize, b)))
    return list_tuples


class OfModelRaft:
    """
    Optical Flow model.
    """

    def __init__(
        self,
        images: torch.Tensor,
        threshold: int = 5,
        resize_resolution: tuple[int, int] | None = None,
        device: str = "cuda",
        chunksize: int = 16,
    ):

        self.device = device
        self.images = images
        self.threshold = threshold
        self.chunksize = chunksize
        self.load_model()

        # The optical flow is computed at this resolution
        self.of_resolution = (images.shape[1], images.shape[2])
        # The optical flow is returned at this resolution
        self.resize_resolution = resize_resolution

    def to(self, device):
        self.images = self.images.to(device)
        self.model = self.model.to(device)
        return self

    def load_model(self):
        # Initialize the optical flow model
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    def forward_model(self, images_i: torch.Tensor, images_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - images_i, images_j: TORCHUINT8 (B, 3, H, W)
        Returns:
            - optical_flow_i_j TORCHFLOAT32 (B, H, W, 2)
        """
        optical_flow = torch.zeros((images_i.shape[0], images_i.shape[2], images_i.shape[3], 2), dtype=torch.float32, device=images_i.device)

        for a, b in partition_index(images_i.shape[0], chunksize=self.chunksize):
            list_of_flows = self.model(images_i[a:b], images_j[a:b])
            gt_optical_flow_forward = list_of_flows[-1].permute(0, 2, 3, 1)
            optical_flow[a:b] = gt_optical_flow_forward

        # Resize
        if self.resize_resolution is not None:
            optical_flow[..., 0] = optical_flow[..., 0] * (self.resize_resolution[0] / gt_optical_flow_forward.shape[1])
            optical_flow[..., 1] = optical_flow[..., 1] * (self.resize_resolution[1] / gt_optical_flow_forward.shape[2])
            optical_flow = imutil.resize_torch(optical_flow, self.resize_resolution)

        return optical_flow

    def compute_optical_flow(
        self,
        X_ind_i: torch.Tensor,
        X_ind_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            - images_i, images_j TORCHFLOAT32 (BATCH, H, W, 3) [self.DEVICE]
        Returns:
            - opticalflow TORCHFLOAT32 (BATCH, H, W, 2) [self.DEVICE]
            - mask (BATCH, H, W) TORCHBOOL
        """
        images_i = self.images[X_ind_i]
        images_j = self.images[X_ind_j]

        """
        # Preprocess images
        Before transforms, images should be in shape TORCHUINT8 (BATCH, 3, H, W)
        """
        images_i_pp, images_j_pp = self.transforms(torch.round(images_i).type(torch.uint8).permute(0, 3, 1, 2), torch.round(images_j).type(torch.uint8).permute(0, 3, 1, 2))
        images_i_pp, images_j_pp = images_i_pp.contiguous(), images_j_pp.contiguous()

        with torch.no_grad():

            gt_optical_flow_forward = self.forward_model(images_i_pp, images_j_pp)
            gt_optical_flow_backward = self.forward_model(images_j_pp, images_i_pp)

            mask_of = forward_backward_flow_check(gt_optical_flow_forward.permute(0, 3, 1, 2).contiguous(), gt_optical_flow_backward.permute(0, 3, 1, 2).contiguous(), threshold=self.threshold)

            mask_of_b = forward_backward_flow_check(gt_optical_flow_backward.permute(0, 3, 1, 2).contiguous(), gt_optical_flow_forward.permute(0, 3, 1, 2).contiguous(), threshold=self.threshold)

        return gt_optical_flow_forward, gt_optical_flow_backward, mask_of, mask_of_b
