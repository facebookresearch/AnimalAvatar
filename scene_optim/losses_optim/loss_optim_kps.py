# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer import PerspectiveCameras
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase
from smal.smal_torch import SMAL


class LossOptimCSEKp:
    """
    Args:
        - cameras PerspectiveCameras (BATCH,) [DEVICE]
        - keyp2d (BATCH, N_kps, 2) TORCHFLOAT32 [0,1]
        - closest_verts (BATCH, N_kps) TORCHINT64
        - valid_indices_cse (N_valid,) TORCHINT32 GLOBAL_INDICES
        - max_ind_csekp (BATCH,) TORCHINT64
        - smal_verts_cse_embedding (N_verts, D_CSE) TORCHFLOAT32
    """

    def __init__(
        self,
        smal: SMAL,
        device: str,
        cse_keypoints_xy: torch.Tensor,
        cse_keypoints_vert_id: torch.Tensor,
        cse_valid_indices: torch.Tensor,
        cse_keypoints_max_indice: torch.Tensor,
        cameras: PerspectiveCameras,
        image_size: int,
    ):

        self.smal = smal
        self.device = device
        self.cse_keypoints_xy = cse_keypoints_xy
        self.cse_keypoints_vert_id = cse_keypoints_vert_id
        self.cse_valid_indices = cse_valid_indices
        self.cameras = cameras
        self.image_size = image_size
        self.cse_keypoints_max_indice = cse_keypoints_max_indice.to(self.device)

        BATCH, N_KPS_MAX, _ = self.cse_keypoints_xy.shape

        self.mask_kp = torch.zeros((BATCH, N_KPS_MAX), dtype=torch.float32, device=self.device)
        for k in range(BATCH):
            self.mask_kp[k, : self.cse_keypoints_max_indice[k]] = 1.0

    def forward_batch(self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, vertices=None, faces=None):

        # Set of valid indices to be applied on the current X_ind
        valid_indices_relative = torch.isin(X_ind, self.cse_valid_indices).reshape((-1,)).nonzero(as_tuple=True)[0]
        X_ind_p = X_ind[valid_indices_relative]
        X_ts_p = X_ts[valid_indices_relative]
        BATCH_P = X_ind_p.shape[0]

        if (vertices is None) or (faces is None):
            vertices, _, _ = compute_pose(self.smal, pose_model, X_ind, X_ts)

        # L2 loss
        total_loss = torch.zeros((X_ind.shape[0], 1), device=self.device)

        if BATCH_P == 0:
            return total_loss

        # Get the 3d coordinates of points on the smal mesh correspoding to cse_keypoints_xy -> (BATCH_P, N_KPS, 3)
        keyp_3d = vertices[valid_indices_relative.reshape(-1, 1), self.cse_keypoints_vert_id[X_ind_p]]
        # project on screen -> (BATCH_P, N_KPS, 2)
        keyp_3dp = self.cameras[X_ind_p].transform_points_screen(keyp_3d, with_xyflip=True, image_size=(self.image_size, self.image_size))[..., :2].reshape((BATCH_P, -1, 2)) / self.image_size

        loss_per_kp = self.mask_kp[X_ind_p] * ((self.cse_keypoints_xy[X_ind_p] - keyp_3dp) ** 2).sum(axis=2).sqrt()
        loss = (loss_per_kp).sum(axis=1) / (1e-6 + self.cse_keypoints_max_indice[X_ind_p].type(torch.float32))
        loss = loss.reshape((-1, 1))
        total_loss[valid_indices_relative] = loss

        return total_loss


class LossOptimSparseKp:
    """
    Args:
        - sparse_keypoints (BATCH, N_KPS, 2) TORCHFLOAT32 XY [0,1]
        - sparse_keypoints_scores (BATCH, N_KPS, 1) TORCHFLOAT32
    """

    def __init__(self, smal: SMAL, device: str, sparse_keypoints: torch.Tensor, sparse_keypoints_scores: torch.Tensor, cameras: PerspectiveCameras, image_size: int):
        self.smal = smal
        self.device = device
        self.cameras = cameras
        self.image_size = image_size
        self.sparse_keypoints = sparse_keypoints
        self.sparse_keypoints_scores = sparse_keypoints_scores

    def forward_batch(self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, vertices=None, faces=None):

        # Compute predicted 3D joint position and project on screen
        BATCH = X_ind.shape[0]
        _, keyp_3d, _ = compute_pose(self.smal, pose_model, X_ind, X_ts)
        keyp_3d_projected = self.cameras[X_ind].transform_points_screen(keyp_3d, with_xyflip=True, image_size=(self.image_size, self.image_size))[..., :2].reshape((BATCH, -1, 2)) / self.image_size

        # L2 loss
        loss_per_kp = self.sparse_keypoints_scores[X_ind, :, 0] * ((self.sparse_keypoints[X_ind] - keyp_3d_projected) ** 2).sum(axis=2).sqrt()
        loss = (loss_per_kp).sum(axis=1) / (1e-6 + self.sparse_keypoints_scores[X_ind, :, 0].sum(axis=1))
        loss = loss.reshape((-1, 1))
        return loss
