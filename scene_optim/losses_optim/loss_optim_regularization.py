# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.structures import Meshes
from smal import SMAL
from losses import mesh_laplacian_smoothing
from model.texture_models import TextureBase
from model.pose_models import PoseBase, compute_pose


class LossLaplacianReg:

    def __init__(self, smal: SMAL, device: str):
        self.smal = smal
        self.device = device

    def forward_batch(
        self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, vertices: torch.Tensor | None = None, faces: torch.Tensor = None
    ) -> torch.Tensor:

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(self.smal, pose_model, X_ind, X_ts)

        return mesh_laplacian_smoothing(Meshes(vertices, faces.detach()))
