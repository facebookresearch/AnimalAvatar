# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.structures import Meshes
from smal import SMAL
from model.texture_models import TextureBase
from model.pose_models import PoseBase, PoseCanonical, compute_pose
from losses.arap_fast import arap_fast


class LossOptimArapFast:

    def __init__(self, smal: SMAL, batch_size: int, device: str):
        """
        ARAP Fast optimizer.
        """
        self.smal = smal
        self.device = device
        self.batch_size = batch_size
        self.vertices_init = None

    def initialization(self, pose_model: PoseBase):
        """
        Get vertices initialization in canonical pose
        """
        model_canonical = PoseCanonical(
            N=1,
            betas=pose_model.betas.cpu().detach().clone(),
            betas_limbs=pose_model.betas_limbs.cpu().detach().clone(),
            vert_off=pose_model.vertices_off.cpu().detach().clone(),
        ).to(self.device)

        vertices, _, _ = compute_pose(self.smal, model_canonical, torch.tensor([0]).reshape((1,)), torch.tensor([0.0]).reshape((1,)))

        self.vertices_init = vertices

    def forward_batch(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        texture_model: TextureBase,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute ARAP fast (<=> edge loss) between optimed_model in canonical pose and in posed shape
        """

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(self.smal, pose_model, X_ind, X_ts)

        # Initialize ARAP fast
        if self.vertices_init is None:
            self.initialization(pose_model)

        # Compute ARAP fast
        loss = arap_fast(Meshes(self.vertices_init.expand(vertices.shape[0], -1, -1), faces), Meshes(vertices, faces))

        return loss
