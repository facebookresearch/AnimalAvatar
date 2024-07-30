# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from model.pose_models import PoseBase
from model.texture_models import TextureBase


class LossOptimTVReg:

    def __init__(self, X_ind_global: torch.Tensor, X_ts_global: torch.Tensor):
        self.delta_tv = 1e-3
        self.X_ind_global = X_ind_global
        self.X_ts_global = X_ts_global

    def forward_batch(
        self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, vertices: torch.Tensor | None = None, faces: torch.Tensor | None = None
    ) -> torch.Tensor:

        # Randomly sample a position between X_ts_i-1 and X_ts_i+1
        rel_indices_to_global_min = torch.clamp(self.X_ind_global - 1, 0, self.X_ind_global.shape[0] - 1)
        rel_indices_to_global_max = torch.clamp(self.X_ind_global + 1, 0, self.X_ind_global.shape[0] - 1)

        sampled_X_ts = self.X_ts_global[rel_indices_to_global_min] + torch.rand(self.X_ind_global.shape[0]) * (
            self.X_ts_global[rel_indices_to_global_max] - self.X_ts_global[rel_indices_to_global_min]
        )

        global_tr_orient_pose = pose_model.compute_global(X_ind, sampled_X_ts)
        global_tr_orient_pose_j = pose_model.compute_global(X_ind, sampled_X_ts + self.delta_tv)

        # Compute TV-reg on global pose [translation, orientation, pose]
        tv_reg = torch.mean((torch.abs((global_tr_orient_pose_j - global_tr_orient_pose))) / self.delta_tv, dim=[0, 1]).reshape((1, 1)).repeat((X_ind.shape[0], 1))

        return tv_reg
