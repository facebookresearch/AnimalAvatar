# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from model.pose_models.model_util import _safe_parameter_init, PoseBase


class PoseFixed(PoseBase):
    def __init__(
        self,
        N: int,
        betas: torch.Tensor | None = None,
        betas_limbs: torch.Tensor | None = None,
        poses: torch.Tensor | None = None,
        orient: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        vertices_off: torch.Tensor | None = None,
    ):
        super(PoseFixed, self).__init__()

        self.tag = "PoseFixed"
        self.N = N

        self.poses = _safe_parameter_init(poses, (N, 34, 6), requires_grad=True)
        self.orient = _safe_parameter_init(orient, (N, 1, 6), requires_grad=True)
        self.transl = _safe_parameter_init(transl, (N, 3), requires_grad=True)

        self.betas = _safe_parameter_init(betas, (1, 30), requires_grad=True)
        self.betas_limbs = _safe_parameter_init(betas_limbs, (1, 7), requires_grad=True)
        self.vertices_off = _safe_parameter_init(vertices_off, (1, 5901), requires_grad=True, init="zeros")

    def compute_pose(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.poses[X_ind]

    def compute_orient(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.orient[X_ind]

    def compute_transl(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.transl[X_ind]

    def compute_betas(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.betas.expand(X_ind.shape[0], -1)

    def compute_betas_limbs(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.betas_limbs.expand(X_ind.shape[0], -1)

    def compute_vertices_off(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.vertices_off.expand(X_ind.shape[0], -1)

    def compute_global(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.compute_transl(X_ind, X_ts), self.compute_orient(X_ind, X_ts).reshape(X_ind.shape[0], -1), self.compute_pose(X_ind, X_ts).reshape(X_ind.shape[0], -1)], dim=-1)
