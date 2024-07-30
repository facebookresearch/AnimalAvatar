# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from model.pose_models.model_util import _safe_parameter_init, PoseBase
from pytorch3d.transforms import matrix_to_rotation_6d


def rot_x(angle_rad: float) -> torch.Tensor:
    tangle_rad = torch.tensor(angle_rad)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, torch.cos(tangle_rad), -torch.sin(tangle_rad)], [0.0, torch.sin(tangle_rad), torch.cos(tangle_rad)]])


def rot_y(angle_rad: float) -> torch.Tensor:
    tangle_rad = torch.tensor(angle_rad)
    return torch.tensor([[torch.cos(tangle_rad), 0.0, torch.sin(tangle_rad)], [0.0, 1.0, 0.0], [-torch.sin(tangle_rad), 0.0, torch.cos(tangle_rad)]])


def rot_z(angle_rad: float) -> torch.Tensor:
    tangle_rad = torch.tensor(angle_rad)
    return torch.tensor([[torch.cos(tangle_rad), -torch.sin(tangle_rad), 0.0], [torch.sin(tangle_rad), torch.cos(tangle_rad), 0.0], [0.0, 0.0, 1.0]])


def get_original_orient(N: int, front_orient: bool, mode_6d: bool = False) -> torch.Tensor:
    """
    Returns:
        (N, 1, 3, 3) TORCHFLOAT32 or (N, 1, 6) TORCHFLOAT32
    """
    if front_orient:
        original_orient_3_3 = (rot_z(0) @ rot_y(-np.pi / 2) @ rot_x(-np.pi / 2)).repeat((N, 1, 1, 1)).to(torch.float32)
    else:
        original_orient_3_3 = torch.eye(3).reshape((1, 1, 3, 3)).repeat((N, 1, 1, 1)).to(torch.float32)

    if mode_6d:
        return matrix_to_rotation_6d(original_orient_3_3)
    else:
        return original_orient_3_3


def get_original_pose(N, N_joints: int = 34, mode_6d: bool = False) -> torch.Tensor:
    """
    Returns:
        (N, N_joints, 3, 3) TORCHFLOAT32
    """
    original_pose_3_3 = torch.eye(3).reshape((1, 1, 3, 3)).repeat((N, N_joints, 1, 1)).to(torch.float32)
    if mode_6d:
        return matrix_to_rotation_6d(original_pose_3_3)
    else:
        return original_pose_3_3


def get_original_translation(N: int) -> torch.Tensor:
    return torch.zeros((N, 3), dtype=torch.float32)


def get_original_global(N: int, front_orient: bool = False) -> torch.Tensor:
    """
    Returns canonical (Translation (3), Orientation (1*6), Pose (34*6))
    """
    return torch.cat(
        [
            get_original_translation(N).reshape(N, -1),
            get_original_orient(N, front_orient, mode_6d=True).reshape(N, -1),
            get_original_pose(N, N_joints=34, mode_6d=True).reshape(N, -1),
        ],
        dim=1,
    )


class PoseCanonical(PoseBase):

    def __init__(
        self,
        N: int,
        betas: torch.Tensor | None = None,
        betas_limbs: torch.Tensor | None = None,
        poses: torch.Tensor | None = None,
        orient: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        vert_off: torch.Tensor | None = None,
        front_orient: bool = True,
        req_grad: bool = False,
    ):
        """
        Canonical Pose model.

        Args:
            - betas (1, 30) TORCHFLOAT32
            - betas_limbs (1, 7) TORCHFLOAT32
            - vert_off (1, 5901) TORCHFLOAT32
            - orient (N, 1, 6) TORCHFLOAT32
            - pose (N, 34, 6) TORCHFLOAT32
            - transl (N, 3) TORCHFLOAT32
        """
        super(PoseCanonical, self).__init__()

        self.betas = _safe_parameter_init(betas, (1, 30), requires_grad=req_grad, init="zeros")
        self.betas_limbs = _safe_parameter_init(betas_limbs, (1, 7), requires_grad=req_grad, init="zeros")
        self.vertices_off = _safe_parameter_init(vert_off, (1, 5901), requires_grad=req_grad, init="zeros")

        if transl is None:
            transl = get_original_translation(N)
        if poses is None:
            poses = get_original_pose(N, mode_6d=True)
        if orient is None:
            orient = get_original_orient(N, front_orient=front_orient, mode_6d=True)

        self.poses = _safe_parameter_init(poses, (N, 34, 6), requires_grad=req_grad)
        self.orient = _safe_parameter_init(orient, (N, 1, 6), requires_grad=req_grad)
        self.transl = _safe_parameter_init(transl, (N, 3), requires_grad=req_grad)

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
