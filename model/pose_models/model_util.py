# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Literal
from smal import SMAL
from pytorch3d.transforms import rotation_6d_to_matrix


class PoseModelTags:
    PoseBase = "PoseBase"
    PoseMLP = "PoseMLP"
    PoseFixed = "PoseFixed"
    PoseCanonical = "PoseCanonical"


class PoseBase(nn.Module):
    """
    Base class for models representing the pose in a scene.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_pose(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError

    def compute_orient(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError

    def compute_transl(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError

    def compute_betas(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError

    def compute_betas_limbs(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError

    def compute_vertices_off(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError

    def compute_global(self, X_ind, X_ts) -> torch.Tensor:
        raise NotImplementedError


def _safe_parameter_init(X: torch.Tensor | None, shape, requires_grad: bool, init: Literal["empty", "zeros"] = "empty") -> nn.Parameter:
    if X is not None:
        assert X.shape == shape, f"Shape mismatch shape: {shape}, X.shape: {X.shape}"
        return nn.Parameter(X.clone().detach(), requires_grad=requires_grad)
    else:
        if init == "empty":
            return nn.Parameter(torch.empty(shape), requires_grad=requires_grad)
        elif init == "zeros":
            return nn.Parameter(torch.zeros(shape), requires_grad=requires_grad)
        else:
            raise Exception(f"Invalid init mode: {init}")


def partition_index(totalsize, chunksize):
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 1e-5)
        m.bias.data.fill_(1e-5)


def get_smal_inputs_from_pose_model(pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From a pose_model, return (betas, betas_limbs, orient, pose, translation, vert_offset) that can be passed to SMAL to get 3D vertices.

    Args:
        - X_ind TORCHINT64 [DEVICE] (BATCH,)
        - X_ts TORCHFLOAT32 [DEVICE] (BATCH,)

    Returns:
        - X_betas (BATCH, 30)
        - X_betas_limbs (BATCH, 7)
        - X_global_pose (BATCH, (34+1)*6)
        - X_transl (BATCH, 3)
        - X_vertices_off (BATCH, 5901)
    """
    global_compute = pose_model.compute_global(X_ind, X_ts)
    X_transl = global_compute[..., 0:3]
    X_orient = global_compute[..., 3 : 3 + 6]
    X_pose = global_compute[..., 3 + 6 : 3 + 35 * 6]
    X_global_pose = torch.cat([X_orient, X_pose], dim=1)

    X_betas = pose_model.compute_betas(X_ind, X_ts)
    X_betas_limbs = pose_model.compute_betas_limbs(X_ind, X_ts)
    X_vertices_off = pose_model.compute_vertices_off(X_ind, X_ts)

    return X_betas, X_betas_limbs, X_global_pose, X_transl, X_vertices_off


def compute_pose(smal: SMAL, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get 3D vertices positions from pose_model.

    Args:
        - pose_modelEVICE] [GRAD]
        - smal [DEVICE]
        - X_ind (BATCH,) TORCHINT64
        - X_ts (BATCH,) TORCHFLOAT32

    Return:
        - vertices (BATCH, 3889, 3) TORCHFLOAT32 [DEVICE] [GRAD]
        - keypoints_3d (BATCH, 24, 3) TORCHFLOAT32 [DEVICE] [GRAD]
        - faces (BATCH, 7774, 3) TORCHINT64 [DEVICE]

    """
    (X_betas, X_betas_limbs, X_global_pose, X_transl, X_vertices_off) = get_smal_inputs_from_pose_model(pose_model, X_ind, X_ts)

    X_pose_mat = rotation_6d_to_matrix(X_global_pose.reshape((-1, 6))).reshape((-1, 34 + 1, 3, 3))

    vertices, keypoints_3d = smal(beta=X_betas, betas_limbs=X_betas_limbs, pose=X_pose_mat, trans=X_transl, vert_off_compact=X_vertices_off)

    faces = smal.faces.unsqueeze(0).expand((X_ind.shape[0], -1, -1))

    return vertices, keypoints_3d, faces
