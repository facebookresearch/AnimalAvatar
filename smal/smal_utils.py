# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytorch3d.transforms
import torch
import pytorch3d


def get_transf_from_RT(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Args:
        - R (BATCH, 3, 3) TORCHFLOAT32
        - t (BATCH, 3, 1) TORCHFLOAT32
    Returns:
        - A (BATCH, 4, 4) TORCHFLOAT32
    """
    transform = pytorch3d.transforms.Transform3d(device=R.device)
    transform = transform.rotate(torch.transpose(R, 1, 2)).translate(t[..., 0])
    X = torch.transpose(transform.get_matrix(), 1, 2)
    return X


def get_global_transf_from_kinematic_transf(
    rotation_joints: torch.Tensor,
    position_joints: torch.Tensor,
    scale_joints: torch.Tensor,
    parent: np.ndarray,
) -> torch.Tensor:
    """
    Computes absolute joint locations given pose.
    Args:
      - rotation_joints: (N, N_JOINTS, 3, 3) TORCHFLOAT32: rotation vector of K joints
      - position_joints: N x 35 x 3, joint locations before posing
      - scale_joints: (N, N_JOINTS, 3, 3) TORCHFLOAT32
      - parent: (N_JOINTS,) TORCHINT64: 24 holding the parent id for each index
    Returns
      A_rel : `Tensor`: N x 35 4 x 4 relative joint transformations for LBS.
    """

    N = rotation_joints.shape[0]
    device = rotation_joints.device

    # root transformation -> (BATCH, 4, 4)
    results = [get_transf_from_RT(rotation_joints[:, 0], position_joints[:, 0].unsqueeze(-1))]

    for i in range(1, parent.shape[0]):

        # Inverse scale parent of joint i ->(BATCH, 3, 3)
        try:
            scale_parent_i_inv = torch.inverse(scale_joints[:, parent[i]])
        except:
            scale_parent_i_inv = torch.max(scale_joints[:, parent[i]], 0.01 * torch.eye(3, device=device)[None, :, :])

        # Translation ->(BATCH, 3, 1)
        position_rel_i = (position_joints[:, i] - position_joints[:, parent[i]]).unsqueeze(-1)
        # Rotation ->(BATCH, 3, 3)
        rot_i = rotation_joints[:, i]
        # Scale ->(BATCH, 3, 3)
        scale_i = scale_joints[:, i]
        # local transformation joint i ->(BATCH, 4, 4)
        transf_i = get_transf_from_RT(scale_parent_i_inv @ rot_i @ scale_i, position_rel_i)
        # global transformation joint i ->(BATCH, 4, 4)
        res_i = torch.matmul(results[parent[i]], transf_i)
        results.append(res_i)

    # ->(BATCH, N_JOINTS, 4, 4)
    results = torch.stack(results, dim=1)

    # Get relative Transf.

    # ->(BATCH, N_JOINTS, 4, 1)
    Js_w0 = torch.cat([position_joints.unsqueeze(-1), torch.zeros((N, 35, 1, 1), device=device)], 2)

    init_bone = torch.matmul(results, Js_w0)
    init_bone = torch.nn.functional.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0))

    # ->(BATCH, N_JOINTS, 4, 4)
    return results - init_bone
