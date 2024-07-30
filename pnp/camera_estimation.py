# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pickle as pk
import os
from pytorch3d.renderer import PerspectiveCameras
import utils.align_cam_utils as aligncamutil


def get_moving_cameras_from_PNP_archive(sequence_index: str, cache_path: str, cameras_original: PerspectiveCameras) -> PerspectiveCameras:
    """
    Get refined cameras from the PnP preprocessing.

    Args:
        - cameras_original: PerspectiveCameras (N,) [DEVICE]
    """

    pnp_preprocess_file_path = os.path.join(cache_path, f"{sequence_index}_pnp_R_T.pk")

    assert os.path.isfile(pnp_preprocess_file_path), (f"Preprocessing file not found for sequence: {sequence_index}.", "Please launch preprocessing before scene optimization.")

    with open(pnp_preprocess_file_path, "rb") as f:
        pnp_solution = pk.load(f)

    valid_indices, R_pnp, T_pnp = pnp_solution["valid_indices_pnp"], pnp_solution["R_PNP"], pnp_solution["T_PNP"]

    return align_cameras_from_PNP_solution(cameras_original, R_pnp, T_pnp, valid_ind=None)


def align_cameras_from_PNP_solution(cameras_original: PerspectiveCameras, R_pnp: torch.Tensor, T_pnp: torch.Tensor, valid_ind=None, align_mode="extrinsics", estimate_scale=True) -> PerspectiveCameras:
    """
    Align cameras of a scene with the set of solution cameras from PnP preprocessing.

    Args:
        - cameras_original PerspectiveCameras (N,)
        - R_pnp TORCHFLOAT32 (N,3,3)
        - T_pnp TORCHFLOAT32 (N,3)

    Returns:
        - cameras_aligned PerspectiveCameras (N,)
    """

    if align_mode not in ["extrinsics", "centers"]:
        raise Exception("Unknown alignment mode")

    if valid_ind is None:
        valid_ind = torch.arange(R_pnp.shape[0])

    cameras_pnp = PerspectiveCameras(
        focal_length=cameras_original.focal_length,
        principal_point=cameras_original.principal_point,
        R=R_pnp,
        T=T_pnp,
        device=cameras_original.device,
        in_ndc=cameras_original.in_ndc(),
        image_size=cameras_original.image_size,
    )

    _, (align_t_R, align_t_T, align_t_s) = aligncamutil.corresponding_cameras_alignment(
        cameras_src=cameras_original[valid_ind], cameras_tgt=cameras_pnp[valid_ind], mode=align_mode, estimate_scale=estimate_scale, return_solution=True
    )
    cameras_aligned = aligncamutil.align_cameras_from_solution(cameras_original, align_t_R, align_t_T, align_t_s)

    return cameras_aligned
