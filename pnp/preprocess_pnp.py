# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import pickle as pk
import os
import numpy as np
from model.pose_models import PoseCanonical, compute_pose
from pnp.p_n_p_optim import get_init_from_pnp
from data.input_cop import InputCop

torch.manual_seed(1003)
random.seed(1003)
np.random.seed(1003)
torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def compute_pnp_R_T(
    sequence_index: str, dataset_source: str, image_size: int, use_RANSAC: bool, N_points_per_frames: int, n_iter_max: int, thre_inl: float, min_inl_percentage: float, device: str
) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:

    ic = InputCop(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cse_mesh_name="smal",
        N_cse_kps=N_points_per_frames,
        device=device,
    )

    N_frames = ic.N_frames_synth
    X_ind = ic.X_ind
    X_ts = ic.X_ts
    smal = ic.smal
    cameras = ic.cameras_canonical.to(device)

    pose_model = PoseCanonical(N=N_frames).to(device)

    # -------------------------------------------------------------------------------------------------

    (valid_indices, cse_keypoints_xy, cse_keypoints_vert_id, cse_max_indice) = ic.cse_keypoints
    cse_keypoints_xy = cse_keypoints_xy.to(device) * image_size

    with torch.no_grad():
        vertices, _, _ = compute_pose(smal=smal, pose_model=pose_model, X_ind=X_ind, X_ts=X_ts)
    keypoints_3d = torch.stack([vertices[i][cse_keypoints_vert_id[i]] for i in range(N_frames)], dim=0)

    keypoint_scores = torch.ones((N_frames, N_points_per_frames, 1))
    for i in range(N_frames):
        keypoint_scores[i, cse_max_indice[i] :] = 0

    # Perpsective-N-point with RANSAC
    valid_ind, R_stack, T_stack, info_dict = get_init_from_pnp(
        keyp_3d=keypoints_3d,
        keyp2d_prediction=cse_keypoints_xy,
        keyp2d_score=keypoint_scores,
        valid_indices=valid_indices,
        cameras=cameras,
        image_size=image_size,
        device=device,
        use_RANSAC=use_RANSAC,
        n_iter_max=n_iter_max,
        thre_inl=thre_inl,
        min_inl_percentage=min_inl_percentage,
    )

    if len(valid_ind) == 0:
        return valid_ind, None, None
    else:
        return valid_ind, R_stack.cpu(), T_stack.cpu()


def is_already_computed_pnp(sequence_index: str, cache_path: str):
    """
    Check if the PNP solution for given sequence_index has already been calculated in cache_path
    """
    return os.path.isfile(os.path.join(cache_path, f"{sequence_index}_pnp_R_T.pk"))


def preprocess_pnp(sequence_index: str, dataset_source: str, cache_path: str, device: str = "cuda"):
    """
    Compute PNP solution for a sequence, and save result in 'cache_path'
    """
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

    save_path = os.path.join(cache_path, f"{sequence_index}_pnp_R_T.pk")

    valid_indices_pnp, R_pnp, T_pnp = compute_pnp_R_T(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        image_size=256,
        use_RANSAC=True,
        N_points_per_frames=200,
        n_iter_max=100,
        thre_inl=0.05,
        min_inl_percentage=0.1,
        device=device,
    )

    with open(save_path, "wb") as f:
        pk.dump({"valid_indices_pnp": valid_indices_pnp, "R_PNP": R_pnp, "T_PNP": T_pnp}, f)
