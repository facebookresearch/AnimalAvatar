# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from cse_embedding.functional_map import squared_euclidean_distance_matrix


def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=1)


def get_edm_X_mesh(X: torch.Tensor, mesh_embedding: torch.Tensor, temperature: float = 1 / 100_000) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        - X TORCHFLOAT32 (X, D_cse)
        - cse_mesh TORCHFLOAT32 (N_verts, D_cse)

    Returns:
        - P_X_to_mesh TORCHDFLOAT32 (X, N_verts)
        - P_mesh_to_X TORCHDFLOAT32 (N_verts, X)
    """
    edm_img_to_mesh = 1 - squared_euclidean_distance_matrix(X, mesh_embedding) / 2
    # Project image cse to mesh cse ->(H_cse*W_cse, N_verts)
    P_X_to_mesh = temperature_scaled_softmax(edm_img_to_mesh, temperature=temperature)
    # Project mesh cse to image cse ->(N_verts, H_cse*W_cse)
    P_mesh_to_X = temperature_scaled_softmax(edm_img_to_mesh.T, temperature=temperature)

    return P_X_to_mesh, P_mesh_to_X


def get_cse_keypoint_cycle_masks(cse_images: torch.Tensor, cse_masks: torch.Tensor, cse_mesh: torch.Tensor, temperature: float = 1 / 100_000) -> torch.Tensor:
    """
    Given a set of CSE images and CSE embedding on the mesh, compute forward/backward to generate a mask of valid points

    Args:
        - cse_images TORCHFLOAT32 (BATCH, H_cse, W_cse, D_cse)
        - cse_masks TORCHFLOAT32 (BATCH, H_cse, W_cse, 1)
        - cse_mesh TORCHFLOAT32 (N_verts, D_cse)

    Returns:
        - cycle_masks TORCHFLOAT32 (BATCH, H_cse, W_cse, 1)
    """

    BATCH, H_cse, W_cse, D_cse = cse_images.shape
    device = cse_images.device
    cycle_masks = torch.zeros((BATCH, H_cse, W_cse), dtype=torch.float32, device=device)

    for i in range(BATCH):

        mask_i = cse_masks[i, ..., 0] > 0

        if torch.sum(mask_i) == 0:
            continue

        P_img_to_mesh, P_mesh_to_img = get_edm_X_mesh(cse_images[i][mask_i].reshape(-1, D_cse), cse_mesh, temperature)

        # Get the weights (diagonal of P_img_to_mesh @ P_mesh_to_img)
        cycle_masks[i, mask_i] = (P_img_to_mesh * P_mesh_to_img.t()).sum(dim=1)

    return cycle_masks
