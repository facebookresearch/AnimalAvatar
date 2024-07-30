# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from tqdm import tqdm
import pickle as pk
from data.input_cop import InputCop
from config.keys import Keys
from cse_embedding.functional_map import get_closest_vertices_matrix
from optical_flow.flow_checker import warp_image
from optical_flow.optical_flow_model import OfModelRaft
from utils.img_utils import resize_torch


def is_already_computed_refined_cse(sequence_index: str, cache_path: str) -> bool:
    computed_file_path = os.path.join(cache_path, f"{sequence_index}_cse_maps_refined.pk")
    return os.path.isfile(computed_file_path)


def preprocess_refined_cse_from_optical_flow(
    sequence_index: str,
    dataset_source: str,
    device: str = "cuda",
):
    """
    Preprocess refined CSE maps of the video via Optical-Flow.
    """
    ic = InputCop(sequence_index=sequence_index, dataset_source=dataset_source, cse_mesh_name="cse")
    image_size = 256
    delta = 20

    geodesic_verts_map = ic.geodesic_distance_mesh.to(device)
    cse_maps, cse_masks, valid_indices = ic.cse_maps
    cse_embedding = ic.cse_embedding.to(device)
    (
        cse_maps,
        cse_masks,
    ) = cse_maps.to(
        device
    ), cse_masks.to(device)

    optical_flow_model = OfModelRaft(
        resize_torch(ic.images_hr.to(device), size=(512, 512)),
        threshold=3,
        resize_resolution=(image_size, image_size),
        device=device,
    )

    cse_maps_refined = refine_cse_images_with_optical_flow(
        cse_maps,
        cse_masks,
        optical_flow_model,
        geodesic_verts_map,
        cse_embedding,
        delta,
    )

    # Save output to cache
    file_path = os.path.join(Keys().preprocess_path_cse, "{}_cse_maps_refined.pk".format(sequence_index))
    with open(file_path, "wb") as f:
        pk.dump(cse_maps_refined.cpu().numpy(), f)


def refine_cse_images_with_optical_flow(
    cse_images: torch.Tensor,
    cse_masks: torch.Tensor,
    optical_flow_model: OfModelRaft,
    geodesic_map: torch.Tensor,
    mesh_cse_embedding: torch.Tensor,
    delta: int,
) -> torch.Tensor:
    """
    Returns:
        - cse_images_improved (N_frames, D_CSE, H, W) TORCHFLOAT32 [DEVICE]
    """
    N_FRAMES, D_CSE, H, W = cse_images.shape
    device = cse_images.device
    cse_images_improved = torch.zeros_like(cse_images)

    # Refine all frames
    for i in tqdm(range(N_FRAMES), desc="CSE-refinement with optical-flow"):

        warped_cse_images_i_to_rest = torch.zeros((N_FRAMES, D_CSE, H, W), dtype=torch.float32, device=device)
        warped_cse_masks_i_to_rest = torch.zeros((N_FRAMES, 1, H, W), dtype=torch.bool, device=device)

        # Compute optical flow i -> ALL OTHERS
        X_i = torch.full((N_FRAMES,), i).reshape((-1,))
        X_j = torch.arange(N_FRAMES).reshape((-1,))
        indices_j = list(range(max(0, i - delta), min(N_FRAMES, i + delta)))
        indice_i = indices_j.index(i)

        gt_optical_flow_forward, _, mask_of, _ = optical_flow_model.compute_optical_flow(X_i[indices_j], X_j[indices_j])
        mask_of[indice_i] = 1

        # Warp image j with optical flow i->j
        for ind_j, j in enumerate(range(max(0, i - delta), min(N_FRAMES, i + delta))):
            if j != i:
                warped_cse_images_i_to_rest[j] = warp_image(cse_images[[j]], gt_optical_flow_forward[[ind_j]].permute(0, 3, 1, 2), mode="nearest")[0]
                warped_cse_masks_i_to_rest[j] = warp_image(cse_masks[[j]].permute(0, 3, 1, 2).type(torch.float32), gt_optical_flow_forward[[ind_j]].permute(0, 3, 1, 2), mode="nearest")[0].type(
                    torch.bool
                )
            else:
                warped_cse_images_i_to_rest[j] = cse_images[i]
                warped_cse_masks_i_to_rest[j] = cse_masks[i].permute(2, 0, 1).type(torch.bool)

        # The valid cse must comply with forward-backward optical flow mask AND be in the warped cse mask
        mask_global = warped_cse_masks_i_to_rest[:, 0, :, :]
        mask_global[indices_j] = mask_global[indices_j] * mask_of

        # Aggregates via geodesic dist. on mesh to generate a new CSE map for frame i
        improved_cse_image_i = aggregate_cse_images(warped_cse_images_i_to_rest, cse_masks[i, ..., 0], mask_global, geodesic_map, mesh_cse_embedding)

        if improved_cse_image_i is not None:
            cse_images_improved[i] = improved_cse_image_i
        else:
            cse_images_improved[i] = cse_images[i]

    return cse_images_improved


def aggregate_cse_images(cse_image_aggregates: torch.Tensor, cse_mask: torch.Tensor, weight_mask: torch.Tensor, geodesic_map: torch.Tensor, mesh_cse_embedding: torch.tensor) -> torch.Tensor:
    """
    For a single frame, given an aggregates of cse_frames, return an aggregated cse_image version.

    Args:
        - cse_image_aggregates (K, D_CSE, H, W) TORCHFLOAT32
        - cse_mask (H, W) TORCHBOOL
        - weight_mask (K, H, W) TORCHINT32
        - geodesic_map (N_verts, N_verts) TORCHFLOAT32
        - mesh_cse_embedding (N_verts, D_CSE) TORCHFLOAT32

    Returns:
        - cse_aggregated_images (D_CSE, H, W) TORCHFLOAT32
    """
    N_FRAMES, D_CSE, H, W = cse_image_aggregates.shape
    device = cse_image_aggregates.device

    cse_img_agg_flat = cse_image_aggregates.permute(2, 3, 0, 1).reshape(-1, N_FRAMES, D_CSE)
    cse_mask_flat = cse_mask.reshape(
        -1,
    )
    weight_mask_flat = weight_mask.permute(1, 2, 0).reshape((-1, N_FRAMES))

    # ->(N_PTS, N_FRAMES, D_CSE), (N_PTS, N_FRAMES)
    cse_img_agg_flat = cse_img_agg_flat[cse_mask_flat > 0]
    weight_mask_flat = weight_mask_flat[cse_mask_flat > 0]
    N_PTS, _, _ = cse_img_agg_flat.shape

    if cse_img_agg_flat.shape[0] < 1:
        return None

    argmin_keypoint = torch.zeros((N_PTS,), dtype=torch.int32, device=device)
    for p in range(N_PTS):
        # (N_FRAMES,D_CSE) , (N_VERTS,D_CSE) -> (N_FRAMES,)
        closest_vertices_p = get_closest_vertices_matrix(cse_img_agg_flat[p], mesh_cse_embedding)
        # ->(N_FRAMES, N_VERTS)
        geodesic_distances = geodesic_map[closest_vertices_p]
        argmin_keypoint[p] = torch.argmin(torch.sum(geodesic_distances * weight_mask_flat[p].reshape((N_FRAMES, 1)), dim=0), dim=0)

    cse_aggregated_images = torch.zeros((H, W, D_CSE), dtype=torch.float32, device=device)
    cse_aggregated_images = cse_aggregated_images.reshape(-1, D_CSE)
    cse_aggregated_images[cse_mask_flat > 0] = mesh_cse_embedding[argmin_keypoint]
    cse_aggregated_images = cse_aggregated_images.reshape(H, W, D_CSE).permute(2, 0, 1)
    return cse_aggregated_images
