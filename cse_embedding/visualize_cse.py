# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import rendering.visualization as viz
import utils.template_mesh_utils as templutil
from model.texture_models.model_cse import TextureCSEFixed


def visualize_cse_maps(images: torch.Tensor, cse_maps: torch.Tensor, cse_masks: torch.Tensor, valid_indices: list[int], cse_embedding_mesh: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """
    Returns:
        - rendered_cse_maps (BATCH, image_size, image_size, 3) NPUINT8
    """
    # Get a color template for the mesh
    X_texture = torch.tensor(templutil.get_mesh_2d_embedding_from_name("cse")).type(torch.float32)
    # Attach CSE vertex-embedding to vertex-color
    texture_model = TextureCSEFixed(verts_colors=X_texture / 255.0, verts_cse_embedding=cse_embedding_mesh, device=device)
    rendered_cse_maps = render_cse_maps_from_texture_model(texture_model, cse_maps, cse_masks, valid_indices, images, device)
    return rendered_cse_maps


def render_cse_maps_from_texture_model(
    texture_model: TextureCSEFixed,
    cse_maps: torch.Tensor,
    cse_masks: torch.Tensor,
    valid_indices: list[int],
    images: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """
    Args:
        - cse_maps (BATCH, D_CSE, H, W) TORCHFLOAT32
        - cse_masks (BATCH, H, W, 1) TORCHINT32
        - valid_indices (N,)
        - images (N_frames, H_img, W_img, 3) TORCHFLOAT32/TORCHUINT8 [0,1]

    Returns:
        - rendered_cse_maps (N_frames, H_img, W_img, 3) NPUINT8 [0,255]
    """
    N_frames, H_img, W_img, _ = images.shape

    rendered_cse_maps = np.zeros((N_frames, H_img, W_img, 3), dtype=np.uint8)

    # Generate the CSE video
    cse_maps = cse_maps.to(device).permute((0, 2, 3, 1))
    cse_masks_npy = cse_masks.cpu().numpy()
    BATCH, H, W, D_CSE = cse_maps.shape

    for j in range(N_frames):
        image_npy = (255 * images[j]).cpu().numpy().astype(np.uint8)
        if j in valid_indices:
            rendered_image = texture_model.render_cse_embeddings(cse_maps[j].reshape((-1, D_CSE))).reshape((H_img, W_img, 3))
            rendered_image = (rendered_image * 255).cpu().numpy().astype(np.uint8)
            rendered_cse_maps[j] = viz.blend_images(image_npy, rendered_image, 0.3, cse_masks_npy[j])
        else:
            rendered_cse_maps[j] = image_npy
    return rendered_cse_maps
