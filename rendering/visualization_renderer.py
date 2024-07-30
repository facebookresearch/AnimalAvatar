# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
from smal import SMAL
from model.pose_models import PoseBase, PoseCanonical, compute_pose
from rendering.renderer import Renderer
import rendering.visualization as viz
from model.texture_models.model_utils import render_images_wrapper
from model.texture_models import TextureBase

## Set of high-level visualization implying a rendering call


def basic_visualization(pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, smal: SMAL, renderer: Renderer, cameras: PerspectiveCameras, texture: torch.Tensor | None = None) -> np.ndarray:
    """
    Visualize posed meshes with vertex-based texture.

    Args:
        - X_ts [N,] TORCHINT64
        - X_ind [N,] TORCHFLOAT32
        - renderer [H,W]
        - texture: (N_verts, 3) or (BATCH, N_verts, 3) TORCHFLOAT32 [0,255]
    Return:
        - rendered (N,H,W,3) NPUINT8 [0,255]
    """
    # Get predicted reconstruction
    with torch.no_grad():
        vertices, _, faces = compute_pose(smal, pose_model, X_ind, X_ts)

    # Texture (N_verts, 3) or (BATCH, N_verts, 3)
    if texture is not None:
        if texture.dim() == 2:
            texture_batch = texture.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        else:
            texture_batch = texture
    else:
        texture_batch = None

    # Compute the visualization
    rendered = (renderer.render_visualization(vertices, faces, cameras, texture=texture_batch).cpu().numpy().transpose((0, 2, 3, 1))).astype(np.uint8)
    return rendered


def blend_visualization(
    pose_model: PoseBase,
    X_ind: torch.Tensor,
    X_ts: torch.Tensor,
    smal: SMAL,
    input_imgs: np.ndarray,
    renderer: Renderer,
    cameras: PerspectiveCameras,
    texture: torch.Tensor | None = None,
    alpha: float = 0.3,
) -> np.uint8:
    """
    Same as basic_visualization, blended with GT images.

    Args:
        - X_ts [N,] TORCHINT64
        - X_ind [N,] TORCHFLOAT32
        - input_imgs [N,H,W,3] NPFLOAT32 [0,1] (H,W)-> SAME AS RENDERER
        - cameras PerspectiveCamera, batch of N cameras
        - renderer [H,W]
    Return:
        - rendered (N,H,W,3) NPUINT8 [0,255]
    """
    rendered = basic_visualization(pose_model, X_ind, X_ts, smal, renderer, cameras, texture).astype(np.float32) / 255.0
    return (viz.blend_N_images(input_imgs, rendered, alpha) * 255).astype(np.uint8)


def texture_render_visualization(
    pose_model: PoseBase,
    X_ind: torch.Tensor,
    X_ts: torch.Tensor,
    smal: SMAL,
    renderer: Renderer,
    cameras: PerspectiveCameras,
    cse_embedding_mesh: torch.Tensor,
    texture_model: TextureBase,
    white_bg: bool = False,
) -> np.ndarray:
    """
    Visualize posed meshes with texture from a texture_model.
    """
    with torch.no_grad():
        output_texture_dark, output_texture_dark_mask = render_images_wrapper(texture_model, smal, renderer, pose_model, X_ind, X_ts, cameras, cse_embedding_verts=cse_embedding_mesh)
    if white_bg:
        output_texture_dark[~(output_texture_dark_mask[..., 0] > 0.0)] = 1.0
    else:
        output_texture_dark[~(output_texture_dark_mask[..., 0] > 0.0)] = 0.0

    output_texture_dark = (255 * output_texture_dark.cpu().numpy()).astype(np.uint8)

    return output_texture_dark


def global_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    pose_model: PoseBase,
    X_ind: torch.Tensor,
    X_ts: torch.Tensor,
    smal: SMAL,
    texture_model: TextureBase,
    cse_embedding_mesh: torch.Tensor,
    renderer: Renderer,
    cameras: PerspectiveCameras,
    texture: torch.Tensor | None = None,
    alpha: float = 0.3,
):
    """
    Main visualization, which concatenate multiple renderings.

    Args:
        - images (BATCH, H_rend, W_rend, 3) TORCHFLOAT32 [0,1]
        - masks (BATCH, H_rend, W_rend, 1) TORCHINT32 {0,1}
        - cse_embedding_mesh (N_verts, D_CSE) TORCHFLOAT32
        - cameras (BATCH,) PerspectiveCameras
    """
    images_gt = (255 * images).cpu().numpy().astype(np.uint8)
    masks_npint = masks.cpu().numpy().astype(np.int32)
    images_masked = np.where(masks_npint == 0, images_gt, np.zeros_like(images_gt))

    texture_render = texture_render_visualization(pose_model, X_ind, X_ts, smal, renderer, cameras, cse_embedding_mesh, texture_model)

    texture_incontextM_render = viz.blend_N_images(images_masked, texture_render, alpha=0.0)

    pose_render = basic_visualization(pose_model, X_ind, X_ts, smal, renderer, cameras, texture=texture)

    pose_incontext_render = blend_visualization(pose_model, X_ind, X_ts, smal, images[X_ind].cpu().numpy(), renderer, cameras, texture=texture, alpha=alpha)

    return (
        images_gt,
        texture_incontextM_render,
        pose_incontext_render,
        texture_render,
        pose_render,
    )
