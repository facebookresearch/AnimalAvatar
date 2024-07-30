# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from pytorch3d.renderer import PerspectiveCameras
from smal import SMAL
from rendering.renderer import Renderer
from cse_embedding.functional_map import get_closest_vertices_matrix
from model.pose_models import compute_pose, PoseBase


class TextureCSEFixed(nn.Module):
    """
    A model which outputs the color from a CSE embedding,
    based on a fixed set of colors on a mesh and closest distance
    """

    def __init__(self, verts_colors: torch.Tensor, verts_cse_embedding: torch.Tensor, device: str):
        """
        Args:
            - verts_colors (N_vertices, 3) TORCHFLOAT32 [0,1]
            - verts_cse_embedding (N_vertices, D) TORCHFLOAT32
        """
        super(TextureCSEFixed, self).__init__()
        self.tag = "TextureCSEFixed"
        self.verts_colors = torch.nn.Parameter(verts_colors.to(device), requires_grad=False)
        self.verts_cse_embedding = torch.nn.Parameter(verts_cse_embedding.to(device), requires_grad=False)

    def render_cse_embeddings(self, cse_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - cse_embedding TORCHFLOAT32 (M,D)
        Return:
            - rendered_colors TORCHFLOAT32 [0,1] (M,3)
        """
        # Get the closest vertices
        closest_verts = get_closest_vertices_matrix(cse_embedding, self.verts_cse_embedding)
        # Return the color
        return self.verts_colors[closest_verts]

    def render_images(
        self,
        smal: SMAL,
        renderer: Renderer,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        cameras: PerspectiveCameras,
        cse_embedding_verts: torch.Tensor,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - rendered_colors (BATCH, H_rend, W_rend, 3) TORCHFLOAT32
            - rendered_masks (BATCH, H_rend, W_rend, 3) TORCHFLOAT32
        """
        if (vertices is None) or (faces is None):
            with torch.no_grad():
                vertices, _, faces = compute_pose(smal, pose_model, X_ind, X_ts)

        # Render cse image
        rendered_cse, rendered_cse_mask = renderer.render_cse(vertices, faces, cameras, cse_embedding_verts)
        rendered_cse = rendered_cse.squeeze(3)

        # Get texture from CSE->texture model
        bs, H, W, D_CSE = rendered_cse.shape

        # Get the closest vertices
        closest_verts = get_closest_vertices_matrix(rendered_cse.reshape(bs * H * W, D_CSE), self.verts_cse_embedding).reshape(bs, H, W, 3)

        rendered_colors = self.verts_colors[closest_verts]

        return rendered_colors, rendered_cse_mask
