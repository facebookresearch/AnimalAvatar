# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import PerspectiveCameras
from model.pose_models import compute_pose, PoseBase
from smal.smal_torch import SMAL
from rendering.renderer import Renderer


class TextureModelTags:
    TextureCSEFixed = "TextureCSEFixed"
    TextureDuplex = "TextureDuplex"


class TextureBase(nn.Module):
    """
    A model which outputs the color from a CSE embedding,
    based on a fixed set of colors on a mesh and closest distance
    """

    def __init__(self, *args, **kwargs):
        super(TextureBase, self).__init__()

    def render_images(
        self,
        smal: SMAL,
        renderer: Renderer,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        cameras: PerspectiveCameras,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - rendered_images (BATCH, H_rend, W_rend, 3) TORCHFLOAT32 [0,1]
            - rendered_masks (BATCH, H_rend, W_rend, 3) TORCHFLOAT32
        """
        raise NotImplementedError


def render_images_wrapper(
    model: TextureBase,
    smal: SMAL,
    renderer: Renderer,
    pose_model: PoseBase,
    X_ind: torch.Tensor,
    X_ts: torch.Tensor,
    cameras: PerspectiveCameras,
    vertices: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
    cse_embedding_verts: torch.Tensor | None = None,
):
    """
    Returns:
        - rendered_images (BATCH, H, W, 3) TORCHFLOAT32 [0,1]
        - rendered_masks (BATCH, H_rend, W_rend, 3) TORCHFLOAT32
    """

    if (vertices is None) or (faces is None):
        vertices, _, faces = compute_pose(smal, pose_model, X_ind, X_ts)

    if model.tag == TextureModelTags.TextureCSEFixed:
        return model.render_images(
            smal=smal,
            renderer=renderer,
            pose_model=pose_model,
            X_ind=X_ind,
            X_ts=X_ts,
            cameras=cameras,
            cse_embedding_verts=cse_embedding_verts,
            vertices=vertices,
            faces=faces,
        )

    elif model.tag == TextureModelTags.TextureDuplex:
        return model.render_images(
            smal=smal,
            renderer=renderer,
            pose_model=pose_model,
            X_ind=X_ind,
            X_ts=X_ts,
            cameras=cameras,
            vertices=vertices,
            faces=faces,
        )

    raise Exception("Unknown texture model: {}".format(model.tag))
