# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from smal import SMAL
from losses import iou
from rendering.renderer import Renderer
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase


class IoUEvaluator:

    def __init__(self, masks: torch.Tensor, image_size: int, device: str = "cuda"):
        """
        Args:
            - masks TORCHINT32 {0,1} (BATCH, IMG_H, IMG_W, 1)
        """
        self.device = device
        self.renderer = Renderer(image_size=image_size)
        self.masks = masks.to(device)

    def evaluate(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        smal: SMAL,
        cameras: PerspectiveCameras,
        texture_model: TextureBase | None = None,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if (vertices is None) or (faces is None):
            with torch.no_grad():
                vertices, _, faces = compute_pose(smal, pose_model, X_ind, X_ts)

        with torch.no_grad():
            rendered_silhouette = self.renderer.render_silhouette_partition(vertices, faces, cameras).permute((0, 2, 3, 1))
            rendered_silhouette = (rendered_silhouette > 0.7).type(torch.int32)

        # Compute IoU
        return iou(rendered_silhouette, self.masks[X_ind]).reshape((-1,))
