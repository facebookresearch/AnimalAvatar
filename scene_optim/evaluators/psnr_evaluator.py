# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from smal import SMAL
from losses import psnr
from rendering.renderer import Renderer
from model.pose_models import PoseBase
from model.texture_models import TextureBase
from model.texture_models.model_utils import render_images_wrapper


class PSNREvaluator:

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, cse_embedding_verts: torch.Tensor | None, image_size: int, device: str = "cuda"):
        """
        Args:
            - images (BATCH, H_rend, W_rend, 3) [0,1] TORCHFLOAT32
            - masks (BATCH, H_rend, W_rend, 1) {0,1} TORCHINT32
            - cse_embedding_verts (N_verts, D_CSE) TORCHFLOAT32
        """

        BATCH, H, W, _ = images.shape
        self.renderer = Renderer(image_size)
        self.cse_embedding_verts = cse_embedding_verts
        self.masks_bool = (masks.reshape(BATCH, H, W) > 0.0).type(torch.bool).to(device)
        self.masked_image = torch.where(self.masks_bool.unsqueeze(-1).expand(-1, -1, -1, 3), images, torch.zeros_like(images)).to(device)

    def evaluate(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        smal: SMAL,
        cameras: PerspectiveCameras,
        texture_model: TextureBase,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            - loss (N,) TORCHFLOAT32
        """

        # ->(BATCH,H,W,3) [0.<->1.]
        with torch.no_grad():
            rendered_colors, rendered_masks = render_images_wrapper(texture_model, smal, self.renderer, pose_model, X_ind, X_ts, cameras, vertices, faces, self.cse_embedding_verts)
            rendered_colors[~(rendered_masks[..., 0] > 0.0)] = 0.0

        BATCH, H, W, _ = rendered_colors.shape
        device = rendered_colors.device

        # Compute peak signal-to-noise ratio
        loss = psnr(rendered_colors, self.masked_image[X_ind])

        # Compute masked PSNR
        loss_masked = torch.zeros((BATCH,), dtype=torch.float32, device=device)
        for i in range(BATCH):
            # ->(1,3,NPTS)
            masked_gt = self.masked_image[X_ind[i]][self.masks_bool[X_ind[i]]].unsqueeze(0)
            masked_pred = rendered_colors[i][self.masks_bool[X_ind[i]]].unsqueeze(0)
            loss_masked[i] = psnr(masked_gt, masked_pred)[0]

        return loss, loss_masked
