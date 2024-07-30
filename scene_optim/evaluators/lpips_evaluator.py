# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import lpips
from pytorch3d.renderer import PerspectiveCameras
from smal import SMAL
from rendering.renderer import Renderer
from model.pose_models import PoseBase
from model.texture_models import TextureBase
from model.texture_models.model_utils import render_images_wrapper
from utils.torch_util import SuppressAllWarnings

with SuppressAllWarnings():
    GLOBAL_LPIPS = lpips.LPIPS(net="vgg", verbose=False)


class LPIPSEvaluator:

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, cse_embedding_verts: torch.Tensor, image_size: int, device: str = "cuda"):
        """
        Args:
            - images (BATCH, H_rend, W_rend, 3) [0,1] TORCHFLOAT32
            - masks (BATCH, H_rend, W_rend, 1) {0,1} TORCHINT32
            - cse_embedding_verts (N_verts, D_CSE) TORCHFLOAT32
        """

        self.device = device
        self.renderer = Renderer(image_size)
        self.cse_embedding_verts = cse_embedding_verts

        # Loss LPIPS
        self.loss_lpips = GLOBAL_LPIPS.to(device)

        BATCH, H, W, _ = images.shape
        masks_bool = (masks.reshape(BATCH, H, W) > 0.0).type(torch.bool)
        self.images = images
        self.images_masked = torch.where(masks_bool.unsqueeze(-1).expand(-1, -1, -1, 3), images, torch.zeros_like(images))

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
            - loss (BATCH,) TORCHFLOAT32
        """

        with torch.no_grad():
            rendered_colors, rendered_masks = render_images_wrapper(texture_model, smal, self.renderer, pose_model, X_ind, X_ts, cameras, vertices, faces, self.cse_embedding_verts)
            rendered_colors[~(rendered_masks[..., 0] > 0.0)] = 0.0

            loss = self.loss_lpips.forward(2 * self.images_masked[X_ind].permute((0, 3, 1, 2)) - 1, 2 * rendered_colors.permute((0, 3, 1, 2)) - 1).reshape((-1,))

        return loss
