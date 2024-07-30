# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import lpips
import torch
from typing import Literal
from pytorch3d.renderer import PerspectiveCameras
from smal import SMAL
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase
from model.texture_models.model_utils import render_images_wrapper
from rendering.renderer import Renderer
import utils.img_utils as imgutil
from utils.torch_util import SuppressAllWarnings


def image_comparison(lpips_loss: lpips.LPIPS, image_gt: torch.Tensor, image_pred: torch.Tensor, method: Literal["l2", "lpips"], resolution: int, mask_gt: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        - images_gt (BATCH, H_gt, W_gt, 3) TORCHFLOAT32 [0,1]
        - mask_gt (BATCH, H_gt, W_gt, 1) TORCHBOOL
        - images_rend (BATCH, H_rend, W_rend, 3) TORCHFLOAT32 [0,1]
    Returns:
        - loss (BATCH,1) TORCHFLOAT32
    """

    BATCH, H, W, _ = image_gt.shape
    _, H_PRED, W_PRED, _ = image_pred.shape

    if mask_gt is not None:
        assert (mask_gt.shape[1] == H) and (mask_gt.shape[2] == W)

    # Resize GT image and mask to work resolution
    if (H != resolution) or (W != resolution):
        image_gt_loss = imgutil.resize_torch(image_gt, (resolution, resolution), mode="bilinear")
        if mask_gt is not None:
            mask_gt_loss = imgutil.resize_torch_bool(mask_gt, (resolution, resolution))
    else:
        image_gt_loss = image_gt

    # Resize rgb prediction to work resolution
    if (H_PRED != resolution) or (W_PRED != resolution):
        image_pred_loss = imgutil.resize_torch(image_pred, (resolution, resolution))
    else:
        image_pred_loss = image_pred

    # Compute loss (l2 or lpips) ---
    if method == "l2":
        if mask_gt is not None:
            output_loss = (torch.sum(mask_gt_loss * (image_gt_loss - image_pred_loss) ** 2, dim=[1, 2, 3]) / (torch.sum(mask_gt_loss, dim=[1, 2, 3]) + 1e-6)).reshape((-1, 1))
        else:
            output_loss = torch.mean((image_gt_loss - image_pred_loss) ** 2, dim=[1, 2, 3]).reshape((-1, 1))

    elif method == "lpips":
        # gt and prediction images must be: (BATCH,C,H,W) TORCHFLOAT32 [-1,1]
        output_loss = lpips_loss.forward(2 * image_gt_loss.permute((0, 3, 1, 2)) - 1, 2 * image_pred_loss.permute((0, 3, 1, 2)) - 1)

    else:
        raise Exception("Unknown method: {}".format(method))

    return output_loss.reshape(-1, 1)


class LossOptimColor:
    """
    Args:
        - smal SMAL model
        - images (BATCH, H_img, W_img, 3) TORCHFLOAT32 [0,1]
    """

    def __init__(
        self,
        smal: SMAL,
        device: str,
        renderer: Renderer,
        cameras: PerspectiveCameras,
        images: torch.Tensor,
        masks: torch.Tensor,
        n_rays_per_image: int,
        resolution_loss: int,
    ):
        """
        Args:
            - cameras PerspectiveCameras (BATCH,)
            - images (BATCH, H_render, W_render, 3) TORCHFLOAT32 [0,1]
            - mask (BATCH, H_render, W_render, 1) TORCHINT32 {0,1}
        """

        self.smal = smal
        self.device = device
        self.renderer = renderer
        self.cameras = cameras

        # if n_rays_per_image is -1, render full image instead of a set of rays
        self.n_rays_per_image = n_rays_per_image
        self.render_full_image = self.n_rays_per_image == -1

        self.images = images
        self.masks = masks
        self.masked_image = torch.where(masks.expand(-1, -1, -1, 3) > 0, images, torch.zeros_like(images))
        self.BATCH_FULL, self.H, self.W, _ = self.images.shape

        # LPIPS loss
        with SuppressAllWarnings():
            self.loss_lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device)

        # Image comparison loss resolution
        self.image_renderer = Renderer(image_size=resolution_loss)
        self.resolution_loss = resolution_loss

    def forward_batch(self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, vertices: torch.Tensor | None = None, faces: torch.Tensor | None = None):

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(self.smal, pose_model, X_ind, X_ts)

        # Sample a batch of rays and render
        if not self.render_full_image:

            # (BATCH, N_RAYS, 3), (BATCH, N_RAYS), (BATCH)
            rendered_rays, sampled_indices, max_indices = texture_model.render_image_sampling(
                self.n_rays_per_image, self.smal, self.image_renderer, pose_model, X_ind, X_ts, self.cameras[X_ind], vertices, faces
            )
            BATCH, N_RAYS, _ = rendered_rays.shape
            device = rendered_rays.device

            # ->(BATCH,N_RAYS, 3)
            images_flat = self.images[X_ind].reshape((BATCH, -1, 3))
            images_flat_rend = torch.gather(images_flat, dim=1, index=sampled_indices.reshape(BATCH, -1, 1).repeat(1, 1, 3))

            # Generate mask for summation
            mask = torch.zeros((BATCH, N_RAYS, 3), dtype=torch.float32, device=device)
            for i in range(BATCH):
                mask[i, : max_indices[i]] = 1

            # L2 error
            output_loss = torch.sum(mask * (rendered_rays - images_flat_rend) ** 2, dim=[1, 2]) / (max_indices + 1e-6)
            output_loss = output_loss.reshape((-1, 1))
            return output_loss

        # Render full predicted image
        else:
            rendered_images, rendered_masks = render_images_wrapper(texture_model, self.smal, self.image_renderer, pose_model, X_ind, X_ts, self.cameras[X_ind], vertices, faces)
            output_loss = image_comparison(self.loss_lpips, self.masked_image[X_ind], rendered_images, "lpips", self.resolution_loss)
            return output_loss
