# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase
from scene_optim.evaluators import IoUEvaluator, PSNREvaluator, LPIPSEvaluator
from .callback_general import CallbackClass
from smal import SMAL


class CallbackEval(CallbackClass):

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, cameras: PerspectiveCameras, smal: SMAL, verts_cse_embeddings: torch.Tensor, image_size: int, device: str = "cuda"):
        """
        Args:
            - images (BACTH, H_rend, W_rend, 3)  TORCHFLOAT32 [0,1]
            - masks (BATCH, IMG_H, IMG_W, 1) TORCHFLOAT32 {0,1}
            - cameras PerspectiveCameras
            - smal
            - verts_cse_embeddings (N_verts, D_CSE) TORCHFLOAT32
            - image_size INT
        """
        self.cameras = cameras
        self.smal = smal
        self.iou_evalr = IoUEvaluator(masks, image_size=image_size, device=device)
        self.psnr_evalr = PSNREvaluator(images, masks, verts_cse_embeddings, image_size, device)
        self.lpips_evalr = LPIPSEvaluator(images, masks, verts_cse_embeddings, image_size, device)

    def call(self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, return_individual: bool = False) -> tuple[dict[str, float], str]:

        with torch.no_grad():
            vertices, _, faces = compute_pose(self.smal, pose_model, X_ind, X_ts)
            l_iou = self.iou_evalr.evaluate(pose_model, X_ind, X_ts, self.smal, self.cameras[X_ind], vertices, faces)
            l_psnr, l_psnrm = self.psnr_evalr.evaluate(pose_model, X_ind, X_ts, self.smal, self.cameras[X_ind], texture_model, vertices, faces)
            l_lpips = self.lpips_evalr.evaluate(pose_model, X_ind, X_ts, self.smal, self.cameras[X_ind], texture_model)

        l_iou_mean = torch.mean(l_iou).item()
        l_psnr_mean = torch.mean(l_psnr).item()
        l_psnrm_mean = torch.mean(l_psnrm).item()
        l_lpips_mean = torch.mean(l_lpips).item()

        str_log = "IoU: {:.3f} PSNR: {:.3f} PSNRM: {:.3f} LPIPS: {:.3f}".format(l_iou_mean, l_psnr_mean, l_psnrm_mean, l_lpips_mean)

        if return_individual:
            return {"l_iou": l_iou.cpu(), "l_psnr": l_psnr.cpu(), "l_psnrm": l_psnrm.cpu(), "l_lpips": l_lpips.cpu()}, str_log
        else:
            return {"l_iou": l_iou_mean, "l_psnr": l_psnr_mean, "l_psnrm": l_psnrm_mean, "l_lpips": l_lpips_mean}, str_log
