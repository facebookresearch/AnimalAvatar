# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer import PerspectiveCameras
import scene_optim.losses_optim as lopt
from smal import SMAL
from rendering.renderer import Renderer
from model.pose_models import PoseBase, compute_pose


class LossOptim:

    def __init__(
        self,
        smal: SMAL,
        device: str,
        cameras: PerspectiveCameras,
        images: torch.Tensor,
        masks: torch.Tensor,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        cse_keypoints_xy: torch.Tensor,
        cse_keypoints_vert_id: torch.Tensor,
        cse_valid_indices: torch.Tensor,
        cse_keypoints_max_indice: torch.Tensor,
        sparse_keypoints: torch.Tensor,
        sparse_keypoints_scores: torch.Tensor,
        config_chamfer: dict,
        config_color: dict,
        image_size: int,
        batch_size: int,
        weight_dict: dict,
    ):
        """
        Args:
            - cameras PerspectiveCameras (BATCH,)

            - images (BATCH, H_render, W_render, 3) TORCHFLOAT32 [0,1]
            - mask (BATCH, H_render, W_render, 1) TORCHINT32 {0,1}

            - keyp2d (BATCH, N_kps, 2) TORCHFLOAT32 [0,1]
            - cse_embeddings (BATCH, N_kps, 16) TORCHFLOAT32
            - closest_verts (BATCH, N_kps) TORCHINT64

            - valid_indices_cse (N_valid,) TORCHINT32
            - cse_imgs (N_valid, D_CSE, H_IMG, W_IMG) TORCHFLOAT32
            - cse_masks (N_valid, H_IMG, W_IMG, 1) TORCHINT32
            - smal_verts_cse_embedding (N_verts, D_CSE) TORCHFLOAT32

            - sparse_keypoints (BATCH, N_KPS, 2) TORCHFLOAT32 [0,1]
            - sparse_keypoints_scores (BATCH, N_KPS, 1) TORCHFLOAT32 [0,1]

        """
        self.smal = smal
        self.device = device
        self.renderer = Renderer(image_size=image_size)
        self.weight_dict = weight_dict

        self.loss_dict = {
            # The only loss to be active even if weight is 0
            "l_optim_color": lopt.LossOptimColor(self.smal, self.device, self.renderer, cameras, images, masks, **config_color),
            "l_optim_chamfer": lopt.LossOptimChamfer(masks, self.smal, self.device, cameras, self.renderer, **config_chamfer) if self.weight_dict["l_optim_chamfer"] > 0 else None,
            "l_optim_cse_kp": (
                lopt.LossOptimCSEKp(self.smal, self.device, cse_keypoints_xy, cse_keypoints_vert_id, cse_valid_indices, cse_keypoints_max_indice, cameras, image_size)
                if self.weight_dict["l_optim_cse_kp"] > 0
                else None
            ),
            "l_optim_sparse_kp": lopt.LossOptimSparseKp(self.smal, self.device, sparse_keypoints, sparse_keypoints_scores, cameras, image_size) if self.weight_dict["l_optim_sparse_kp"] > 0 else None,
            "l_laplacian_reg": lopt.LossLaplacianReg(self.smal, self.device) if self.weight_dict["l_laplacian_reg"] > 0 else None,
            "l_tv_reg": lopt.LossOptimTVReg(X_ind, X_ts) if self.weight_dict["l_tv_reg"] > 0 else None,
            "l_arap_reg": lopt.LossOptimArap(self.smal, batch_size, self.device) if self.weight_dict["l_arap_reg"] > 0 else None,
            "l_arap_fast_reg": lopt.LossOptimArapFast(self.smal, batch_size, self.device) if self.weight_dict["l_arap_fast_reg"] > 0 else None,
        }

    def forward_batch(self, *args, **kwargs) -> dict[str, torch.Tensor]:

        vertices, faces = self.compute_pose(*args)

        if vertices.isnan().sum() > 0:
            raise Exception("mesh vertices coords are NaN")

        kwargs.update({"vertices": vertices, "faces": faces})

        return_dict = {"total": 0.0}

        for l_key in self.loss_dict:
            if (self.weight_dict[l_key] > 0) or (l_key == "l_optim_color"):
                value = self.loss_dict[l_key].forward_batch(*args, **kwargs)
                return_dict[l_key] = value
                return_dict["total"] += self.weight_dict[l_key] * value

        return return_dict

    def compute_pose(self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        vertices, _, faces = compute_pose(self.smal, pose_model, X_ind, X_ts)
        return vertices, faces

    def no_shape_training(self) -> bool:
        """
        Should we train both models (pose, texture) or only texture model.
        """
        for key in ["l_optim_chamfer", "l_optim_cse_kp", "l_optim_sparse_kp", "l_laplacian_reg", "l_tv_reg", "l_arap_reg", "l_arap_fast_reg"]:
            if self.weight_dict[key] > 0:
                return False
        return True
