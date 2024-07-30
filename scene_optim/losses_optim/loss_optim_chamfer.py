# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras, get_screen_to_ndc_transform
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from smal import SMAL
from model.pose_models import PoseBase, compute_pose
from rendering.renderer import Renderer
from losses import sample_points_from_meshes
from model.texture_models import TextureBase


def sample_point_cloud_from_mask(masks: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Args:
        - masks (BATCH, H, W, 1) TORCHINT32
    Returns:
        - pc_gt (BATCH, MAX_L, 2) TORCHFLOAT32 (follows SCREEN coordinate system)
        - max_l_gt: MAX_L
        - length_gt: (BATCH,) number of points per batch
    """
    BATCH, _, _, _ = masks.shape
    device = masks.device

    # (BATCH, H, W, 1) -> (N_PTS, 4)
    X_gt = masks.nonzero()

    count_gt, length_gt = torch.unique(X_gt[:, 0], return_counts=True)
    max_l_gt = torch.max(length_gt)

    pc_gt = torch.zeros((BATCH, max_l_gt, 2), dtype=torch.float32, device=device)

    for k in range(BATCH):
        pc_gt[k, : length_gt[k]] = X_gt[X_gt[:, 0] == count_gt[k]][:, [2, 1]]

    return pc_gt, max_l_gt, length_gt


class LossOptimChamfer:

    def __init__(
        self,
        mask: torch.Tensor,
        smal: SMAL,
        device: str,
        cameras: PerspectiveCameras,
        renderer: Renderer,
        N_pts_sampled: int,
        resample_freq: int,
        weight_invisible: float = 1.0,
    ):
        """
        Chamfer distance optimizer.
        """
        self.i = 0
        self.sampling_seed = torch.randint(low=0, high=1000000, size=(1,)).item()

        self.cameras = cameras
        self.renderer = renderer
        self.smal = smal
        self.device = device
        self.mask = mask
        self.img_size = self.renderer.image_size
        self.chamfer_n_pts_sampled = N_pts_sampled
        self.resample_freq = resample_freq
        self.weight_invisible = weight_invisible

    def compute_visibility_weights(self, pts_faces_id: torch.Tensor, X_ind: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            - weights (BATCH, N_PTS) TORCHFLOAT32
        """
        # Compute visibility map for 2D points -> (BATCH, N_vertices,1) TORCHINT32
        _, faces_visibility_map = self.renderer.get_visibility_map(vertices, faces, self.cameras[X_ind])

        # Get the visibility for all sampled points on the mesh ->(BATCH, N_SAMPLED)
        pts_faces_id = (pts_faces_id - (torch.arange(0, faces.shape[0], device=pts_faces_id.device) * faces.shape[1]).reshape((-1, 1))).type(torch.int64)
        pts_faces_visibility = torch.gather(faces_visibility_map, dim=1, index=pts_faces_id)

        # Compute weights
        weights = torch.where(pts_faces_visibility > 0, 1.0, self.weight_invisible)

        return weights

    def forward_batch(
        self, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, texture_model: TextureBase, vertices: torch.Tensor | None = None, faces: torch.Tensor | None = None
    ) -> torch.Tensor:

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(self.smal, pose_model, X_ind, X_ts)

        N, H, W, _ = self.mask[X_ind].shape

        self.i += 1
        if self.i % self.resample_freq == 0:
            self.sampling_seed = torch.randint(low=0, high=1000000, size=(1,)).item()

        # randomly sample point on the global mesh and project them to NDC space
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.sampling_seed)
            pts_3d, pts_faces_id = sample_points_from_meshes(Meshes(verts=vertices, faces=faces), num_samples=self.chamfer_n_pts_sampled)
            points_predicted_ndc = self.cameras[X_ind].transform_points_ndc(pts_3d, eps=1e-4)[:, :, :2]

        # get the ground truth mask point-cloud
        pc_gt, max_l_gt, length_gt = sample_point_cloud_from_mask(self.mask[X_ind] > 0.5)

        screen_to_ndc_transform = get_screen_to_ndc_transform(self.cameras[X_ind], with_xyflip=True, image_size=(self.img_size, self.img_size))

        pc_gt_transformed = screen_to_ndc_transform.transform_points(torch.cat([pc_gt, torch.ones((N, max_l_gt, 1), device=pc_gt.device, dtype=torch.float32)], dim=2))[..., :2]

        # Compute visibility weights for all sampled points on the mesh
        weights = self.compute_visibility_weights(pts_faces_id, X_ind, vertices, faces)

        # Compute the chamfer distance (x: loss on mesh points, y: loss on GT mask points)
        loss_x, loss_y = chamfer_distance(x=points_predicted_ndc, y=pc_gt_transformed, y_lengths=length_gt, batch_reduction=None, point_reduction=None, norm=2)[0]
        chamf_x = (weights * loss_x).sum(1) / weights.sum(1)
        chamf_y = loss_y.sum(1) / length_gt.clamp(min=1)
        loss = (chamf_x + chamf_y).reshape((-1, 1))

        return loss
