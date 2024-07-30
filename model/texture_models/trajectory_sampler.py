# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from smal import scale_mesh_from_normal, get_smal_model
from rendering.renderer import Renderer
from model.pose_models import PoseBase, compute_pose


class TrajectoryPoseDeformed:
    """
    Trajectory sampler which find for each pixel 2 corresponding points on the upscaled (OUT) and downscaled (IN) mesh in world space
    """

    def __init__(self, scale: float, device: str):

        self.scale = scale
        self.device = device

        # Get the canonical vertices positions
        smal = get_smal_model(device=self.device)
        verts_canonical = smal.v_template.unsqueeze(0)
        faces_canonical = smal.faces.unsqueeze(0)

        # Get the upscaled and downscaled canonical meshes
        downscaled_canonical_vertices = scale_mesh_from_normal(verts_canonical, faces_canonical, scale=1.0)
        downscaled_canonical_meshes = Meshes(downscaled_canonical_vertices, faces_canonical)
        upscaled_canonical_vertices = scale_mesh_from_normal(verts_canonical, faces_canonical, scale=self.scale)
        upscaled_canonical_meshes = Meshes(upscaled_canonical_vertices, faces_canonical)

        self.upscaled_canonical_faces_verts_packed = upscaled_canonical_meshes.verts_packed()[upscaled_canonical_meshes.faces_packed()]
        self.downscaled_canonical_faces_verts_packed = downscaled_canonical_meshes.verts_packed()[downscaled_canonical_meshes.faces_packed()]

        # Scale SMAL mesh up and down -------------------------------------
        self.smal_upscaled = get_smal_model(device=self.device, scale_factor=self.scale)
        self.smal_downscaled = get_smal_model(device=self.device, scale_factor=1.0)

    def generate_point_map_in_out(
        self, renderer: Renderer, pose_model: PoseBase, X_ind: torch.Tensor, X_ts: torch.Tensor, cameras: PerspectiveCameras, vertices=None, faces=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - points_in_out: (BATCH, H_rend, W_rend, 2, 3) TORCHFLOAT32
            - valid_masks: (BATCH, H_rend, W_rend) TORCHBOOL
        """
        # Compute posed upscaled & downscaled template
        verts_upscaled, _, faces = compute_pose(self.smal_upscaled, pose_model, X_ind, X_ts)
        verts_downscaled, _, faces = compute_pose(self.smal_downscaled, pose_model, X_ind, X_ts)
        BATCH, N_verts, _ = verts_upscaled.shape

        # Find points hit on the raterized posed upscaled & downscaled mesh
        with torch.no_grad():
            faces_hit_up, baryc_coord_up, valid_mask_up = renderer.get_rasterization(verts_upscaled, faces, cameras)
            faces_hit_down, baryc_coord_down, valid_mask_down = renderer.get_rasterization(verts_downscaled, faces, cameras)

        # Get position in canonical space of points IN and OUT
        points_in = interpolate_face_attributes(faces_hit_up, baryc_coord_up, self.upscaled_canonical_faces_verts_packed.repeat(BATCH, 1, 1))
        points_out = interpolate_face_attributes(faces_hit_down, baryc_coord_down, self.downscaled_canonical_faces_verts_packed.repeat(BATCH, 1, 1))
        points_in_out = torch.concatenate([points_in, points_out], dim=3)

        valid_masks = valid_mask_up * valid_mask_down

        return points_in_out, valid_masks
