# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import os
from pytorch3d.structures import Meshes
from config.keys import Keys

# ---
from smal.smal_utils import get_global_transf_from_kinematic_transf
from smal.initialize_smal import process_original_smal, CANONICAL_MODEL_JOINTS_REFINED


class SMAL(nn.Module):
    """
    PyTorch implementation of SMAL.
        See BITE code: "https://github.com/runa91/bite_release"
    """

    def __init__(self, buffer: dict, mode_original: bool = False, scale_factor: float = 1.0):
        super(SMAL, self).__init__()

        self.mode_original = mode_original

        # betas_scale_mask TORCHFLOAT32 (7, 105)
        self.register_buffer("betas_scale_mask", buffer["betas_scale_mask"])

        # faces TORCHINT64 (7774, 3)
        self.register_buffer("faces", buffer["faces"])

        # Scaling mesh
        if scale_factor == 1.0:
            # Mean template vertices (TORCHFLOAT32 (3889, 3))
            self.register_buffer("v_template", buffer["v_template"])
        else:
            v_template_scale = scale_mesh_from_normal(buffer["v_template"].unsqueeze(0), self.faces.unsqueeze(0), scale=scale_factor)
            self.register_buffer("v_template", v_template_scale.squeeze())

        # Size of mesh (Number of vertices, 3)
        self.size = [self.v_template.shape[0], 3]

        # Shape blend shape basis (TORCHFlOAT32, (78, 11667)) [in practice sliced to (30, 11667)]
        self.register_buffer("shapedirs", buffer["shapedirs"])

        # Regressor for joint locations given shape (TORCHFLOAT32, (3889,35))
        self.register_buffer("joint_regressor", buffer["joint_regressor"])

        # Pose blend shape basis (TORCHFLOAT32, (306, 11667))
        self.register_buffer("posedirs", buffer["posedirs"])

        # indices of parents for each joints (NPINT32, (35,))
        self.parents = buffer["kintree_table"].numpy()

        # LBS weights (TORCHFLOAT32, (3889, 35))
        self.register_buffer("weights", buffer["weights"])

        # Symmetric deformation
        self.register_buffer("inds_back", buffer["inds_back"])
        self.n_center = buffer["n_center"]
        self.n_left = buffer["n_left"]
        self.sl = buffer["s_left"]

    def get_symetric_vertex_offset(self, vert_off_compact: torch.Tensor, device: str) -> torch.Tensor:
        """
        Args:
            - vert_off_compact (BATCH,5901) TORCHFLOAT32
        Returns:
            - vertex_offsets (BATCH, 3889, 3) TORCHFLOAT32
        """

        # vert_off_compact (BATCH, 2*self.n_center + 3*self.n_left)
        zero_vec = torch.zeros((vert_off_compact.shape[0], self.n_center)).to(device)

        half_vertex_offsets_center = torch.stack((vert_off_compact[:, : self.n_center], zero_vec, vert_off_compact[:, self.n_center : 2 * self.n_center]), axis=1)

        half_vertex_offsets_left = torch.stack(
            (
                vert_off_compact[:, self.sl : self.sl + self.n_left],
                vert_off_compact[:, self.sl + self.n_left : self.sl + 2 * self.n_left],
                vert_off_compact[:, self.sl + 2 * self.n_left : self.sl + 3 * self.n_left],
            ),
            axis=1,
        )

        half_vertex_offsets_right = torch.stack(
            (
                vert_off_compact[:, self.sl : self.sl + self.n_left],
                -vert_off_compact[:, self.sl + self.n_left : self.sl + 2 * self.n_left],
                vert_off_compact[:, self.sl + 2 * self.n_left : self.sl + 3 * self.n_left],
            ),
            axis=1,
        )

        half_vertex_offsets_tot = torch.cat((half_vertex_offsets_center, half_vertex_offsets_left, half_vertex_offsets_right), dim=2)

        # ->(BATCH, 3889, 3)
        vertex_offsets = torch.index_select(half_vertex_offsets_tot, dim=2, index=self.inds_back.to(half_vertex_offsets_tot.device)).permute((0, 2, 1))

        return vertex_offsets

    def __call__(
        self, beta: torch.Tensor, betas_limbs: torch.Tensor, pose: torch.Tensor, trans: torch.Tensor, vert_off_compact: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Linear-Blend-Skinning.
        """
        device = beta.device
        nBetas = beta.shape[1]
        num_batch = pose.shape[0]

        # Add possibility to have additional vertex offsets.
        if vert_off_compact is None:
            vertex_offsets = torch.zeros_like(self.v_template)
        else:
            vertex_offsets = self.get_symetric_vertex_offset(vert_off_compact, device)

        # 1. Add shape blend shapes. ->(BATCH, N_VERTS, 3)
        v_shape_blend_shape = torch.matmul(beta, self.shapedirs[:nBetas, :]).reshape(-1, self.size[0], self.size[1])
        v_shaped = self.v_template + v_shape_blend_shape + vertex_offsets

        # 2. Infer shape-dependent joint locations.
        # (BATCH, N_VERTS, 3) @ (N_VERTS, N_JOINTS) ->(BATCH, N_JOINTS, 3)
        joint_locations = torch.einsum("nij,il->nlj", v_shaped, self.joint_regressor)

        # 3. Add pose blend shapes and ignore global rotation.
        pose_feature = (pose[:, 1:, :, :] - torch.eye(3, device=device)).reshape(-1, 306)
        v_pose_blend_shape = torch.matmul(pose_feature, self.posedirs).reshape(-1, self.size[0], self.size[1])
        v_posed = v_pose_blend_shape + v_shaped

        # Add corrections of bone lengths to the template.
        # ->(NBATCH, 105)
        betas_scale = torch.exp(betas_limbs @ self.betas_scale_mask.to(betas_limbs.device))
        # ->(NBATCH, 35, 3)
        scaling_factors = betas_scale.reshape(-1, 35, 3)
        scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

        # 4. Get the global joint location (BATCH, 35, 3) + relative joint transformations for LBS. (BATCH, 35, 4, 4)
        A = get_global_transf_from_kinematic_transf(pose, joint_locations, scale_factors_3x3, self.parents)

        # Recenter the mesh (original SMAL centered at beginning of the tail).
        if not self.mode_original:
            kp1, kp2 = 37, 1808
            v_center = (v_posed[:, kp1, None] + v_posed[:, kp2, None]) / 2
            v_posed = v_posed - v_center

        # 5. skinning-weights. ->(BATCH, 3889, 35):
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

        # The weighted transformation to apply on v_posed.  ->(BATCH, 3889, 4, 4)
        T = torch.matmul(W, A.reshape(num_batch, 35, 16)).reshape(num_batch, -1, 4, 4)

        # -> (BATCH, 3889, 3)
        v_posed_homo = torch.cat([v_posed, torch.ones((num_batch, v_posed.shape[1], 1), device=device)], 2)
        verts = torch.matmul(T, v_posed_homo.unsqueeze(-1))[:, :, :3, 0]

        # 6. Apply translation. ->(BATCH, 3889, 3)
        verts = verts + trans[:, None, :]

        # 7. Get joint positions. ->(BATCH, 35, 3)
        joints = torch.einsum("nij,il->nlj", verts, self.joint_regressor)

        joints = torch.cat(
            [
                joints,
                verts[:, None, 1863],  # end_of_nose
                verts[:, None, 26],  # chin
                verts[:, None, 2124],  # right ear tip
                verts[:, None, 150],  # left ear tip
                verts[:, None, 3055],  # left eye
                verts[:, None, 1097],  # right eye
                verts[:, None, 1330],  # front paw, right
                verts[:, None, 3282],  # front paw, left
                verts[:, None, 1521],  # back paw, right
                verts[:, None, 3473],  # back paw, left
                verts[:, None, 6],  # throat
                verts[:, None, 20],  # withers
            ],
            dim=1,
        )

        return verts, joints[:, CANONICAL_MODEL_JOINTS_REFINED[:24], :]


def scale_mesh_from_normal(vertices: torch.Tensor, faces: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Args:
        - vertices TORCHFLOAT32 [BATCH, N_v, 3]
        - faces TORCHFLOAT32 [BATCH, N_v, 3]

    Returns:
        - vertices_scaled TORCHFLOAT32 [BATCH, N_v, 3]
    """
    BATCH, N_vertices, _ = vertices.shape

    # Get the meshes normals
    meshes = Meshes(verts=vertices, faces=faces)
    vertices_packed = meshes.verts_packed()
    normals_packed = meshes.verts_normals_packed()

    # Scale through the computed normals to get updated vertices positions
    output_vertices_packed = vertices_packed + scale * normals_packed
    output_vertices = output_vertices_packed.reshape((BATCH, N_vertices, 3))

    return output_vertices


def get_smal_model(device: str, mode_original: bool = False, scale_factor: float = 1.0) -> SMAL:

    smal_path = os.path.join(Keys().source_smal_folder, "my_smpl_39dogsnorm_newv3_dog.pkl")
    sym_ind_path = os.path.join(Keys().source_smal_folder, "symmetry_inds.json")

    assert os.path.isfile(smal_path), f"SMAL archive file missing: {smal_path}"
    assert os.path.isfile(sym_ind_path), f"SMAL sym. indices file missing: {sym_ind_path}"

    buffer = process_original_smal(
        path_original=smal_path,
        symmetry_json_path=sym_ind_path,
    )

    smal = SMAL(buffer=buffer, mode_original=mode_original, scale_factor=scale_factor).to(device)

    return smal
