# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from pytorch3d.common.workaround import symeig3x3


def assert_finite(x: torch.Tensor):
    assert np.isfinite(x.sum().data.cpu().numpy()), "bad assert"


def dets3x3(a: torch.Tensor) -> torch.Tensor:
    return (
        a[:, 0, 0] * (a[:, 1, 1] * a[:, 2, 2] - a[:, 2, 1] * a[:, 1, 2])
        - a[:, 1, 0] * (a[:, 0, 1] * a[:, 2, 2] - a[:, 2, 1] * a[:, 0, 2])
        + a[:, 2, 0] * (a[:, 0, 1] * a[:, 1, 2] - a[:, 1, 1] * a[:, 0, 2])
    )


def sefl_mult(A: torch.Tensor, swap: bool = False) -> torch.Tensor:
    if swap:
        A = torch.bmm(A, A.transpose(1, 2))
    else:
        A = torch.bmm(A.transpose(1, 2), A)
    A = 0.5 * (A + A.transpose(1, 2))
    return A


def batch_svd3x3(A: torch.Tensor, stab_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_shape = A.shape
    A = A.reshape(-1, 3, 3) * stab_mult
    A_V = sefl_mult(A)

    dV, V = symeig3x3(A_V)
    if (~torch.isfinite(V)).any():
        V = torch.eye(3).to(V).repeat(A.shape[0], 1, 1)
        print("Warning: SVD result not finite!")

    assert_finite(dV), "bad eig"

    S = torch.clamp(dV, 1e-12 * stab_mult**2).sqrt()
    U = torch.bmm(A, V) / S[:, None]

    return U.reshape(*a_shape), S.reshape(*a_shape[:-1]) / stab_mult, V.reshape(*a_shape)


class ArapLoss(torch.nn.Module):
    def __init__(self, template: torch.Tensor, faces: torch.Tensor):
        """
        ARAP (As-Rigid-As-Possible Surface Modeling) implementation, see: https://igl.ethz.ch/projects/ARAP/arap_web.pdf.

        Args:
                - template: (3, N_VERTS) TORCHFLOAT32
                - faces: (3, N_FACES) TORCHINT64
        """
        super().__init__()

        edges, cotans = self._get_angle_cotans(template, faces)

        adj_list_indices, adj_list_weights = self._get_weighted_adj_list(edges, cotans, template.shape[-1])
        self.register_buffer("adj_list_indices", adj_list_indices)  # ->(N_VERTS, MAX_DEG)
        self.register_buffer("adj_list_weights", adj_list_weights)  # ->(N_VERTS, MAX_DEG)

        template_edge_vectors = self._get_edge_vectors(template)
        template_ev_weighted = template_edge_vectors * self.adj_list_weights[..., None]
        self.register_buffer("template_edge_vectors", template_edge_vectors.transpose(-1, -2))  # ->(N_VERTS, 3, MAX_DEG,)
        self.register_buffer("template_ev_weighted", template_ev_weighted)  # ->(N_VERTS, MAX_DEG, 3)

    def _get_angle_cotans(
        self,
        template: torch.Tensor,
        faces: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cot(angle(u, v)) = dot(u, v) / |u x v|
        Args:
                - templates: (D=3, N_VERTS) TORCHFLOAT32
                - faces: (3, N_FACES) TORCHINT64
        Returns:
                - edge_indices: (2, 3, N_FACES) TORCHINT64
                - cotans: (3, N_FACES) TORCHFLOAT32
        """
        # ->(D, 3, N_FACES)
        faces_coords = template[:, faces]
        cotans = []
        edges = []
        for i in range(3):
            u = faces_coords[:, (i + 1) % 3] - faces_coords[:, i]
            v = faces_coords[:, (i - 1) % 3] - faces_coords[:, i]
            edges.append(faces[[(i + 1) % 3, (i - 1) % 3]])
            cotans.append((u * v).sum(dim=0).abs() / torch.cross(u, v, dim=0).norm(dim=0).clamp(eps))
        return torch.stack(edges, dim=1), torch.stack(cotans, dim=0)

    def _get_weighted_adj_list(self, edges: torch.Tensor, cotans: torch.Tensor, n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        build the list of incident edges and their weights.
        Args:
                - edges: (2, 3, N_FACES) TORCHINT64
                - cotans: (3, N_FACES) TORCHFLOAT32
                - n_points: INT (<-> N_VERTS)
        Returns:
                - adj_list_indices: (N_POINTS, MAX_DEGREE) TORCHINT64
                - adj_list_weights: (N_POINTS, MAX_DEGREE) TORCHDLOAT32
        """

        edge_weights = torch.sparse.FloatTensor(edges.reshape(2, -1), cotans.reshape(-1), torch.Size([n_points] * 2))
        edge_weights = (edge_weights + edge_weights.transpose(0, 1)).coalesce()
        edges = edge_weights._indices()

        max_degree = (edges[0] == edges[0].mode().values).sum()
        adj_list_indices = torch.zeros(n_points, max_degree, dtype=torch.int64)
        adj_list_weights = torch.zeros(n_points, max_degree, dtype=torch.float32)

        for i, ew_row in enumerate(edge_weights):
            nnz = ew_row._indices().shape[1]
            assert nnz <= max_degree
            adj_list_indices[i, :nnz] = ew_row._indices()[0]
            adj_list_weights[i, :nnz] = ew_row._values()

        return adj_list_indices, adj_list_weights

    def _get_edge_vectors(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
                - points (..., 3, N_VERTS) TORCHFLOAT32
        Returns:
                - edge_vectors (..., N_VERTS, MAX_DEG, 3) TORCHFLOAT32
        """
        start_pt = points.transpose(-1, -2)
        end_pt = start_pt[..., self.adj_list_indices, :]
        return end_pt - start_pt[..., None, :]

    def forward(self, prediction: torch.Tensor, rotations: torch.Tensor | None = None, nrg_clamp: float = 1.0) -> torch.Tensor:
        """
        Args:
                - prediction: (B, 3, N_VERTS) TORCHFLOAT32
        Returns
                - loss: (B,) TORCHFLOAT32
        """
        b, _, n_points = prediction.shape

        pred_edge_vectors = self._get_edge_vectors(prediction).transpose(-1, -2)

        if rotations is None:
            svd_input = torch.matmul(pred_edge_vectors, self.template_ev_weighted)
            u, s, v = batch_svd3x3(svd_input, stab_mult=1e3)
            eye = u.new_zeros(b, n_points, 3, 3)
            eye[..., 0, 0] = 1.0
            eye[..., 1, 1] = 1.0
            R_test = torch.matmul(u, v.transpose(-1, -2))
            eye[..., -1, -1] = dets3x3(R_test.reshape(-1, 3, 3)).reshape(b, n_points)
            # find the rotation matrix by composing U and V again
            rotations = torch.matmul(torch.matmul(u, eye), v.transpose(-1, -2)).detach()

        nrg = (self.adj_list_weights * (pred_edge_vectors - torch.matmul(rotations, self.template_edge_vectors)).norm(dim=-2)).sum(-1)
        assert (~torch.isnan(nrg)).all()
        return nrg.clamp(max=nrg_clamp).mean(dim=1)
