# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.structures import Meshes


def arap_fast(meshes_t1: Meshes, meshes_t2: Meshes) -> torch.Tensor:
    """
    Enforce that edge distance are preserved in Meshes between timestep t1 and timestep t2.

    Args:
        - meshes_t1, meshes_t2: Meshes [BATCH]

    Returns:
        - arap_fast_loss (BATCH,1) TORCHFLOAT32
    """

    # Get number of vertices in the mesh
    N_meshes = len(meshes_t1)
    assert N_meshes == len(meshes_t2)

    # (SUM_E, 2)
    edges_packed_t1 = meshes_t1.edges_packed()
    # (SUM_V, 3)
    vertices_packed_t1 = meshes_t1.verts_packed()
    # (SUM_E, 2)
    edges_packed_t2 = meshes_t2.edges_packed()
    # (SUM_V, 3)
    vertices_packed_t2 = meshes_t2.verts_packed()

    # (SUM_E,)
    edge_to_mesh_idx = meshes_t1.edges_packed_to_mesh_idx()

    # (Edges in mesh_t1 (V0->V1))
    v0_t1, v1_t1 = vertices_packed_t1[edges_packed_t1].unbind(1)
    edges_coords_t1 = torch.norm(v1_t1 - v0_t1, p=2, dim=1, keepdim=True)

    # (Edges in mesh_t2 (V0->V1))
    v0_t2, v1_t2 = vertices_packed_t2[edges_packed_t2].unbind(1)
    edges_coords_t2 = torch.norm(v1_t2 - v0_t2, p=2, dim=1, keepdim=True)

    loss = torch.abs(edges_coords_t2 - edges_coords_t1)

    return torch.cat([torch.sum(loss[edge_to_mesh_idx == i]).reshape((1, 1)) for i in range(N_meshes)], dim=0)
