# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import math
import pickle as pk


def get_lbo_from_name(mesh_name: str, query_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        - mesh_name (smal/cse)
    Returns:
        - lbo_basis (N_verts,256) NPFLOAT32
        - lbo_areas (N_verts,N_verts) NPFLOAT32
    """

    lbo_filepath = os.path.join(query_folder, f"lbo_{mesh_name}.pk")

    assert os.path.isfile(lbo_filepath)
    print(f"loading: {lbo_filepath}")

    with open(lbo_filepath, "rb") as f:
        X = pk.load(f)

    lbo_areas = X["lbo_areas"]
    lbo_basis = X["lbo_basis"]

    return lbo_basis, lbo_areas


def get_mapping_from_name(mesh_source_name: str, mesh_destination_name: str, query_folder: str) -> np.ndarray:
    """
    Args:
        - mesh_source_name STR
        - mesh_destination_name STR
    Returns:
        - C (128,128) NPFLOAT32 functional map source -> dest
    """
    mapping_filepath = os.path.join(query_folder, f"lbo_{mesh_source_name}_to_{mesh_destination_name}.pk")

    assert os.path.isfile(mapping_filepath)
    print(f"loading: {mapping_filepath}")

    with open(mapping_filepath, "rb") as f:
        X = pk.load(f)

    return X["C"]


def map_X_from_source_to_destination(X_A: np.ndarray, lbo_areas_A: np.ndarray, lbo_basis_A: np.ndarray, lbo_areas_B: np.ndarray, lbo_basis_B: np.ndarray, C_A_to_B: np.ndarray) -> np.ndarray:
    """
    Map a function X_A on mesh A to mesh B

    Args:
        - X_A (N_verts_A, D) NPFLOAT32
        - lbo_basis_A (N_verts_A, 128) NPFLOAT32
        - areas_A (N_verts_A,) NPFLOAT32
        - lbo_basis_B (N_verts_B, 128) NPFLOAT32
        - areas_B (N_verts_B,) NPFLOAT32
        _ C_A_to_B (128,128) NPFLOAT32
    """
    M = math.sqrt(lbo_areas_B.sum() / lbo_areas_A.sum())
    X_B = M * lbo_basis_B @ C_A_to_B @ lbo_basis_A.T @ lbo_areas_A @ X_A
    X_B = X_B.astype(np.float32)
    return X_B


def get_geodesic_distance_from_name(mesh_name: str, query_folder: str) -> np.ndarray:
    geodesic_dist_path = os.path.join(query_folder, f"{mesh_name}_geodesic_dist.pk")
    assert os.path.isfile(geodesic_dist_path)
    print(f"loading: {geodesic_dist_path}")
    with open(geodesic_dist_path, "rb") as f:
        return pk.load(f)


def squared_euclidean_distance_matrix(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """ ""
    Computes pairwise squared Euclidean distances between points

    Args:
        pts1: Tensor [M x D], M is the number of points, D is feature dimensionality
        pts2: Tensor [N x D], N is the number of points, D is feature dimensionality

    Return:
        Tensor [M, N]: matrix of squared Euclidean distances; at index (m, n)
            it contains || pts1[m] - pts2[n] ||^2
    """
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(1, keepdim=True).t()
    return edm.contiguous()


def get_closest_vertices_matrix(X: torch.Tensor, Y: torch.Tensor, size_chunk: int = 10_000) -> torch.Tensor:
    """
    Get for each point x in X the closest point y in Y.
    Args:
        - X (M, D) TORCHFLOAT32
        - Y (N, D) TORCHFLOAT32
    Return:
        - output (M,) TORCHINT64
    """
    closest_vertices_X_to_Y = torch.zeros((X.shape[0],), dtype=torch.int64, device=X.device)

    size_X = X.shape[0]
    for chunk in range((size_X - 1) // size_chunk + 1):
        X_chunk = X[size_chunk * chunk : size_chunk * (chunk + 1)]
        distance_X_chunk_to_y = squared_euclidean_distance_matrix(X_chunk, Y)
        closest_vertices_X_to_Y[size_chunk * chunk : size_chunk * (chunk + 1)] = torch.argmin(distance_X_chunk_to_y, dim=1).type(torch.int64)

    return closest_vertices_X_to_Y
