# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Get CSE embeddings from other meshes via mesh alignment correspondence
import os
import torch
import pickle as pk
import numpy as np
import random
from tqdm import tqdm
import utils.img_utils as imutil
import cse_embedding.functional_map as fmap


class CseProcessing:

    animal_cat = "dog"
    allowed_densepose_version = ["V2_I2M"]
    cse_mesh_name_from_version = {"V2_I2M": {"dog": "cse"}}
    cse_category_from_animal_category = {"dog": 3}

    def __init__(self, cse_data_root_path: str, sequence_index: str, densepose_version: str, lbo_data_folder: str):
        """
        Args:
            - cse_data_root_path: root folder which contains the computation of all CSE sequences
            - sequence_index: COP3D TAG of the sequence (XXXX_XXX_XXXX)
            - densepose_version" algorithm version of CSE embedding
            - lbo_data_folder: folder which contains lbos computation of meshes
        """
        self.densepose_version = densepose_version
        self.cse_data_root_path = cse_data_root_path
        self.sequence_index = sequence_index
        self.data_path = os.path.join(self.cse_data_root_path, f"{self.sequence_index}_cse_predictions.pk")

        assert os.path.isfile(self.data_path), (f"Preprocessing file not found for sequence: {sequence_index}.", "Please launch preprocessing before scene optimization.")

        # Make sure the CSE sequence has been computed
        self.check_sequence_is_computed()

        # Get the animal category
        self.animal_cat_id = self.cse_category_from_animal_category[self.animal_cat]

        # Name of the cse mesh
        self.cse_mesh_name = self.cse_mesh_name_from_version[densepose_version][self.animal_cat]

        # Load functional mapping (LBO basis and functional mapping)
        self.lbo_data_folder = os.path.join(lbo_data_folder, "lbos")
        self.mesh_cse_data_folder = os.path.join(lbo_data_folder, "cse")

        # Load cse data
        with open(os.path.join(self.data_path), "rb") as f:
            self.data = pk.load(f)
        self.N = len(self.data["scores"])

    def check_sequence_is_computed(self):
        if not os.path.isfile(self.data_path):
            raise Exception("No CSE data found: {}".format(self.data_path))

    # Access CSE data methods ------------------------------------------------------------------------------------------------------------------------
    def get_frame_data_info(self, frame_index: int, normalize: bool = False) -> dict[str, np.ndarray | int | None]:
        """
        data contains:
            - E (NPARRAY [K, D, h_cse, w_cse]) (D-dimensional embedding vectors for every point of the default-sized box) (1,16,112,112) NPFLOAT32
            - S (NPARRAY [K, 2, h_cse, w_cse]): 2-dimensional segmentation mask for every point of the default-sized box (1,2,112,112) NPFLOAT32
            - bboxes_x1y1x2y2 Array([[x,y,~v,~u],...]) (K,4) NPFLOAT32
            - scores Array([s,...]) K
            - pred_classes List([c,...]) K
            - recommended_index INT32
        """
        keys_cse = ["S", "E", "bboxes_x1y1x2y2", "scores", "pred_classes", "image_size"]
        data = {k: self.data[k][frame_index] for k in keys_cse}

        # Verify densepose category prediction is the same as the category we're looking for (<=>dog)
        if (data["pred_classes"] is not None) and (len(data["pred_classes"]) > 0):
            scores_filtered = [s if (data["pred_classes"][i] == self.animal_cat_id) and ((data["scores"][i] > 0.5)) else -1 for i, s in enumerate(data["scores"])]
            if all([i == -1 for i in scores_filtered]):
                data["recommended_index"] = -1
            else:
                data["recommended_index"] = int(np.argmax(scores_filtered))
        else:
            data["recommended_index"] = -1

        if data["recommended_index"] != -1:
            data["S"] = data["S"][data["recommended_index"], None]
            data["E"] = data["E"][data["recommended_index"], None]
            data["bboxes_x1y1x2y2"] = [data["bboxes_x1y1x2y2"][data["recommended_index"]]]
            data["scores"] = data["scores"][data["recommended_index"]].reshape((-1,))
            data["pred_classes"] = [data["pred_classes"][data["recommended_index"]]]
            if normalize:
                data["E"] = imutil.normalize_numpy(data["E"], dim=1)
        else:
            data["E"] = None
            data["S"] = None
            data["bboxes_x1y1x2y2"] = None
            data["scores"] = None
            data["pred_classes"] = None
        return data

    def get_all_frame_data_info(self, normalize: bool = False) -> tuple[dict, list[int]]:
        """
        Get data for all frames in the sequence
        """
        output = [self.get_frame_data_info(i, normalize) for i in range(self.N)]
        valid_indices = [i for i in range(self.N) if (output[i] is not None) and (output[i]["E"] is not None)]
        return output, valid_indices

    def load_functional_mapping(self, name_source: str, name_convert: str, query_folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get LBOs
        source_lbo_basis, source_lbo_areas = fmap.get_lbo_from_name(name_source, query_folder)
        dest_lbo_basis, dest_lbo_areas = fmap.get_lbo_from_name(name_convert, query_folder)

        # Get functional mapping
        if name_source != name_convert:
            C_source_to_dest = fmap.get_mapping_from_name(name_source, name_convert, query_folder=query_folder)
        else:
            C_source_to_dest = None
        return source_lbo_basis, source_lbo_areas, dest_lbo_basis, dest_lbo_areas, C_source_to_dest

    def get_cse_embedding(self, convert_mesh_name: str, normalize: bool = False) -> torch.Tensor:
        """
        Get vertex-based CSE embedding of the mesh

        Args:
            - convert_mesh_name [cse, smal] name of the mesh cse embedding to return
        Return:
            - embedding (N_vertices, D_CSE) TORCHFLOAT32
        """
        cse_mesh_embedding_path = os.path.join(self.mesh_cse_data_folder, f"{self.cse_mesh_name}_embedding.pk")
        assert os.path.isfile(cse_mesh_embedding_path)

        with open(cse_mesh_embedding_path, "rb") as f:
            source_mesh_raw = pk.load(f)[f"{self.cse_mesh_name}_{self.animal_cat}"]

        if convert_mesh_name != self.cse_mesh_name:

            (cse_lbo_basis, cse_lbo_areas, smal_lbo_basis, smal_lbo_areas, C_cse_to_smal) = self.load_functional_mapping(self.cse_mesh_name, convert_mesh_name, self.lbo_data_folder)

            source_mesh_raw = fmap.map_X_from_source_to_destination(source_mesh_raw, cse_lbo_areas, cse_lbo_basis, smal_lbo_areas, smal_lbo_basis, C_cse_to_smal)

        if normalize:
            source_mesh_raw = imutil.normalize_numpy(source_mesh_raw, dim=1)

        return torch.Tensor(source_mesh_raw).type(torch.float32)

    # High Level methods ------------------------------------------------------------------------------------------------------------------------
    def get_cse_maps(self, normalize: bool, output_resolution: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Return the map of embeddings resized from cse size to image size (N, 16, H_img, W_img)

        Returns:
            - cse_maps TORCHFLOAT32 (N, D_CSE, H_output, W_output)
            - cse_masks TORCHFLOAT32 (N, 2, H_output, W_output)
            - valid_indices
        """
        cse_info, valid_indices = self.get_all_frame_data_info(normalize)
        original_img_size = cse_info[0]["image_size"]
        N = len(cse_info)

        cse_maps = torch.zeros((N, 16, original_img_size[0], original_img_size[1]), dtype=torch.float32)
        cse_masks = torch.zeros((N, 2, original_img_size[0], original_img_size[1]), dtype=torch.float32)
        cse_masks[:, 0] = 1.0

        for i in range(N):
            if i in valid_indices:
                x1, y1, x2, y2 = cse_info[i]["bboxes_x1y1x2y2"][0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1
                # Resize embedding and mask to image_size resolution
                cse_maps[i, :, y1:y2, x1:x2] = imutil.resize_torch(cse_info[i]["E"].permute((0, 2, 3, 1)), (h, w), mode="nearest-exact").permute(0, 3, 1, 2)
                cse_masks[i, :, y1:y2, x1:x2] = imutil.resize_torch(cse_info[i]["S"].permute((0, 2, 3, 1)), (h, w), mode="nearest-exact").permute(0, 3, 1, 2)

        # Resize to output resolution
        cse_maps = imutil.resize_torch(cse_maps.permute((0, 2, 3, 1)), (output_resolution[0], output_resolution[1]), mode="nearest-exact").permute(0, 3, 1, 2)
        cse_masks = imutil.resize_torch(cse_masks.permute((0, 2, 3, 1)), (output_resolution[0], output_resolution[1]), mode="nearest-exact").permute(0, 3, 1, 2)

        return cse_maps, cse_masks, valid_indices


def get_closest_vertices_map_from_cse_map(cse_maps: torch.Tensor, cse_masks: torch.Tensor, cse_embeddings_mesh: torch.Tensor) -> torch.Tensor:
    """
    Get maps where on each pixel (i,j) cse_closest_vertices[i,j] give the vertex index of the closest point on the mesh (in CSE space)

    Args:
        - cse_maps (BATCH, H, W, D_CSE) TORCHFLOAT32
        - cse_masks (BATCH, H, W) TORCHINT32 {0,1}
        - cse_embedding_mesh (N_verts, D_CSE) TORCHFLOAT32

    Returns:
        - cse_closest_vertices (BATCH, H, W) TORCHINT64
    """
    BATCH, H, W, D_CSE = cse_maps.shape
    device = cse_maps.device
    mask_flat = cse_masks.reshape(-1) > 0

    cse_closest_vertices = torch.zeros((BATCH, H, W, 1), dtype=torch.int64, device=device).reshape((-1, 1))

    cse_closest_vertices[mask_flat] = fmap.get_closest_vertices_matrix(cse_maps.reshape(-1, D_CSE)[mask_flat], cse_embeddings_mesh).reshape(-1, 1)

    cse_closest_vertices = cse_closest_vertices.reshape((BATCH, H, W))

    return cse_closest_vertices


def get_cse_keypoints_from_single_cse_map(
    cse_closest_vert: torch.Tensor,
    cse_mask: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Args:
        - cse_masks (H, W,) TORCHFLOAT32
        - cse_closest_verts (H, W) TORCHINT64
        - N: max number of keypoints to extract

    Returns:
        - keypoints_cse (N_KP_MAX, 2) in image-space (H,W)
        - closest_verts_cse (N_KP_MAX)
    """

    H, W = cse_closest_vert.shape
    device = cse_closest_vert.device

    keypoints_cse = torch.zeros((N, 2), dtype=torch.float32, device=device)
    closest_verts_cse = torch.zeros((N,), dtype=torch.int64, device=device)

    # Get cse_mask (i,j) indices (shuffled)
    indices_i, indices_j = cse_mask.reshape((H, W)).nonzero(as_tuple=True)
    x_indices = list(zip(indices_i, indices_j))
    random.shuffle(x_indices)

    k = 0
    if len(x_indices) > 0:
        indices_i, indices_j = zip(*x_indices)
        for i, j in zip(indices_i, indices_j):
            # Add the value only if the point is inside the additional mask
            keypoints_cse[k, 0] = j
            keypoints_cse[k, 1] = i
            closest_verts_cse[k] = cse_closest_vert[i, j]
            k += 1
            if N is not None and k == N:
                break

    return keypoints_cse, closest_verts_cse, k


def get_cse_keypoints_from_cse_maps(
    cse_closest_verts: torch.Tensor,
    cse_masks: torch.Tensor,
    valid_indices: list[int],
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        - cse_maps (BATCH, H, W, D_CSE) TORCHFLOAT32
        - cse_masks (BATCH, H, W, D_CSE) TORCHFLOAT32
        - cse_closest_verts (BATCH, H, W) TORCHINT64
        - additional_mask (BATCH, H, W, D_CSE) TORCHBOOL

    Returns:
        - keypoints_xy (BATCH, N_KP_MAX, 2)
        - keypoints_vert_id (BATCH, N_KP_MAX)
        - N_max_keypoints (BATCH,)
    """
    BATCH, H, W = cse_closest_verts.shape
    device = cse_closest_verts.device

    keypoints_xy = torch.zeros((BATCH, N, 2), dtype=torch.float32, device=device)
    keypoints_vert_id = torch.zeros((BATCH, N), dtype=torch.int64, device=device)
    N_max_keypoints = torch.zeros((BATCH,), dtype=torch.int32, device=device)

    for i in tqdm(range(BATCH), desc="Generating cse-keypoints"):
        if i in valid_indices:
            (keypoints_xy[i], keypoints_vert_id[i], N_max_keypoints[i]) = get_cse_keypoints_from_single_cse_map(cse_closest_verts[i], cse_masks[i], N)

    # Reduce to the maximum number of keypoints across the batch
    max_keypoints = int(torch.max(N_max_keypoints))
    keypoints_xy = keypoints_xy[:, :max_keypoints]
    keypoints_vert_id = keypoints_vert_id[:, :max_keypoints]

    return keypoints_xy, keypoints_vert_id, N_max_keypoints
