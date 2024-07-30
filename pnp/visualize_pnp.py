# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pickle as pk
import rendering.visualization as viz
import rendering.visualization_renderer as vizrend
from pnp.preprocess_pnp import is_already_computed_pnp
from model.pose_models import PoseCanonical
from data.input_cop import InputCop
from pnp.camera_estimation import align_cameras_from_PNP_solution


def visualize_pnp(sequence_index: str, dataset_source: str, cache_path: str, device: str = "cuda"):
    """
    Visualize cameras solution from PnP (with a canonical mesh).
    """

    ic = InputCop(sequence_index=sequence_index, dataset_source=dataset_source)
    images = ic.images
    renderer = ic.renderer
    smal = ic.smal
    N_frames = ic.N_frames_synth
    X_ind = ic.X_ind
    X_ts = ic.X_ts
    cameras_original = ic.cameras_original.to(device)

    pose_model = PoseCanonical(N=N_frames).to(device)

    assert is_already_computed_pnp(sequence_index, cache_path)

    # Load PnP solution
    with open(os.path.join(cache_path, f"{sequence_index}_pnp_R_T.pk"), "rb") as f:
        pnp_solution = pk.load(f)
    valid_indices_pnp, R_pnp, T_pnp = pnp_solution["valid_indices_pnp"], pnp_solution["R_PNP"], pnp_solution["T_PNP"]
    R_pnp = R_pnp.to(device)
    T_pnp = T_pnp.to(device)

    # Align the PNP solution with the GT-cameras
    cameras_movcam = align_cameras_from_PNP_solution(
        cameras_original=cameras_original,
        R_pnp=R_pnp,
        T_pnp=T_pnp,
        align_mode="extrinsics",
    )

    # Visualization (MOVING CAMERA)
    output = vizrend.basic_visualization(pose_model, X_ind, X_ts, smal, renderer, cameras_movcam)
    viz.make_video_list([(255 * images[X_ind].numpy()).astype(np.uint8), output], os.path.join(cache_path, f"{sequence_index}_visualize_pnp_R_T.mp4"))
