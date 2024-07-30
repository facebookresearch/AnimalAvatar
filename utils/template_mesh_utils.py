# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle as pk
import os
from config.keys import Keys


def get_mesh_2d_embedding_from_name(mesh_name: str) -> np.ndarray:
    """
    Args:
        - mesh_name: str
    Returns:
        - embedding_map (N_verts, 3) NPUINT8 [0,255]
    """
    texture_path = os.path.join(Keys().source_texture_folder, f"texture_{mesh_name}.pk")
    assert os.path.isfile(texture_path), f"Unknown file: {texture_path}"
    with open(texture_path, "rb") as f:
        return pk.load(f)
