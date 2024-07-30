# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

"""
File path to update:
    - EXTERNAL_DATA_PATH: Where we store scene preprocessings
    - COP3D_DATA_ROOT_PATH: Path of the CoP3D dataset
"""
EXTERNAL_DATA_PATH = None
COP3D_DATA_ROOT_PATH = None


assert EXTERNAL_DATA_PATH is not None, "EXTERNAL_DATA_PATH missing in 'config/keys.py, please manually add it'"
assert COP3D_DATA_ROOT_PATH is not None, "COP3D_DATA_ROOT_PATH missing in 'config/keys.py', please manually add it"


class Keys:
    def __init__(self):

        # Code path
        self.code_path = Path(os.path.abspath(__file__)).parent.parent.as_posix()
        self.default_config_path = os.path.join(self.code_path, "config", "config.yaml")

        # Preprocessings
        self.preprocess_path = os.path.join(EXTERNAL_DATA_PATH, "preprocessing")
        if not os.path.isdir(self.preprocess_path):
            os.makedirs(self.preprocess_path)
        self.preprocess_path_cse = os.path.join(self.preprocess_path, "cse_embeddings")
        self.preprocess_path_pnp = os.path.join(self.preprocess_path, "pnp_ransac")

        # External data manually downloaded
        self.external_data_path = EXTERNAL_DATA_PATH
        self.source_cse_folder = os.path.join(self.external_data_path, "cse")
        self.source_lbo_folder = os.path.join(self.external_data_path, "lbos")
        self.source_smal_folder = os.path.join(self.external_data_path, "smal")
        self.source_texture_folder = os.path.join(self.external_data_path, "textures")
        self.source_refined_masks = os.path.join(self.external_data_path, "refined_masks")
        self.source_refined_keypoints = os.path.join(self.external_data_path, "sparse_keypoints")
        self.source_init_pose = os.path.join(self.external_data_path, "init_pose")

        # COP3D Dataset root
        self.dataset_root = COP3D_DATA_ROOT_PATH

        # Densepose CSE model version
        self.densepose_version = "V2_I2M"

        # List of CoP3D sequences for which we provide precomputed refined masks and sparse keypoints.
        self.subsetcop3d = [
            "1030_23106_17099",
            "1050_43243_38827",
            "1022_15133_10069",
            "1009_3157_2796",
            "591_89070_175572",
            "1009_3094_2348",
            "1043_36548_32210",
            "1050_43093_37690",
            "1009_3103_2528",
            "518_74410_144191",
            "600_92491_183070",
            "1006_69_348",
            "1006_75_704",
            "1010_4414_3789",
            "1033_26269_19535",
            "1050_43188_38370",
            "1047_40116_35067",
            "1048_41121_36801",
            "565_81664_160332",
            "591_89203_178102",
            "1014_8127_5793",
            "1013_7089_5359",
            "1019_13120_8271",
            "1020_14124_9201",
            "1022_15268_10864",
            "1023_16110_11841",
            "1024_17072_12455",
            "1028_21488_15551",
            "1018_12109_7686",
            "1037_30410_23055",
            "1043_36452_31495",
            "594_90172_179453",
            "1023_16220_12185",
            "1024_17145_12825",
            "1046_39156_34275",
            "1053_46158_41526",
            "1054_47075_43215",
            "1054_47111_43483",
            "493_70723_136017",
            "591_89108_176400",
        ]
