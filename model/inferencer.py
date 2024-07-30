# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import glob
import pickle as pk
import importlib.util
from pathlib import Path
from omegaconf import OmegaConf
from config.keys import Keys
from model.pose_models import PoseBase
from model.texture_models import TextureBase


class Inferencer:
    """
    Load a pretrained model from a saved archive
    Args:
        - save_path: str
        - use_archived_code (for code files like model, choose source or archived codebase)
    """

    def __init__(self, save_path: str, use_archived_code: bool = True):

        self.save_path = Path(save_path)
        self.checkpoint_path = Path(save_path) / "checkpoints"
        self.args_path = Path(save_path) / "checkpoints" / "args.pk"
        self.args_txt = Path(save_path) / "checkpoints" / "args.txt"

        self.use_archived_code = use_archived_code

        if self.use_archived_code:
            self.code_dir = Path(save_path) / "program"
        else:
            self.code_dir = Keys().code_path

        self.cfg = OmegaConf.load(self.args_txt)

        with open(self.args_path, "rb") as f:
            self.args = pk.load(f)["_content"]

    def load_pose_model(self) -> PoseBase:
        """
        Load an optimized pose model from an archive path.
        """

        model_class_path = "{}.py".format(os.path.join(self.code_dir, "/".join(self.args["exp"]["mlp_model"]["_target_"].split(".")[:-1])))

        cp_list = sorted(list(glob.glob(os.path.join(self.checkpoint_path, "pose_model_*.pt"))))
        cp_list = [c for c in cp_list if "current_train.pt" in c] + [c for c in cp_list if "current_train.pt" not in c]
        if len(cp_list) == 0:
            return None
        cp = Path(cp_list[-1]).name

        path_cp = self.checkpoint_path / cp
        print("Loading model: {}".format(cp))

        model_args = {i: self.args["exp"]["mlp_model"][i] for i in self.args["exp"]["mlp_model"] if i != "_target_"}
        target_model = self.args["exp"]["mlp_model"]["_target_"].split(".")[-1]
        model_class_name = "module.{}".format(target_model)
        spec = importlib.util.spec_from_file_location("pose_model_module", model_class_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = eval(model_class_name)(**model_args)
        model.load_state_dict(torch.load(path_cp, map_location=torch.device("cpu")))
        return model

    def load_texture_model(self) -> TextureBase:
        """
        Load an optimized texture model from an archive path.
        """

        model_class_path = "{}.py".format(os.path.join(self.code_dir, "/".join(self.args["exp"]["texture_mlp_model"]["_target_"].split(".")[:-1])))

        cp_list = sorted(list(glob.glob(os.path.join(self.checkpoint_path, "texture_model_*.pt"))))
        cp_list = [c for c in cp_list if "current_train.pt" in c] + [c for c in cp_list if "current_train.pt" not in c]
        if len(cp_list) == 0:
            return None
        cp = Path(cp_list[-1]).name

        path_cp = self.checkpoint_path / cp
        print("Loading model: {}".format(cp))

        model_args = {i: self.args["exp"]["texture_mlp_model"][i] for i in self.args["exp"]["texture_mlp_model"] if i != "_target_"}
        target_model = self.args["exp"]["texture_mlp_model"]["_target_"].split(".")[-1]
        model_class_name = "module.{}".format(target_model)
        spec = importlib.util.spec_from_file_location("texture_model_module", model_class_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = eval(model_class_name)(**model_args)
        model.load_state_dict(torch.load(path_cp, map_location=torch.device("cpu")))
        return model
