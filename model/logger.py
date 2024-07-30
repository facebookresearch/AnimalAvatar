# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from torch.utils.tensorboard import SummaryWriter
import pickle as pk
from pathlib import Path
from shutil import copyfile
from omegaconf import OmegaConf
import shutil
import glob
from copy import copy


class Logger:
    """
    The logger takes care of all saving actions required during optimization, which includes
        - logs
        - model weights
        - source code
    """

    program_list = [
        "model/pose_models",
        "model/positional_embedding.py",
        "main_optimize_scene.py",
        "model/texture_models/model_cse.py",
        "model/texture_models/model_duplex.py",
        "model/texture_models/model_utils.py",
        "model/texture_models/trajectory_sampler.py",
    ]

    def __init__(self, args, source_code_folder: str, log_folder: str = "logs", tag: str = "", initialize_directory: bool = True):

        if initialize_directory:

            if type(args) == dict:
                self.args = args
            else:
                self.args = vars(args)

            self.log_folder = log_folder
            self.source_code_folder = source_code_folder
            self.tag = tag

            # Create the folder for the current experiment
            self.experiment_folder, self.checkpoint_folder, self.program_folder = self.create_experiment_folder()

            # Save the source code in the log folder
            self.save_program()

            self.logs = {"epoch": [], "time": []}  # Log dictionnary that will store all metrics/losses
            self.logs_ext = {}
            self.log_path = os.path.join(self.experiment_folder, "logs.pk")
            self.log_path_ext = os.path.join(self.experiment_folder, "logs_ext.pk")
            self.args_path = os.path.join(self.checkpoint_folder, "args.pk")
            self.args_path_txt = os.path.join(self.checkpoint_folder, "args.txt")

            # Save the arguments used to generate the model in the checkpoint folder
            self.dump_args()

        self.writer = None
        self.current_best_value = None
        self.current_best_epoch = None

    def create_experiment_folder(self):
        """
        Create a folder to store logs and model checkpoints
        """
        # Create the log folder it it does not exist yet
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        experiment_folder = os.path.join(self.log_folder, self.tag)

        if os.path.isdir(experiment_folder):
            shutil.rmtree(experiment_folder)

        checkpoint_folder = os.path.join(experiment_folder, "checkpoints")
        program_folder = os.path.join(experiment_folder, "program")

        os.makedirs(experiment_folder)
        os.makedirs(checkpoint_folder)
        os.makedirs(program_folder)

        return experiment_folder, checkpoint_folder, program_folder

    def add_ext_logs(self, new_logs: dict, dump_logs: bool = True):
        """
        Add new logs in the logger dictionnary that are not linked to the epoch and time keys
        """
        for k in new_logs:
            if k not in self.logs_ext:
                self.logs_ext[k] = []
            self.logs_ext[k].append(new_logs[k])
        if dump_logs:
            self.dump_ext_logs()

    def add_logs(self, new_logs: dict, epoch: int, dump_logs: bool = True):
        """
        Add new logs in the logger dictionnary
        """
        for k in new_logs:
            if k not in self.logs:
                self.logs[k] = []
            self.logs[k].append((epoch, new_logs[k]))

        self.logs["time"].append((epoch, time.strftime("%Y_%d_%m_%H_%M_%S")))

        if dump_logs:
            self.dump_logs()

    def add_tensorboard_logs(self, new_logs: dict, epoch: int):
        """
        Args:
            - new_logs [str, (N,) TORCHFLOAT32]
        Add the logs 'new_logs' to tensorboard
        """
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.experiment_folder)
        for k in new_logs:
            if "image" in k:
                continue
            self.writer.add_scalar(k, new_logs[k].mean().item(), epoch)

    def dump_logs(self):
        """
        Save the log dictionnary in a pickle file inside the log folder
        """
        with open(self.log_path, "wb") as f:
            pk.dump(self.logs, f)

    def dump_ext_logs(self):
        """
        Save the log dictionnary in a pickle file inside the log folder
        """
        with open(self.log_path_ext, "wb") as f:
            pk.dump(self.logs_ext, f)

    def dump_args(self):
        """
        Save the arguments to later restore a model
        """
        if self.args == {}:
            return
        with open(self.args_path, "wb") as f:
            pk.dump(self.args, f)
        with open(self.args_path_txt, "w") as f:
            f.write(OmegaConf.to_yaml(self.args["_content"]))

    def save_program(self):
        """
        Save the key python files in the program
        """
        if self.source_code_folder is not None:
            for program in self.program_list:

                if os.path.isdir(os.path.join(self.source_code_folder, program)):

                    for program_files in glob.glob(os.path.join(self.source_code_folder, program, "**", "*.py"), recursive=True):

                        destination_path = os.path.join(self.program_folder, Path(program_files).relative_to(self.source_code_folder).as_posix())

                        # Create the folder before
                        os.makedirs(Path(destination_path).parent.as_posix(), exist_ok=True)
                        # Copy the files
                        copyfile(program_files, destination_path)
                else:
                    destination_path = os.path.join(self.program_folder, program)

                    # Create the folder before
                    os.makedirs(Path(destination_path).parent.as_posix(), exist_ok=True)

                    copyfile(os.path.join(self.source_code_folder, program), destination_path)


def restore_logger(experiment_folder: str) -> Logger:

    # Make sure the source_path is a correct directory
    assert1 = os.path.isdir(os.path.join(experiment_folder, "checkpoints"))
    assert2 = os.path.isdir(os.path.join(experiment_folder, "program"))
    assert3 = os.path.isfile(os.path.join(experiment_folder, "checkpoints", "args.pk"))
    if (not assert1) or (not assert2) or (not assert3):
        return None

    logger = Logger(None, None, initialize_directory=False)

    # Initialize log directories
    logger.log_folder = Path(experiment_folder).parent.as_posix()
    logger.source_code_folder = None
    logger.tag = None

    logger.experiment_folder = experiment_folder
    logger.checkpoint_folder = os.path.join(logger.experiment_folder, "checkpoints")
    logger.program_folder = os.path.join(logger.experiment_folder, "program")
    logger.writer = SummaryWriter(log_dir=logger.experiment_folder)
    logger.log_path = os.path.join(logger.experiment_folder, "logs.pk")
    logger.log_path_ext = os.path.join(logger.experiment_folder, "logs_ext.pk")
    logger.args_path = os.path.join(logger.checkpoint_folder, "args.pk")
    logger.args_path_txt = os.path.join(logger.checkpoint_folder, "args.txt")

    # Restore logs
    with open(logger.log_path, "rb") as f:
        logger.logs = pk.load(f)

    # Restore logs_ext
    with open(logger.log_path_ext, "rb") as f:
        logger.logs_ext = pk.load(f)

    return logger


def get_similar_logger(log_folder: str, tag: str) -> list[str]:
    return list(glob.glob(os.path.join(log_folder, "{}Train-*".format(tag))))


def get_logger(cfg, tag, source_code_folder, log_folder, add_time_to_tag=True) -> Logger:

    # Will not do anything in config DEBUG
    if add_time_to_tag:
        tag = "{}Train-{}".format(tag, time.strftime("%Y_%d_%m_%H_%M_%S"))

    # Define the logger
    return Logger(args=cfg, source_code_folder=source_code_folder, log_folder=log_folder, tag=tag)
