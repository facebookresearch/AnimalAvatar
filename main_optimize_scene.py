# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import glob
import torch
import hydra
import random
import numpy as np
from data.input_cop import get_input_cop_from_cfg
from model.logger import get_logger
from model.logger import get_similar_logger, restore_logger
from model.inferencer import Inferencer
import rendering.visualization as viz
import rendering.visualization_renderer as vizrend
from scene_optim.scene_optimizer import SceneOptimizer
from model.model_initialization import initialize_pose_model, initialize_texture_model
from scene_optim.callbacks import CallbackEval, CallbackEarlyStop


def seed_everything(seed: int = 1003):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.2")
def main_train(cfg):
    """
    Process a single COP3D video (with custom-TTA)
    """
    device = cfg.training.device
    image_size = cfg.exp.image_size

    # LOGGER ----------------------------------------------------------------------------------------------------
    logger = get_logger(cfg, tag=cfg.training.tag, source_code_folder=os.getcwd(), log_folder=cfg.exp.experiment_folder, add_time_to_tag=True)

    # Data Inputs -----------------------------------------------------------------------------------------------

    seed_everything(1003)

    ic = get_input_cop_from_cfg(cfg, device=device)

    # Indices and timesteps
    X_ind, X_ts = ic.X_ind, ic.X_ts
    X_ind_train, X_ts_train = ic.X_ind_train, ic.X_ts_train
    X_ind_test, X_ts_test = ic.X_ind_test, ic.X_ts_test

    # Images, Masks, Cameras, CSE_embedding
    images = ic.images.to(device)
    masks = ic.masks.to(device)
    cameras = ic.cameras.to(device)
    cse_embedding = ic.cse_embedding.to(device)

    # SMAL model
    smal = ic.smal

    # Renderer
    renderer = ic.renderer
    texture = ic.texture.to(device)

    # CSE Keypoints
    if cfg.exp.l_optim_cse_kp > 0:
        (cse_valid_indices, cse_keypoints_xy, cse_keypoints_vert_id, cse_keypoints_max_indice) = ic.cse_keypoints
        cse_keypoints_xy = cse_keypoints_xy.to(device)
        cse_keypoints_vert_id = cse_keypoints_vert_id.to(device)
    else:
        (cse_valid_indices, cse_keypoints_xy, cse_keypoints_vert_id, cse_keypoints_max_indice) = None, None, None, None

    # Sparse keypoints
    if cfg.exp.l_optim_sparse_kp > 0:
        sparse_keypoints, sparse_keypoints_scores = ic.sparse_keypoints
        sparse_keypoints = sparse_keypoints.to(device)
        sparse_keypoints_scores = sparse_keypoints_scores.to(device)
    else:
        sparse_keypoints, sparse_keypoints_scores = None, None

    # Model (Pose, Texture) initialization -----------------------------------------------------------------------------

    seed_everything(1003)

    # Test if some model checkpoints are available
    checkpoint_candidates_mlp = list(glob.glob(os.path.join(logger.checkpoint_folder, "*.pt")))
    pose_model, texture_model = None, None

    # Load trained models
    if len(checkpoint_candidates_mlp) > 0:
        inferencer = Inferencer(logger.experiment_folder, use_archived_code=True)

        pose_model = inferencer.load_pose_model()
        texture_model = inferencer.load_texture_model()

        if pose_model is not None:
            pose_model = pose_model.to(device)
        if texture_model is not None:
            texture_model = texture_model.to(device)

    if pose_model is None:
        if cfg.exp.init_pose:
            init_betas, init_betas_limbs = ic.init_betas
        else:
            init_betas, init_betas_limbs = None, None
        pose_model = initialize_pose_model(cfg, X_ind, X_ts, init_betas, init_betas_limbs, device)

    if texture_model is None:
        texture_model = initialize_texture_model(cfg, device)

    # Init Eval --------------------------------------------------------------------------------------------------------------

    # Callbacks
    callback_eval = CallbackEval(images, masks, cameras, smal, cse_embedding, image_size, device)
    callback_earlystop = CallbackEarlyStop()
    _, init_eval = callback_eval.call(pose_model, X_ind_test, X_ts_test, texture_model, return_individual=True)
    print("START: {}".format(init_eval))

    # RENDERING - BEFORE TRAINING
    viz.make_video_list(
        vizrend.global_visualization(images, masks, pose_model, X_ind, X_ts, smal, texture_model, cse_embedding, renderer, cameras, texture=texture),
        os.path.join(logger.experiment_folder, "rendered_init.mp4"),
    )

    # Optimization ---------------------------------------------------------------------------------------------------------------
    start = time.time()
    logger.add_ext_logs({"time": [start, start]})
    torch.cuda.reset_peak_memory_stats()

    scene_optimizer = SceneOptimizer(
        cameras,
        images,
        masks,
        X_ind,
        X_ts,
        cse_keypoints_xy,
        cse_keypoints_vert_id,
        cse_valid_indices,
        cse_keypoints_max_indice,
        sparse_keypoints,
        sparse_keypoints_scores,
        smal,
        logger,
        device,
        cfg.exp,
    )

    # The main optimization loop
    scene_optimizer.optimize_scene(
        N_epochs=cfg.exp.n_steps,
        pose_model=pose_model,
        X_ind_train=X_ind_train,
        X_ts_train=X_ts_train,
        texture_model=texture_model,
        callback_eval=callback_eval,
        callback_earlystop=callback_earlystop,
        checkpoint_freq=cfg.exp.checkpoint_freq,
        X_ind_test=X_ind_test,
        X_ts_test=X_ts_test,
    )

    stop = time.time()
    logger.logs_ext["time"][-1][1] = stop
    gpu_metric = torch.cuda.memory_stats(device=0)

    # Final Evaluation -----------------------------------------------------------------------------------------------------------------

    final_eval, final_eval_str = callback_eval.call(pose_model, X_ind_test, X_ts_test, texture_model)
    print("FINAL: {}, GPU: {}".format(final_eval_str, gpu_metric["reserved_bytes.all.peak"]))

    # Add metrics in logger
    logger.add_ext_logs(
        {
            "final_psnr": final_eval["l_psnr"],
            "final_psnrm": final_eval["l_psnrm"],
            "final_iou": final_eval["l_iou"],
            "final_lpips": final_eval["l_lpips"],
            "processing_time": sum([(X[-1] - X[0]) if len(X) > 1 else 0 for X in logger.logs_ext["time"]]),
            "GPU_memory": gpu_metric["reserved_bytes.all.peak"],
        },
        dump_logs=True,
    )

    # RENDERING - AFTER TRAINING
    viz.make_video_list(
        vizrend.global_visualization(images, masks, pose_model, X_ind, X_ts, smal, texture_model, cse_embedding, renderer, cameras, texture=texture),
        os.path.join(logger.experiment_folder, "rendered_optimized.mp4"),
    )


if __name__ == "__main__":
    main_train()
