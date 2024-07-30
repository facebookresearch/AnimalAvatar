# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from data.input_cop import get_input_cop_from_archive
from model.inferencer import Inferencer
import rendering.visualization as viz
import rendering.visualization_renderer as vizrend
from scene_optim.callbacks import CallbackEval


def visualize_reconstruction(archive_path: str, device: str = "cuda"):

    # Load inputs
    ic = get_input_cop_from_archive(archive_path, device=device)

    image_size = ic.image_size
    images = ic.images.to(device)
    masks = ic.masks.to(device)
    X_ind = ic.X_ind
    X_ts = ic.X_ts
    X_ind_test = ic.X_ind_test
    X_ts_test = ic.X_ts_test
    smal = ic.smal
    renderer = ic.renderer
    texture = ic.texture.to(device)
    cameras = ic.cameras.to(device)
    cse_embedding = ic.cse_embedding.to(device)

    # Load model (pose, texture)
    inferencer = Inferencer(archive_path, use_archived_code=True)
    pose_model = inferencer.load_pose_model().to(device)
    texture_model = inferencer.load_texture_model().to(device)

    # -- STATS
    callback_eval = CallbackEval(images, masks, cameras, smal, cse_embedding, image_size, device)

    final_eval, final_eval_str = callback_eval.call(pose_model, X_ind_test, X_ts_test, texture_model)
    print("Results: {}".format(final_eval_str))

    # -- RENDERING
    viz.make_video_list(
        vizrend.global_visualization(images, masks, pose_model, X_ind, X_ts, smal, texture_model, cse_embedding, renderer, cameras, texture=texture),
        os.path.join(archive_path, "rendered_optimized.mp4"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize reconstruction results")
    parser.add_argument("archive_path", type=str, help="Archive of the reconstruction")
    args = parser.parse_args()

    print(f"Archive path: {args.archive_path}")

    visualize_reconstruction(args.archive_path)
