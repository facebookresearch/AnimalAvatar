# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Literal
import torch
import numpy as np
import pickle as pk
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from densepose.config import add_densepose_config
from config.keys import Keys
from data.input_cop import InputCop
from cse_embedding.visualize_cse import visualize_cse_maps
import rendering.visualization as viz


class DensePosePredictor:

    # Load densepose model and config file
    cse_config_filepath = os.path.join(Keys().source_cse_folder, "densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml")
    model_filepath = os.path.join(Keys().source_cse_folder, "model_final_8c9d99.pkl")

    assert os.path.isfile(cse_config_filepath), f"Config model is missing: {cse_config_filepath}"
    assert os.path.isfile(model_filepath), f"Model checkpoint is missing: {model_filepath}"

    def __init__(self):
        # Initialize model
        setup_logger()
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.cse_config_filepath)
        cfg.MODEL.WEIGHTS = self.model_filepath
        self.predictor = DefaultPredictor(cfg)

    def predict_single(self, image_rgb: np.ndarray) -> dict[str, torch.Tensor]:
        """
        Predict DensePose CSE on a single image.
        Args:
            -image_rgb (H, W, 3) NPUINT8
        """
        outputs = self.predictor(image_rgb)

        if outputs["instances"].pred_classes.shape[0] == 0:
            output_dict = {
                "E": None,
                "S": None,
                "bboxes_x1y1x2y2": None,
                "scores": None,
                "pred_classes": None,
                "image_size": (image_rgb.shape[0], image_rgb.shape[1]),
            }
        else:
            output_dict = {
                # ->(1,16,112,112) TORCHFLOAT32
                "E": outputs["instances"].pred_densepose.embedding.cpu(),
                # ->(1,2,112,112) TORCHFLOAT32
                "S": outputs["instances"].pred_densepose.coarse_segm.cpu(),
                # ->(1,4) TORCHFLOAT32
                "bboxes_x1y1x2y2": outputs["instances"].pred_boxes.tensor.cpu(),
                # ->(1,) TORCHFLOAT32
                "scores": outputs["instances"].scores.cpu(),
                # ->(1,) TORCHINT64
                "pred_classes": outputs["instances"].pred_classes.cpu(),
                "image_size": (image_rgb.shape[0], image_rgb.shape[1]),
            }
        return output_dict

    def predict_batch(self, images_rgb: np.array) -> dict[str, torch.Tensor]:
        """
        Predict DensePose CSE on a batch of images.
        Args:
            -images_rgb (BATCH, H, W, 3) NPUINT8
        """
        results = [self.predict_single(images_rgb[i]) for i in range(images_rgb.shape[0])]
        return {key: [results[i][key] for i in range(images_rgb.shape[0])] for key in results[0]}


def is_already_computed_cse(sequence_index: str, cache_path: str) -> bool:
    computed_file_path = os.path.join(cache_path, f"{sequence_index}_cse_predictions.pk")
    return os.path.isfile(computed_file_path)


def preprocess_cse(sequence_index: str, dataset_source: str, cache_path: str):
    """
    Launch and save CSE preprocessing of a sequence.
    """
    ic = InputCop(sequence_index=sequence_index, dataset_source=dataset_source)
    images_rgb = (255 * ic.images_hr.numpy()).astype(np.uint8)

    cse_predictions_filepath = os.path.join(cache_path, f"{sequence_index}_cse_predictions.pk")
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

    # Generate CSE embeddings
    cse_processor = DensePosePredictor()
    output = cse_processor.predict_batch(images_rgb)

    # Save
    with open(cse_predictions_filepath, "wb") as f:
        pk.dump(output, f)


def visualize_cse(sequence_index: str, dataset_source: str, cache_path: str, cse_version: Literal["original", "of_refined"]):
    """
    Visualize preprocessing of a sequence (must be already computed)
    """
    assert is_already_computed_cse(sequence_index, cache_path)

    ic = InputCop(sequence_index=sequence_index, dataset_source=dataset_source, cse_mesh_name="cse", cse_version=cse_version)

    cse_maps, cse_masks, valid_indices = ic.cse_maps

    rendered_cse_maps = visualize_cse_maps(
        cse_maps=cse_maps,
        cse_masks=cse_masks,
        valid_indices=valid_indices,
        cse_embedding_mesh=ic.cse_embedding,
        images=ic.images,
    )

    if cse_version == "original":
        save_video_path = f"{sequence_index}_cse_visualization.mp4"
    elif cse_version == "of_refined":
        save_video_path = f"{sequence_index}_cse_of_refined_visualization.mp4"
    else:
        raise Exception(f"Unknown cse_version: {cse_version}")

    viz.make_video(rendered_cse_maps, output_path=os.path.join(cache_path, save_video_path))
