# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Centralize preprocessing-steps to perform before being able to use a sequence from COP3D dataset
# Covers:
#   * 1) CSE embeddings
#   * 2) PNP-RANSAC cameras initialization
#   * 3) Refined CSE embeddings via optical-flow

import argparse
from config.keys import Keys
from cse_embedding.preprocess_cse import preprocess_cse, visualize_cse, is_already_computed_cse
from pnp.preprocess_pnp import preprocess_pnp, is_already_computed_pnp
from pnp.visualize_pnp import visualize_pnp
from optical_flow.cse_optical_flow_preprocess import preprocess_refined_cse_from_optical_flow, is_already_computed_refined_cse


def preprocess_sequence(sequence_index: str, dataset_source: str = "COP3D", recompute: bool = False, visualize: bool = False, device: str = "cuda"):

    print(f"Results and visualizations will be saved in : {Keys().preprocess_path}")

    # CSE ---------------------------------------------------------------------------------------------
    preprocess_path_cse = Keys().preprocess_path_cse

    if recompute or (not is_already_computed_cse(sequence_index, preprocess_path_cse)):
        preprocess_cse(sequence_index, dataset_source, preprocess_path_cse)

    if is_already_computed_cse(sequence_index, preprocess_path_cse) and visualize:
        visualize_cse(sequence_index, dataset_source, preprocess_path_cse, cse_version="original")

    # PNP-RANSAC ---------------------------------------------------------------------------------------
    preprocess_path_pnp = Keys().preprocess_path_pnp

    if (recompute) or (not is_already_computed_pnp(sequence_index, preprocess_path_pnp)):
        preprocess_pnp(sequence_index, dataset_source, preprocess_path_pnp, device)

    if is_already_computed_pnp(sequence_index, preprocess_path_pnp) and visualize:
        visualize_pnp(sequence_index, dataset_source, preprocess_path_pnp, device)

    # Not available yet (mesh geodesic distance missing).
    # # CSE-OF-Refined ----------------------------------------------------------------------------------
    # preprocess_path_ofcse = Keys().preprocess_path_cse

    # if (recompute) or (not is_already_computed_refined_cse(sequence_index, preprocess_path_ofcse)):
    #     preprocess_refined_cse_from_optical_flow(sequence_index, dataset_source)

    # if is_already_computed_refined_cse(sequence_index, preprocess_path_ofcse) and visualize:
    #     visualize_cse(sequence_index, dataset_source, preprocess_path_ofcse, cse_version="of_refined")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a sequence based on the provided index.")
    parser.add_argument("-sequence_index", type=str, help="Index of the sequence to preprocess")
    parser.add_argument("--visualize", action="store_true", help="Save a visualization of each preprocess")
    parser.add_argument("--recompute", action="store_true", help="Force preprocess recomputation")
    parser.add_argument("--custom", action="store_true", help="If the sequence is not from COP3D dataset")
    args = parser.parse_args()

    dataset_source = "CUSTOM" if args.custom else "COP3D"

    print(f"Sequence Index: {args.sequence_index}, Dataset: {dataset_source}")

    preprocess_sequence(
        sequence_index=args.sequence_index,
        dataset_source=dataset_source,
        recompute=args.recompute,
        visualize=args.visualize,
    )
