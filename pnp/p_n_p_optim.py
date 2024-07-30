# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm
from pytorch3d.ops.perspective_n_points import efficient_pnp, EpnpSolution
from pytorch3d.renderer.cameras import PerspectiveCameras, get_screen_to_ndc_transform
from utils.torch_util import SuppressAllWarnings

# Optimization of the root pose and orientation via PNP method


def efficient_PNP_RANSAC(
    X_3d: torch.Tensor, y_2d_ndc: torch.Tensor, camera: PerspectiveCameras, n: int = 4, n_iter_max: int = 5, thre_inl: float = 0.2, min_inl: int = 4
) -> tuple[EpnpSolution | None, dict[str, float]]:
    """
    Args:
        - X_3d: (1, NPOINTS, 3) TORCHFLOAT32 [DEVICE]
        - y_2d_ndc: (NPOINTS, 2) TORCHFLOAT32 [DEVICE] 2D points in NDC space

        - n: minimum number of points to evaluate
        - n_iter_max: max number of allowed iterations (if -1, all combinations considered)
        - thre_inl: threshold to determine if datapoint fits to the model
        - min_inl: min number of points to consider that the model is ok
    """

    list_log_dict = []
    error_best, pnp_best = 1e10, None

    # Uncalibrate 2D points
    y_2d_uncalibrated = camera.unproject_points(torch.cat([y_2d_ndc, torch.ones((y_2d_ndc.shape[0], 1)).to(y_2d_ndc.device)], dim=1), world_coordinates=False)[None, :, :2]

    # If (n_iter_max == -1), trigger the evaluation of all combinations
    use_combinations = n_iter_max == -1
    if use_combinations:
        combs_ = torch.combinations(torch.arange(X_3d.shape[1]), r=n, with_replacement=False)
        n_iter_max = combs_.shape[0]

    for i in range(n_iter_max):

        c_dict = {
            "error1": False,
            "error2": False,
            "maybeinliers": None,
            "inliers": None,
            "pair_error_2d_pre": None,
            "pair_error_2d_post": None,
            "loss": None,
            "is_best": False,
            "2d_kp": None,
            "3d_kp": None,
            "3dp_kp": None,
        }

        # 1) Get a set of n candidate inliers
        if use_combinations:
            maybeinlier_index = combs_[i]
        else:
            maybeinlier_index = torch.randperm(X_3d.shape[1])[:n]

        # 2) PNP solution with these candidate inliers
        try:
            with SuppressAllWarnings():
                pnp_init = efficient_pnp(
                    x=X_3d[:, maybeinlier_index],
                    y=y_2d_uncalibrated[:, maybeinlier_index],
                    skip_quadratic_eq=False,
                )
        except:
            c_dict["error1"] = True
            list_log_dict.append(c_dict)
            continue

        # 3) Evaluate the candidate inlier solution

        # Use the predicted camera parameters to project X_3d to 2D coordinates and evalutate distance to y_2d
        eval_camera = PerspectiveCameras(camera.focal_length, camera.principal_point, pnp_init.R, pnp_init.T, in_ndc=True, device=camera.device)

        x_2d_ndc_eval = eval_camera.transform_points_ndc(X_3d)[..., :2]

        # Eval distance ->(1,NB_PTS)
        dist_2d = torch.sum((x_2d_ndc_eval - y_2d_ndc) ** 2, dim=2).sqrt()

        # Get the index of inlier points
        inlier_index = (dist_2d[0] < thre_inl).nonzero().squeeze(1)
        n_index = inlier_index.shape[0]

        c_dict["maybeinliers"] = maybeinlier_index.cpu().numpy()
        c_dict["inliers"] = inlier_index.cpu().numpy()
        c_dict["pair_error_2d_pre"] = dist_2d.cpu().numpy()

        # 4) If there are enough inlier points, consider it as a solution and reprocess the solution with all inliers
        if n_index > min_inl:
            try:
                with SuppressAllWarnings():
                    pnp_better = efficient_pnp(
                        x=X_3d[:, inlier_index],
                        y=y_2d_uncalibrated[:, inlier_index],
                    )
            except:
                c_dict["error2"] = True
                list_log_dict.append(c_dict)
                continue

            eval_camera = PerspectiveCameras(camera.focal_length, camera.principal_point, pnp_better.R, pnp_better.T, in_ndc=True, device=camera.device)

            x_2d_ndc_eval = eval_camera.transform_points_ndc(X_3d[:, inlier_index])[..., :2]
            pair_error_2d = torch.sum((x_2d_ndc_eval - y_2d_ndc[inlier_index]) ** 2, dim=2).sqrt()
            error_better = torch.mean(pair_error_2d) / (n_index + 1e-8)

            c_dict["pair_error_2d_post"] = pair_error_2d.cpu().numpy()
            c_dict["loss"] = error_better.cpu().numpy()

            if error_better < error_best:
                pnp_best = pnp_better
                error_best = error_better
                c_dict["is_best"] = True

        list_log_dict.append(c_dict)
    return pnp_best, list_log_dict


def get_init_from_pnp(
    keyp_3d: torch.Tensor,
    keyp2d_prediction: torch.Tensor,
    keyp2d_score: torch.Tensor,
    valid_indices: list[int],
    cameras: PerspectiveCameras,
    image_size: int,
    use_RANSAC: bool = False,
    filter_score: float = 0.2,
    n_iter_max: int = 1000,
    thre_inl: float = 0.05,
    min_inl_percentage: float = 0.6,
    device: str = "cuda",
):
    """
    Initialize the camera via PNP algorithm

    N -> number of frames
    N_pts -> max number of points sampled on each frame

    Args:
        - keyp_3d TORCHFLOAT32 (N, N_pts, 3) [DEVICE]
        - keyp2d_prediction TORCHFLOAT32 (N, N_pts, 2) [DEVICE] [0,img_size]x[0,img_size]
        - keyp2d_score TORCHFLOAT32 (N, N_pts, 1) [DEVICE]
        - valid_indices LIST

    Returns:
        - valid_ind List
        - R TORCHFLOAT32 (N, 3, 3) [DEVICE]
        - T TORCHFLOAT32 (N, 3) [DEVICE]
        - info_dict (err_2d, err_3d, logs)

    """
    info_dict_ransac = None
    output_list, info_list = [], []
    N_frames = keyp_3d.shape[0]

    for i in tqdm(range(cameras.R.shape[0]), desc="PnP", total=N_frames):

        # If the current frame has no valid CSE, skip it
        if i not in valid_indices:
            output_list.append(None)
            info_list.append(None)
            continue

        # Get the camera on frame i
        camera_i: PerspectiveCameras = cameras[i].to(device)

        # Filter keypoints with low prediction score
        filter_i = (keyp2d_score[i, :, 0] > filter_score).nonzero().reshape((-1,))

        # At least 3 keypoint with valid score to process
        if filter_i.shape[0] < 3:
            output_list.append(None)
            info_list.append(None)
            continue

        # Filter by keypoint score ->(1, N_PTS_FILTERED, 3)
        keyp_3d_i = keyp_3d[i, filter_i].unsqueeze(0)

        # Project 2D points from SCREEN -> NDC
        transform_screen_to_ndc = get_screen_to_ndc_transform(camera_i, with_xyflip=True, image_size=(image_size, image_size))
        keyp2d_prediction_i = transform_screen_to_ndc.transform_points(
            torch.cat([keyp2d_prediction[i, filter_i], torch.ones((keyp2d_prediction[i, filter_i].shape[0], 1)).to(keyp2d_prediction.device)], dim=1)
        )

        with torch.no_grad():
            try:
                # PNP with RANSAC estimation
                if use_RANSAC:
                    output, info_dict_ransac = efficient_PNP_RANSAC(
                        X_3d=keyp_3d_i,
                        y_2d_ndc=keyp2d_prediction_i[:, :2],
                        camera=camera_i,
                        n=4,
                        n_iter_max=n_iter_max,
                        thre_inl=thre_inl,
                        min_inl=round(keyp2d_prediction_i.shape[0] * min_inl_percentage),
                    )
                # PNP estimation
                else:
                    # Project 2D points from NDC -> CAMERA
                    keyp2d_unp = camera_i.unproject_points(
                        torch.cat([keyp2d_prediction_i, torch.ones((keyp2d_prediction_i.shape[0], 1)).to(keyp2d_prediction_i.device)], dim=1), world_coordinates=False
                    )[:, :2].unsqueeze(0)
                    with SuppressAllWarnings():
                        output = efficient_pnp(x=keyp_3d_i, y=keyp2d_unp)

            except Exception as e:
                print("Error processing")
                print(e)
                output_list.append(None)
                info_list.append(None)
                continue

        # Save the values
        output_list.append(output)
        info_list.append(info_dict_ransac)

    # ---------------------------------------------------------------
    # Return :
    # - valid indices
    # - stack of R and T tensors
    # INFO DICT:
    # - Nb keypoints per frame
    # - Error2D + Error3D per frame (from efficient_pnp method)
    # ---------------------------------------------------------------

    valid_ind = [i for i, elt in enumerate(output_list) if elt is not None]
    if use_RANSAC:
        valid_ind = [i for i in range(len(info_list)) if (output_list[i] is not None) and (info_list[i] is not None) and any([elt["is_best"] for elt in info_list[i]])]

    # Initialize returns
    info_dict = {}
    R_stack = torch.eye(3, dtype=torch.float32, device=device).reshape((1, 3, 3)).repeat((N_frames, 1, 1))
    T_stack = torch.zeros((N_frames, 3), dtype=torch.float32, device=device)
    info_dict["err_2d"] = torch.zeros((N_frames,), dtype=torch.float32, device="cpu")
    info_dict["err_3d"] = torch.zeros((N_frames,), dtype=torch.float32, device="cpu")
    info_dict["logs"] = [None for _ in range(N_frames)]
    info_dict["nb_keypoints"] = torch.tensor([len((keyp2d_score[i] > filter_score).nonzero()) for i in range(N_frames)])

    for j in valid_ind:
        info_dict["err_2d"][j] = output_list[j].err_2d.cpu()
        info_dict["err_3d"][j] = output_list[j].err_3d.cpu()
        info_dict["logs"][j] = info_list[j]
        R_stack[j] = output_list[j].R
        T_stack[j] = output_list[j].T

    return valid_ind, R_stack, T_stack, info_dict
