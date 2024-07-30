# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def flow_to_matches(flow: torch.Tensor) -> torch.Tensor:
    """
    Args:
        - flow (BATCH, 2, H, W) TORCHFLOAT32 [DEVICE]
    Return:
        For optical_flow i->j
        - grid_source (BATCH, H, W) (corresponding position on j)
    """

    vx, vy = flow[:, 0], flow[:, 1]
    n, h, w = vx.shape

    # create grid X, Y of shape (n, h, w)
    Y, X = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, device=flow.device), torch.linspace(0.5, w - 0.5, w, device=flow.device), indexing="ij")
    X = X[None].repeat(n, 1, 1)
    Y = Y[None].repeat(n, 1, 1)

    # apply optical flow
    X_hat = X + vx
    Y_hat = Y + vy

    normalize = lambda x, L: 2.0 * x / L - 1.0
    grid_source = torch.stack((normalize(X_hat, w), normalize(Y_hat, h)), dim=-1)
    return grid_source


def warp_image(image_j: torch.Tensor, flow_i_to_j: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """
    Args:
        - image_j (BATCH, D, H, W) TORCHFLOAT32 [DEVICE]
        - flow_i_to_j (BATCH, 2, H, W) TORCHFLOAT32 [DEVICE]
    Return:
        - im_warped_in_i (BATCH, D, H, W) TORCHFLOAT32 [DEVICE]
    """
    BATCH, _, H, W = flow_i_to_j.shape

    if (image_j.shape[0] != BATCH) or (image_j.shape[2:] != (H, W)):
        raise Exception("Image and flow shape don't match")

    grid = flow_to_matches(flow_i_to_j).to(dtype=image_j.dtype)

    im_warped_in_i = F.grid_sample(image_j, grid, align_corners=False, mode=mode)

    return im_warped_in_i


def forward_backward_flow_check(f_flow: torch.Tensor, b_flow: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Args:
        - f_flow, b_flow TORCHFLOAT32 (BATCH, 2, H, W) [DEVICE]

    Returns:
        - mask (BATCH, H, W) TORCHBOOL
    """
    b_flow_warped = warp_image(b_flow, f_flow)
    diffs = torch.norm(f_flow + b_flow_warped, dim=1)
    mask = diffs <= threshold
    return mask
