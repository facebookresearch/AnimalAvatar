# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

SMOOTH = 1e-6


def iou(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Args:
        - output: (BATCH, M, N)
        - labels: (BATCH, M, N)
    Returns:
        - iou: (BATCH,)
    """
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou


def psnr(x_gt: torch.Tensor, y: torch.Tensor, max_p: float = 1.0) -> torch.Tensor:
    """
    Args:
        - x (BATCH, ...) TORCHFLOAT32 [DEVICE]
        - y (BATCH, ...) TORCHFLOAT32 [DEVICE]
    Returns:
        - psnr (BATCH,) TORCHFLOAT32 [DEVICE]
    """
    size = x_gt.dim()
    mse = torch.mean((x_gt - y) ** 2, dim=list(range(1, size)))
    psnr = 20 * torch.log10(max_p**2 / (torch.sqrt(mse) + 1e-6))
    return psnr
