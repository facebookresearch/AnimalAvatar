# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Resize methods ----------------------------------------------------------------------------------------------------------------------------------------------------------------


def resize_torch(X: torch.Tensor, size: tuple[int, int], mode="bilinear") -> torch.Tensor:
    """
    Args:
        - X TORCHFLOAT32 (BATCH, H, W, CHANNEL) OR (H,W,CHANNEL) or List of elements
        - size (h_new, w_new)

    Return:
        - output TORCHFLOAT32 (BATCH, h_new, w_new, CHANNEL) or (h_new, w_new, CHANNEL)
    """
    if type(X) == list:
        return [resize_torch(x) for x in X]
    if len(X.shape) == 3:
        return F.interpolate(X.permute((2, 0, 1)).unsqueeze(0), (size[0], size[1]), mode=mode).permute((0, 2, 3, 1))[0]
    else:
        return F.interpolate(X.permute((0, 3, 1, 2)), (size[0], size[1]), mode=mode).permute((0, 2, 3, 1))


def resize_torch_bool(X: torch.Tensor, size: tuple[int, int], alpha: float = 0.5, mode: str = "bilinear") -> torch.Tensor:
    """
    Args:
        - X TORCHBOOL (BATCH, H, W, CHANNEL) OR (H,W,CHANNEL) or List of elements
        - size (h_new, w_new)
    Return:
        - output TORCHFLOAT32 (BATCH, h_new, w_new, CHANNEL) or (h_new, w_new, CHANNEL)
    """
    if type(X) == list:
        return [resize_torch_bool(x) for x in X]
    if len(X.shape) == 3:
        return (F.interpolate(X.type(torch.float32).permute((2, 0, 1)).unsqueeze(0), (size[0], size[1]), mode=mode).permute((0, 2, 3, 1))[0] >= alpha).type(torch.bool)
    else:
        return (F.interpolate(X.type(torch.float32).permute((0, 3, 1, 2)), (size[0], size[1]), mode=mode).permute((0, 2, 3, 1)) >= alpha).type(torch.bool)


def resize_np(X: np.ndarray, size: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    """
    Args:
        - X NPFLOAT32 (BATCH, H, W, CHANNEL) OR (H,W,CHANNEL) or List of elements
        - size (h_new, w_new)

    Return:
        - output NPFLOAT32 (BATCH, h_new, w_new, CHANNEL) or (h_new, w_new, CHANNEL)
    """
    if type(X) == list:
        return [resize_np(x) for x in X]
    if len(X.shape) == 3:
        return F.interpolate(torch.tensor(X).permute((2, 0, 1)).unsqueeze(0), (size[0], size[1]), mode=mode).permute((0, 2, 3, 1))[0].numpy()
    else:
        return F.interpolate(torch.tensor(X).permute((0, 3, 1, 2)), (size[0], size[1]), mode=mode).permute((0, 2, 3, 1)).numpy()


def resize_np_uint8(X: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Args:
        - X NPUINT8 (H,W,CHANNEL) or (BATCH, H, W, CHANNEL) or List of elements

    Return:
        - output NPUINT8 (H,W,CHANNEL) or (BATCH, H, W, CHANNEL)

    """
    if type(X) == list:
        return [resize_np_uint8(x) for x in X]
    if len(X.shape) == 3:
        return cv2.resize(X, (size[0], size[1]), cv2.INTER_AREA)
    else:
        return np.stack([cv2.resize(X[i], (size[0], size[1]), cv2.INTER_AREA) for i in range(X.shape[0])], axis=0)


def resize_image(
    image: np.ndarray | torch.Tensor,
    image_height: int | None,
    image_width: int | None,
    mode: str = "bilinear",
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """
    CoP3D resizing method.

    Args:
        - images TORCH/NPY (BATCH, H_in, W_in, 3)

    Returns
        - imre_ (BATCH, image_height, image_width, 3)
        - minscale
        - mask (BATCH, image_height, image_width, 3)
    """

    if type(image) == np.ndarray:
        image = torch.from_numpy(image)

    if image_height is None or image_width is None:
        # skip the resizing
        return image, 1.0, torch.ones_like(image[:1])
    # takes numpy array or tensor, returns pytorch tensor
    minscale = min(
        image_height / image.shape[-2],
        image_width / image.shape[-1],
    )
    imre = torch.nn.functional.interpolate(
        image[None],
        scale_factor=minscale,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        recompute_scale_factor=True,
    )[0]
    imre_ = torch.zeros(image.shape[0], image_height, image_width)
    imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
    mask = torch.zeros(1, image_height, image_width)
    mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
    return imre_, minscale, mask


# Normalize methods ----------------------------------------------------------------------------------------------------------------------------------------------------------------


def normalize_torch(embeddings: torch.Tensor, dim: int = 1, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Normalize N D-dimensional embedding vectors arranged in a tensor [N, D]

    Args:
        embeddings (tensor [N, D]): N D-dimensional embedding vectors
        epsilon (float): minimum value for a vector norm
    Return:
        Normalized embeddings (tensor [N, D]), such that L2 vector norms are all equal to 1.
    """
    return embeddings / torch.clamp(embeddings.norm(p=None, dim=dim, keepdim=True), min=epsilon)


def normalize_numpy(embeddings: np.ndarray, dim: int = 1, epsilon: float = 1e-6) -> np.ndarray:
    """
    Normalize N D-dimensional embedding vectors arranged in a numpy array [N, D]
    Args:
        embeddings (np.ndarray [N, D]): N D-dimensional embedding vectors
        dim (int): Dimension along which to compute the norm.
        epsilon (float): Minimum value for a vector norm to prevent division by zero.
    Returns:
        Normalized embeddings (np.ndarray [N, D]), such that L2 vector norms are all equal to 1.
    """
    norms = np.linalg.norm(embeddings, axis=dim, keepdims=True)
    return embeddings / np.clip(norms, a_min=epsilon, a_max=None)
