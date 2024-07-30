# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path

## Image blending ------------------------------------------------------------------------------------------------------------


def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float, image2_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Blend the images using alpha
    Args:
        - image1 (IMG_W, IMG_H, 3) NUMPY UINT8/FLOAT32
        - image2 (IMG_W, IMG_H, 3) NUMPY UINT8/FLOAT32
        - image2_mask (IMG_W, IMG_H, 1) NUMPY UINT8/FLOAT32 {0,1}
    Return:
        - (IMG_W, IMG_H, 3) NUMPY UINT8/FLOAT32
    """
    assert 0.0 <= alpha <= 1.0
    assert image1.shape[-1] == 3
    assert image2.shape[-1] == 3

    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    if image2_mask is None:
        blended_image[(image2[:, :, 2] < 0.01) & (image2[:, :, 1] < 0.01) & (image2[:, :, 0] < 0.01)] = image1[(image2[:, :, 2] < 0.01) & (image2[:, :, 1] < 0.01) & (image2[:, :, 0] < 0.01)]
    else:
        blended_image[(image2_mask[:, :, 0] == 0)] = image1[(image2_mask[:, :, 0] == 0)]

    return blended_image


def blend_N_images(X_imgs1: np.ndarray, X_imgs2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend N images with original image, reconstruction

    Args:
        - X_imgs1 (BATCH, IMG_H, IMG_W, 3) NUMPY UINT8/FLOAT32
        - X_imgs2 (BATCH, IMG_H, IMG_W, 3) NUMPY UINT8/FLOAT32
    Return:
        - (BATCH, IMG_H, IMG_W, 3) NUMPY UINT8/FLOAT32
    """
    return np.stack([blend_images(X_imgs1[k], X_imgs2[k], alpha) for k in range(X_imgs1.shape[0])], axis=0)


## Video generation ------------------------------------------------------------------------------------------------------------


def make_video(X_imgs: np.ndarray, output_path: str, fps: int = 20):
    """
    Args:
        - X_imgs (BATCH,M,N,3) NPUINT8
    """
    duration = int(np.round(1000 * X_imgs.shape[0] / fps))

    with imageio.get_writer(output_path, mode="I", fps=fps) as writer:
        for k in tqdm(range(X_imgs.shape[0]), "save video ({})".format(Path(output_path).name)):
            writer.append_data(X_imgs[k])


def make_video_list(list_X_imgs: list[np.array], output_path: str, fps: int = 20):
    """
    Args:
        - list_X_imgs [(BATCH,M,N,3), BATCH,M,N,3), ...] NPUINT8
    """
    with imageio.get_writer(output_path, mode="I", fps=fps) as writer:
        for k in tqdm(range(list_X_imgs[0].shape[0]), desc="save video ({})".format(Path(output_path).name)):
            output = np.concatenate([X_imgs[k] for X_imgs in list_X_imgs], axis=1)
            writer.append_data(output)


## Keypoints  --------------------------------------------------------------------------------------


def get_cv_colorlist(nb_colors: int, colormap) -> np.ndarray:
    """
    return array((nb_colors,3))
    """
    l_v = np.linspace(0, 255, nb_colors)
    return cv2.applyColorMap(np.array([l for l in l_v]).astype(np.uint8).reshape((-1, 1)), colormap=colormap).reshape(-1, 3)


def add_keypoints_on_image_cmap(image: np.ndarray, keypoints: np.ndarray, nb_colors: int = -1, size: int = 2, colormap=cv2.COLORMAP_HSV):
    """
    Add keypoints on an image

    Args:
        - image (H,W,3) NPUINT8
        - keypoints (M,2) NPFLOAT32 (in image size (H,W))

    Return:
        - image (H,W,3) NPUINT8

    """
    if nb_colors == -1:
        nb_colors = keypoints.shape[0]
    colors = get_cv_colorlist(nb_colors, colormap)

    output_image = np.ascontiguousarray(np.copy(image))

    for i, kp in enumerate(keypoints):
        x, y = np.round(kp[0]), np.round(kp[1])
        cv2.circle(output_image, (int(x), int(y)), size, [int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])], -1)

    return output_image


def N_add_keypoints_on_image_cmap(image: np.ndarray, keypoints: np.ndarray, size: int = 2, colormap=cv2.COLORMAP_HSV):
    """
    Batched version of add_keypoints_on_image_cmap

    Args:
        - image (BATCH,H,W,3) NPUINT8
        - keypoints (BATCH,M,2) NPFLOAT32 (in image size (H,W))
    Return:
        - image (BATCH,H,W,3) NPUINT8
    """
    return np.stack([add_keypoints_on_image_cmap(image[k], keypoints[k], size=size, colormap=colormap) for k in range(image.shape[0])], axis=0)
