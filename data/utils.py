# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def train_test_split_frames(X_ind: torch.Tensor, train_n: int = 15, test_n: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each train_n training frames, have test_n frames next (and repeat).
    Args:
        - X_ind (B,) TORCHINT32
    """
    if test_n == 0:
        return X_ind, X_ind

    N = X_ind.shape[0]

    train_list, test_list = [], []
    for i in range(N):
        if i % (train_n + test_n) < train_n:
            train_list.append(i)
        else:
            test_list.append(i)

    X_train = X_ind[torch.tensor(train_list).reshape((-1,)).to(X_ind.device)]
    X_test = X_ind[torch.tensor(test_list).reshape((-1,)).to(X_ind.device)]
    return X_train, X_test


def array_list_to_stack(array_list: list[np.ndarray]) -> list[np.ndarray] | np.ndarray:
    """
    If the arrays have the same dimension, concatenate, otherwise return a list
    """
    same_array_shape = all(arr.shape == array_list[0].shape for arr in array_list)
    if same_array_shape:
        return np.concatenate(array_list, axis=0)
    else:
        return array_list
