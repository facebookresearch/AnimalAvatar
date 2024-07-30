# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import math
from typing import Literal


def get_freqs_poslin(L: int, max_period: int = 10_000) -> torch.Tensor:
    half = L // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    return freqs


def get_freqs_power_nls(L: int) -> torch.Tensor:
    freqs = torch.linspace(1.0, 2.0 ** ((L // 2) - 1), L // 2, dtype=torch.float32)
    return freqs


def get_freqs_power_ls(L: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(L // 2, dtype=torch.float32)
    return freqs


def get_freq_linear(L: int):
    freqs = torch.arange(L // 2, dtype=torch.float32)
    return freqs


def get_freq_gaussian(L: int) -> torch.Tensor:
    mean_, std_ = 0.0, 0.01
    freqs = torch.normal(mean=torch.full((L // 2,), mean_), std=torch.full((L // 2,), std_)).type(torch.float32)
    return freqs


def compute_embedding(X: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        - X (BATCH, 1) TORCHFLOAT32
        - freqs (F,) TORCHFLOAT32
    """
    args = X.float() * freqs
    return torch.cos(args), torch.sin(args)


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding of a Tensor
    """

    def __init__(self, L: int = 10, mode: Literal["poslin", "power", "power_nls", "lin", "normal"] = "lin"):
        super(PositionalEmbedding, self).__init__()
        self.L = L
        self.mode = mode

        if mode == "poslin":
            self.freqs = nn.Parameter(get_freqs_poslin(self.L), requires_grad=False)

        elif mode == "power":
            self.freqs = nn.Parameter(get_freqs_power_ls(self.L), requires_grad=False)

        elif mode == "power_nls":
            self.freqs = nn.Parameter(get_freqs_power_nls(self.L), requires_grad=False)

        elif mode == "lin":
            self.freqs = nn.Parameter(get_freq_linear(self.L), requires_grad=False)

        elif mode == "normal":
            self.freqs = nn.Parameter(get_freq_gaussian(self.L), requires_grad=False)

        else:
            raise Exception("unknown harmonic mode")

    def forward(self, X: torch.Tensor, include_original: bool = False):

        cos_encoding, sin_encoding = compute_embedding(X, self.freqs)

        if include_original:
            return torch.cat([X, cos_encoding, sin_encoding], axis=-1)
        else:
            return torch.cat([cos_encoding, sin_encoding], axis=-1)
