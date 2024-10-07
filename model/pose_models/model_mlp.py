# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Literal
from model.positional_embedding import PositionalEmbedding
from model.pose_models.model_util import _safe_parameter_init, partition_index, PoseBase


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        nb_layers: int,
        embedding_dimension: int,
        output_dimension: int,
        activ: Literal["softplus", "relu", "leaky_relu"] = "relu",
    ):
        super(MLP, self).__init__()

        if activ == "softplus":
            activation = nn.Softplus()
        elif activ == "relu":
            activation = nn.ReLU()
        elif activ == "leaky_relu":
            activation = nn.LeakyReLU()
        else:
            raise Exception("unknown activation")

        self.input_dimension = input_dimension
        self.nb_layers = nb_layers
        self.embedding_dimension = embedding_dimension
        self.output_dimension = output_dimension

        if self.nb_layers == 1:
            list_layers1 = [nn.Linear(self.input_dimension, self.output_dimension)]
        else:
            # First Layer
            list_layers1 = [nn.Linear(self.input_dimension, self.embedding_dimension), activation]
            # Hidden Layers
            if self.nb_layers > 2:
                for _ in range(self.nb_layers - 2):
                    list_layers1.extend([nn.Linear(self.embedding_dimension, self.embedding_dimension), activation])
            # Last Layer
            list_layers1.append(nn.Linear(self.embedding_dimension, self.output_dimension))

        self.model = nn.Sequential(*list_layers1)

    def forward(self, input):
        return self.model(input)

    def forward_partition(self, input, chunksize=10_000):
        BATCH, _ = input.shape
        list_partition = partition_index(BATCH, chunksize)
        return torch.cat([self.model(input[a:b]) for (a, b) in list_partition], dim=0)


class PoseMLP(PoseBase):

    def __init__(
        self,
        mlp_n_layers: int,
        mlp_hidd_dim: int,
        mlp_activation: Literal["softplus", "relu", "leaky_relu"],
        pos_embedding_dim: int,
        pos_embedding_mode: Literal["power", "lin"],
        init_betas: torch.Tensor | None = None,
        init_betas_limbs: torch.Tensor | None = None,
    ):
        """
        Animal Avatar learnable pose model.
        """
        super(PoseMLP, self).__init__()
        self.tag = "PoseMLP"

        # Positional embedder
        self.positional_embedder = PositionalEmbedding(L=pos_embedding_dim, mode=pos_embedding_mode)

        # Vertices additional offset (NON-TRAINABLE)
        self.vertices_off = _safe_parameter_init(None, (1, 5901), requires_grad=True, init="zeros")

        # Shape (Beta [1,30] + Beta limbs length [1,7])
        self.betas = _safe_parameter_init(init_betas, (1, 30), requires_grad=True)
        self.betas_limbs = _safe_parameter_init(init_betas_limbs, (1, 7), requires_grad=True)

        # MLP model (Translation (3) + Orientation (6) + Pose (34*6)) -> 213
        self.input_dim = self.positional_embedder.L
        self.output_dim = 3 + 6 + 34 * 6
        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_dim = mlp_hidd_dim

        self.model = MLP(
            input_dimension=self.input_dim,
            nb_layers=self.mlp_n_layers,
            embedding_dimension=self.mlp_hidden_dim,
            output_dimension=self.output_dim,
            activ=mlp_activation,
        )

    def forward(self, x: torch.Tensor):
        if x.device != self.betas.device:
            x = x.to(self.betas.device)
        if x.dim() == 1:
            x = x.reshape((-1, 1))
        embedded_x = self.positional_embedder.forward(x)
        return self.model.forward(embedded_x)

    def compute_pose(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.forward(X_ts)[..., 9:]

    def compute_orient(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.forward(X_ts)[..., 3:9]

    def compute_transl(self, X_ind: torch.Tensor, X_ts: torch.Tensor):
        pass

    def compute_betas(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        if self.betas.shape[0] > 1:
            return self.betas[X_ind].reshape(X_ind.shape[0], self.betas.shape[1])
        else:
            return self.betas.expand(X_ind.shape[0], -1)

    def compute_betas_limbs(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        if self.betas_limbs.shape[0] > 1:
            return self.betas_limbs[X_ind].reshape(X_ind.shape[0], self.betas_limbs.shape[1])
        else:
            return self.betas_limbs.expand(X_ind.shape[0], -1)

    def compute_vertices_off(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        if self.vertices_off.shape[0] > 1:
            return self.vertices_off[X_ind].reshape(X_ind.shape[0], self.vertices_off.shape[1])
        else:
            return self.vertices_off.expand(X_ind.shape[0], -1)

    def compute_global(self, X_ind: torch.Tensor, X_ts: torch.Tensor) -> torch.Tensor:
        return self.forward(X_ts)

    def get_trainable_list(self):
        return list(self.model.parameters()) + [self.betas, self.betas_limbs]
