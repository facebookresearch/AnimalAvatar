# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .loss_optim_kps import LossOptimCSEKp, LossOptimSparseKp
from .loss_optim_regularization import LossLaplacianReg
from .loss_optim_color_rend import LossOptimColor
from .loss_optim_chamfer import LossOptimChamfer
from .loss_optim_tv_reg import LossOptimTVReg
from .loss_optim_arap_reg import LossOptimArap
from .loss_optim_arap_fast_reg import LossOptimArapFast
