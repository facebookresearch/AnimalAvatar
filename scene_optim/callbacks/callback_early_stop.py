# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .callback_general import CallbackDataClass


class CallbackEarlyStop(CallbackDataClass):

    def __init__(self, min_condition_dict: dict[str, float] | None = None, max_condition_dict: dict[str, float] | None = None, min_epoch: int = 0):
        self.min_condition_dict = min_condition_dict
        self.max_condition_dict = max_condition_dict
        self.min_epoch = min_epoch
        self.state = False

    def call(self, epoch: int, data_dict: dict) -> bool:

        if (self.min_condition_dict is None) and (self.max_condition_dict is None):
            return False

        if self.state == True:
            return self.state

        if epoch < self.min_epoch:
            return self.state

        for k in data_dict:
            if k in self.min_condition_dict and (data_dict[k] <= self.min_condition_dict[k]):
                print("Early stop {}:{}".format(k, data_dict[k]))
                self.state = True
                return self.state
            if k in self.max_condition_dict and (data_dict[k] >= self.max_condition_dict[k]):
                print("Early stop {}:{}".format(k, data_dict[k]))
                self.state = True
                return self.state

        return self.state
