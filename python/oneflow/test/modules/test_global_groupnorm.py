"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False)
def _test_global_group_norm(test_case, placement, input_sbp):
    if placement.type == "cpu":
        return
    batch_size = 8
    channel_size = 4
    num_groups = 2
    m = torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=channel_size, affine=True
    )
    m.train(random())
    m.weight = torch.nn.Parameter(
        m.weight.to_global(
            placement=placement, sbp=[flow.sbp.broadcast] * len(placement.ranks.shape)
        )
    )
    m.bias = torch.nn.Parameter(
        m.bias.to_global(
            placement=placement, sbp=[flow.sbp.broadcast] * len(placement.ranks.shape)
        )
    )
    x = random_tensor(ndim=4, dim0=batch_size, dim1=channel_size).to_global(
        placement=placement, sbp=input_sbp
    )
    y = m(x)
    return y


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGroupNormModule(flow.unittest.TestCase):
    @globaltest
    def test_global_group_norm_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_group_norm(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
