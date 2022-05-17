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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
from random import shuffle


@flow.unittest.skip_unless_1n1d()
class TestCeilModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_ceil_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.ceil(input)
        return y

    @autotest(check_graph=True)
    def test_ceil_flow_stride_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        permute_list = list(range(5))
        shuffle(permute_list)
        input2 = input.permute(permute_list)
        y = torch.ceil(input2)
        return y

    @autotest(check_graph=True)
    def test_ceil_flow_with_random_0d_data(test_case):
        device = random_device()
        input = random_tensor(ndim=0).to(device)
        y = torch.ceil(input)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_ceil_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.ceil(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_ceil_with_0shape_0d_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.ceil(x)
        return y


if __name__ == "__main__":
    unittest.main()
