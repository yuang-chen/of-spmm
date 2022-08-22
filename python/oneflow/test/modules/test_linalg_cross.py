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
import numpy as np
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestLinalgCross(flow.unittest.TestCase):
    @autotest(n=20, auto_backward=True)
    def test_linalg_cross_with_random_data(test_case):
        device = random_device()
        ndim = np.random.randint(2, 6)
        shape = list(np.random.randint(16, size=ndim))
        index = np.random.randint(ndim)
        shape[index] = 3
        print(ndim, index, shape)

        x = random_tensor(ndim, *shape).to(device)
        y = random_tensor(ndim, *shape).to(device)
        return torch.linalg.cross(x, y, dim=index)


if __name__ == "__main__":
    unittest.main()
