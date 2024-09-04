from __future__ import print_function

import sys
import os
import unittest
import logging
import torch
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_close_view_chain(self):
        os.environ["TORCH_MIN_CNLOG_LEVEL"] = "INFO"
        os.environ["ENABLE_PRINT_VIEW_CHAIN"] = "1"
        os.environ["DISABLE_VIEWCHAIN_FUSED_FUNC"] = "1"
        shape_permute = [(3, 24, 1, 24), (0, 2, 1, 3), 1, (2, 0, 1)]
        shape, permute_index1, squeeze_dim, permute_index2 = shape_permute
        input_t = torch.rand(shape)
        input_mlu = input_t.mlu()
        output_cpu = (
            input_t[:, 0:20:2, ...]
            .permute(permute_index1)
            .squeeze(squeeze_dim)
            .permute(permute_index2)
            + 1
        )
        output_mlu = (
            input_mlu[:, 0:20:2, ...]
            .permute(permute_index1)
            .squeeze(squeeze_dim)
            .permute(permute_index2)
            + 1
        )
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
