from __future__ import print_function

import sys
import logging
import os
import copy
import unittest
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestKlDivOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_kl_div(self):
        shape_list = [(2, 4, 3)]
        reduct_lst = ["none", "batchmean", "sum"]
        log_target_lst = [True, False]
        for lt in log_target_lst:
            for reduct in reduct_lst:
                for shape in shape_list:
                    kl_div = torch.nn.KLDivLoss(reduction=reduct, log_target=lt)
                    x = torch.randn(shape, dtype=torch.float)
                    y = torch.randn(shape, dtype=torch.float)
                    x_mlu = copy.deepcopy(x).to("mlu")
                    y_mlu = copy.deepcopy(y).to("mlu")

                    x.requires_grad = True
                    y.requires_grad = True
                    x_mlu.requires_grad = True
                    y_mlu.requires_grad = True

                    out_cpu = kl_div(x, y)
                    out_mlu = kl_div(x_mlu, y_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

                    out_cpu.backward(out_cpu)
                    out_mlu.backward(out_mlu)

                    self.assertTensorsEqual(
                        x.grad, x_mlu.grad.cpu(), 3e-3, use_MSE=True
                    )


if __name__ == "__main__":
    unittest.main()
