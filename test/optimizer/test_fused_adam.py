from __future__ import print_function
import logging
import sys
import os
import unittest

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class TestFusedAdam(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFusedAdam, self).__init__(*args, **kwargs)
        self.options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        self.iters = 10
        self.max_abs_diff = 1e-3
        self.max_rel_diff = 1
        self.ori_optim = torch.optim.Adam
        self.fused_optim = torch_mlu.optimizers.FusedAdam

    def fused_adam_dtype(self, test_type=torch.float, sz=[40, 40], len=2):
        tensors = []
        fused_opt_params = []
        opt_params = []

        for i in range(len):
            tensors.append(
                torch.clamp(
                    torch.rand(sz, dtype=torch.float).to("mlu"), min=0.01, max=100.0
                ).to(test_type)
            )

        for tensor in tensors:
            fused_opt_params.append(torch.nn.Parameter(tensor.clone()))
            opt_params.append(torch.nn.Parameter(tensor.clone()))
        fused_optimizer = torch_mlu.optimizers.FusedAdam(
            fused_opt_params, **self.options
        )
        optimizer = torch.optim.Adam(opt_params, **self.options)

        for _ in range(self.iters):
            for p_opt, p_fused_opt in zip(opt_params, fused_opt_params):
                p_opt.grad = torch.rand_like(p_opt).to("mlu")
                p_fused_opt.grad = p_opt.grad
            optimizer.step()
            fused_optimizer.step()
            max_abs_diff = max_rel_diff = 0
            for p_ref, p_tst in zip(opt_params, fused_opt_params):
                max_abs_diff_p = (p_ref - p_tst).abs().max().item()
                max_rel_diff_p = ((p_ref - p_tst) / p_ref).abs().max().item()
                if max_abs_diff_p > max_abs_diff:
                    max_abs_diff = max_abs_diff_p
                if max_rel_diff_p > max_rel_diff:
                    max_rel_diff = max_rel_diff_p

            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_adam_fp32(self):
        shape_list = [(1), (15, 20), (2, 3, 4), (8, 1, 2, 3), (2, 1, 2, 1, 4)]
        for sz in shape_list:
            for wd in [0, 0.01]:
                self.options["weight_decay"] = wd
                for len in [2, 40]:
                    self.fused_adam_dtype(test_type=torch.float, sz=sz, len=len)


if __name__ == "__main__":
    unittest.main()
