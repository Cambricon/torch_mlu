import logging
import sys
import os
import unittest

import torch
import torch_mlu
from torch.autograd import Variable

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class TestFusedSGD(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFusedSGD, self).__init__(*args, **kwargs)
        self.options = {"lr": 0.25, "momentum": 0.125}
        self.fused_optimizer = torch_mlu.optimizers.FusedSGD
        self.origin_optimizer = torch.optim.SGD
        self.iters = 10
        self.max_abs_diff = 1e-3
        self.max_rel_diff = 1

    def fused_sgd_dtype(self, test_type=torch.float, sz=[40, 40]):
        sizes = [
            sz,
        ]
        tensors = []
        fused_opt_params = []
        opt_params = []
        torch.manual_seed(9876)

        for size in sizes:
            tensors.append(
                (torch.rand(size, dtype=torch.float)).to("mlu").to(test_type)
            )
        for tensor in tensors:
            fused_opt_params.append(torch.nn.Parameter(tensor.clone()))
            opt_params.append(torch.nn.Parameter(tensor.clone()))
        fused_optimizer = self.fused_optimizer(fused_opt_params, **self.options)
        origin_optimizer = self.origin_optimizer(opt_params, **self.options)

        for i in range(self.iters):
            for p_opt, p_fused_opt in zip(opt_params, fused_opt_params):
                p_opt.grad = torch.rand_like(p_opt).to("mlu")
                p_fused_opt.grad = p_opt.grad
            origin_optimizer.step()
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

    @testinfo()
    def test_fused_sgd_fp32(self):
        shape_list = [
            [1],
            [300],
            [4096],
            [278011],
            [10, 10],
            [100, 100],
            [4096, 1024],
            [4096, 2048],
            [32320, 1024],
        ]
        for sz in shape_list:
            self.fused_sgd_dtype(test_type=torch.float, sz=sz)

    # TODO:fix fp16 absolute error and relativate error between fused sgd and original sgd.
    @unittest.skip("not test")
    def test_fused_sgd_fp16(self):
        shape_list = [
            [1],
            [300],
            [4096],
            [278011],
            [10, 10],
            [100, 100],
            [4096, 1024],
            [4096, 2048],
            [32320, 1024],
        ]
        for sz in shape_list:
            self.fused_sgd_dtype(test_type=torch.half, sz=sz)

    @testinfo()
    def test_fused_sgd_exception(self):
        ref_msg = "Found param with index=0 was not contiguous."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            device = "mlu"
            weight = torch.randn(10, 5, 2).to(device)[..., 0]
            bias = torch.randn(10, 2).to(device)[..., 0]
            input = torch.randn(5, 1).to(device)

            weight = Variable(weight, requires_grad=True)
            bias = Variable(bias, requires_grad=True)
            optimizer = self.fused_optimizer([weight, bias], lr=1e-3)

            def fn():
                optimizer.zero_grad()
                y = weight.matmul(input)
                if y.is_mlu and bias.is_mlu and y.get_device() != bias.get_device():
                    y = y.mlu(bias.get_device())
                loss = (y + bias).pow(2).sum()
                loss.backward()
                return loss

            optimizer.step(fn)


if __name__ == "__main__":
    unittest.main()
