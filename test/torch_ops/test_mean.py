from __future__ import print_function
import sys
import os
import itertools
import unittest
import logging
from itertools import product
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_mean_dim(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        param_list = [type_list, shape_list, func_list]
        for test_type, shape, func in itertools.product(*param_list):
            dim_len = len(shape)
            for i in range(1, dim_len + 1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + list(
                    itertools.permutations(range(-dim_len, 0), i)
                )
                for test_dim in dim_lists:
                    x = torch.randn(shape, dtype=torch.float)
                    out_cpu = func(x).mean(test_dim, keepdim=test_type)
                    out_mlu = func(self.to_mlu(x)).mean(test_dim, keepdim=test_type)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_mean(self):
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.mean(x)
            out_mlu = torch.mean(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mean_opt_dtype(self):
        shape = (2, 3, 4)
        type_list = [torch.int, torch.short, torch.int8, torch.long, torch.uint8]
        opt_list = [torch.half, torch.float, torch.double]
        for t, opttype in product(type_list, opt_list):
            x = (torch.randn(shape, dtype=torch.float) * 100).to(t)
            out_cpu = x.mean(dim=1, keepdim=True, dtype=opttype)
            out_mlu = x.to("mlu").mean(dim=1, keepdim=True, dtype=opttype)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

        for t, opttype in product(type_list, opt_list):
            x = (torch.randn(shape, dtype=torch.float) * 100).to(t)
            out_cpu = x.mean(dtype=opttype)
            out_mlu = x.to("mlu").mean(dtype=opttype)

            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_mean_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.mean(x)
        out_mlu = torch.mean(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mean_out(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len + 1):
                dim_lists = list(itertools.permutations(range(0, dim_len), i)) + list(
                    itertools.permutations(range(-dim_len, 0), i)
                )
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = torch.randn(1)
                        out_mlu = self.to_mlu(torch.randn(1))
                        x_mlu = self.to_mlu(x)
                        torch.mean(x, test_dim, keepdim=test_type, out=out_cpu)
                        torch.mean(x_mlu, test_dim, keepdim=test_type, out=out_mlu)
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_mean_empty(self):
        x = torch.randn(1, 0, 1)
        out_mlu = self.to_mlu(x).mean()
        # MLU370 returns nan. nan != nan itself.
        assert out_mlu.cpu().item() != out_mlu.cpu().item()
        # if is_using_floating_device():
        #    # MLU370 returns nan. nan != nan itself.
        #    assert out_mlu.cpu().item() != out_mlu.cpu().item()
        # else:
        #    assert out_mlu.cpu().item() == 0

    # @unittest.skip("not test")
    @testinfo()
    def test_mean_backward(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_device(x)

                out_cpu = torch.mean(x, item[1], keepdim=item[0])
                grad = torch.randn(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                out_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()

                out_mlu = torch.mean(x_mlu, item[1], keepdim=item[0])
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_mean_exception(self):
        a = torch.randn((3, 4)).int().to("mlu")
        b = torch.tensor(1).to("mlu")
        ref_msg = r"mean\(\): could not infer output dtype. "
        ref_msg = (
            ref_msg
            + "Input dtype must be either a floating point or complex dtype. Got: Int"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.mean(dim=1)

        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.mean()

        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.mean(a, out=b, dim=1)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_mean_bfloat16(self):
        left = torch.testing.make_tensor(
            (1, 1, 1024), dtype=torch.bfloat16, device="cpu"
        )
        left_cpu = torch.nn.Parameter(left)
        left_mlu = torch.nn.Parameter(left.mlu())
        out_cpu = torch.mean(left_cpu, 1, keepdim=True)
        out_mlu = torch.mean(left_mlu, 1, keepdim=True)
        grad = torch.randn(out_cpu.shape).bfloat16()
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            left_cpu.grad.float(), left_mlu.grad.cpu().float(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_mean_large(self):
        shape_list = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.mean(x)
            out_mlu = torch.mean(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
