from __future__ import print_function
import sys
import os
import unittest
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    run_tests,
    testinfo,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestErfinvOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_contiguous(self):
        for func in [torch.erfinv, torch.special.erfinv]:
            for shape in [
                (),
                (2, 3),
                (2, 3, 4),
                (1, 1, 1, 1),
                (1, 3, 16, 16),
                (1, 3, 16, 16, 3),
            ]:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = func(x)
                out_mlu = func(x.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

                # test inplace operation
                x_cpu = copy.deepcopy(x)
                x_mlu = x.to("mlu")
                x_cpu.erfinv_()
                x_mlu.erfinv_()
                self.assertTensorsEqual(
                    x_cpu, x_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_channel_last(self):
        for func in [torch.erfinv, torch.special.erfinv]:
            for shape in [(1, 3, 16, 16)]:
                x = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                out_cpu = func(x)
                out_mlu = func(x.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

                # test inplace operation
                x_cpu = copy.deepcopy(x)
                x_mlu = x.to("mlu")
                x_cpu.erfinv_()
                x_mlu.erfinv_()
                self.assertTensorsEqual(
                    x_cpu, x_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_not_dense(self):
        for func in [torch.erfinv, torch.special.erfinv]:
            for shape in [(1, 3, 16, 16)]:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = func(x[:, :, :, :8])
                out_mlu = func(x.to("mlu")[:, :, :, :8])
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

                # test inplace operation
                x_cpu = copy.deepcopy(x)
                x_mlu = x.to("mlu")
                x_cpu.erfinv_()
                x_mlu.erfinv_()
                self.assertTensorsEqual(
                    x_cpu, x_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_backward(self):
        for func in [torch.erfinv, torch.special.erfinv]:
            for shape in [(2, 3), (1, 1, 1, 1), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = func(x)
                out_mlu = func(x.to("mlu"))
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x_cpu = copy.deepcopy(x)
                x.grad.zero_()
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )
                self.assertTensorsEqual(x, x_cpu, 0.003, use_MSE=True)

                # test inplace operation
                x.grad.zero_()
                x_mlu = x.to("mlu")
                x_mlu.erfinv_()
                x_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_permute(self):
        shape_list = [
            (12, 24, 2, 2, 4),
            (10, 3, 2, 2),
            (2, 3, 4),
            (25, 25, 12, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for func in [torch.erfinv, torch.special.erfinv]:
            for i in range(4):
                x = torch.randn(shape_list[i], dtype=torch.float)
                out = torch.randn(shape_list[i], dtype=torch.float)
                x_mlu = copy.deepcopy(x).to("mlu")
                out_mlu = copy.deepcopy(out).to("mlu")
                x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
                x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                    permute_shape[i]
                )
                func(x, out=out)
                func(x_mlu, out=out_mlu)
                self.assertTrue(out.stride() == out_mlu.stride())
                self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
                self.assertTensorsEqual(
                    out, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_type(self):
        shape_list = [(2, 3, 4)]
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        for func in [torch.erfinv, torch.special.erfinv]:
            for shape in shape_list:
                for type in type_list:
                    x_cpu = torch.randn(shape).to(type)
                    x_mlu = self.to_mlu(x_cpu)
                    if type == torch.half:
                        x_cpu = x_cpu.float()
                    out_cpu = func(x_cpu)
                    out_mlu = func(x_mlu)
                    self.assertTensorsEqual(
                        out_cpu, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                    )
                    if type == torch.half:
                        self.assertTrue(torch.half == out_mlu.dtype)
                    else:
                        self.assertTrue(out_cpu.dtype == out_mlu.dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_erfinv_scalar(self):
        for func in [torch.erfinv, torch.special.erfinv]:
            for scalar in [0.2, 0.555, 0.88, 1, -1, float("nan"), float("inf")]:
                x = torch.tensor(scalar)
                out_cpu = func(x)
                out_mlu = func(x.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("50GB")
    def test_erfinv_large(self):
        shape_list = [
            (5, 1024, 1024, 1024),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.erfinv(x)
            out_mlu = torch.erfinv(x.to("mlu"))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_erfinv_bfloat16(self):
        for func in [torch.erfinv, torch.special.erfinv]:
            shape = [2, 3, 4]
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            out_cpu = func(x)
            out_mlu = func(x.to("mlu"))
            grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x_cpu = copy.deepcopy(x)
            x.grad.zero_()
            out_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True, allow_inf=True
            )
            self.assertTensorsEqual(x, x_cpu, 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
