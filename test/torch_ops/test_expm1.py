from __future__ import print_function
import logging
import unittest
import sys
import os
import copy
import itertools
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase, TEST_BFLOAT16  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestExpm1Op(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_expm1(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        shape_list = [
            (10, 12, 10, 13),
            (2, 10, 15),
            (0, 3, 2, 1),
            (2, 0, 1),
            (1, 3, 6, 2, 4),
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for (dtype, err), shape, func in itertools.product(
            dtype_list, shape_list, func_list
        ):
            x = torch.randn(shape, dtype=dtype)
            out_cpu = torch.expm1(func(x.float()))
            out_mlu = torch.expm1(func(self.to_mlu_dtype(x, dtype)))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            x = torch.randn(shape, dtype=dtype)
            x_cpu = func(x.float())
            x_mlu = func(self.to_mlu_dtype(x, dtype))
            x_cpu.expm1_()
            x_mlu.expm1_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu().float(), err, use_MSE=True)

            x = torch.randn(shape, dtype=dtype)
            x_cpu = func(x.float())
            x_mlu = func(self.to_mlu_dtype(x, dtype))
            out = torch.randn(shape, dtype=dtype)
            out_cpu = func(out.float())
            out_mlu = func(self.to_mlu_dtype(out, dtype))
            torch.expm1(x_cpu, out=out_cpu)
            torch.expm1(x_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_expm1_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_cpu, out_cpu = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x.mlu().permute(permute_shape[i]), out.mlu().permute(
                permute_shape[i]
            )
            torch.expm1(x_cpu, out=out_cpu)
            torch.expm1(x_mlu, out=out_mlu)
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_expm1_backward(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        type_list = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(data_type)
                x_mlu = x.to("mlu")

                # use float on cpu kernel
                out_cpu = x_0.expm1()
                out_mlu = x_mlu.expm1()

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to("mlu")

                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)

                self.assertTensorsEqual(
                    out_grad_cpu,
                    out_grad_mlu.cpu().float()
                    if data_type == torch.half
                    else out_grad_mlu.cpu(),
                    0.003,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_expm1_bfloat16(self):
        shape = (39, 48)
        data_type = torch.bfloat16
        x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        x_mlu = x.to("mlu")
        out_cpu = x.expm1()
        out_mlu = x_mlu.expm1()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = grad.to("mlu")
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
