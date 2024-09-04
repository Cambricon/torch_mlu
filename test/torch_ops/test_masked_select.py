from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

# pylint: disable=C0413,C0411
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestMaskedSelect(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,)]
        dtype = [
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
            torch.double,
        ]
        for type_ in dtype:
            for shape in shapes:
                is_64bit = type_ in (torch.double, torch.long)
                x = torch.rand(shape).to(type_)
                x_mlu = self.to_device(x)
                if type_.is_floating_point:
                    x = torch.nn.Parameter(x)
                    x_mlu = torch.nn.Parameter(x_mlu)
                mask = torch.randn(shape) > 0
                mask_mlu = self.to_device(mask)
                out_cpu = torch.masked_select(x, mask)
                out_mlu = torch.masked_select(x_mlu, mask_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.00 if not is_64bit else 0.003
                )
                if type_.is_floating_point:
                    grad = torch.randn_like(out_cpu)
                    grad_mlu = self.to_device(grad)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        x.grad, x_mlu.grad.cpu(), 0.00 if not is_64bit else 0.003
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_mask_select_broadcast(self):
        shapes = [((3, 4, 5), (4, 5)), ((2, 5, 5, 3), (3))]
        dtype = [
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]
        for type_ in dtype:
            for shape1, shape2 in shapes:
                a = torch.randn(shape1)
                b = torch.randn(shape2)
                a_mlu, b_mlu = a.to("mlu"), b.to("mlu")
                out_cpu = torch.masked_select(a, b > 0)
                out_mlu = torch.masked_select(a_mlu, b_mlu > 0)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.00)

                out_cpu = torch.masked_select(b, a > 0)
                out_mlu = torch.masked_select(b_mlu, a_mlu > 0)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select_out(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,)]
        out_shapes = [(512, 2, 5), (100, 512, 2), (512), (100, 1)]
        dtype = [
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]
        for type_ in dtype:
            for shape, out_shape in zip(shapes, out_shapes):
                x = torch.rand(shape).to(type_)
                x_mlu = self.to_device(x)
                mask = torch.randn(shape) > 0
                out_cpu = torch.rand(out_shape).to(type_)
                out_mlu = out_cpu.to(torch.device("mlu"))
                mask_mlu = self.to_device(mask)
                torch.masked_select(x, mask, out=out_cpu)
                torch.masked_select(x_mlu, mask_mlu, out=out_mlu)

                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select_channels_last_and_not_dense(self):
        # test channels last
        self_tensor = torch.randn(100, 512, 2, 5)
        mask_tensor = torch.randn(100, 512, 2, 5) > 0
        cpu_result = torch.masked_select(
            self_tensor.to(memory_format=torch.channels_last),
            mask_tensor.to(memory_format=torch.channels_last),
        )

        device_result = torch.masked_select(
            self_tensor.to("mlu").to(memory_format=torch.channels_last),
            mask_tensor.to("mlu").to(memory_format=torch.channels_last),
        )
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0.003, use_MSE=True)

        # test not dense
        self_tensor = self_tensor[..., :2]
        mask_tensor = mask_tensor[..., :2]
        cpu_result = torch.masked_select(self_tensor, mask_tensor)
        device_result = torch.masked_select(
            self_tensor.to("mlu"), mask_tensor.to("mlu")
        )
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0.003, use_MSE=True)

        # test channels last broadcast
        self_tensor = torch.randn(2, 5, 5, 3).to(memory_format=torch.channels_last)
        mask_tensor = torch.tensor([True, False, True])
        cpu_result = torch.masked_select(self_tensor, mask_tensor)
        device_result = torch.masked_select(
            self_tensor.to("mlu"), mask_tensor.to("mlu")
        )
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_select_exception(self):
        self_tensor = torch.randn(1, 2, 5)
        mask_tensor = torch.randn(1, 2, 5).to(torch.float32)
        ref_msg = "masked_select: expected BoolTensor or ByteTensor for mask"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.masked_select(self_tensor.to("mlu"), mask_tensor.to("mlu"))
        self_tensor = torch.rand((1,), device="mlu").expand((3,))
        mask_tensor = torch.tensor(
            [True, False, True, True, False, False], device="mlu"
        )
        err_msg = "The size of tensor a (6) must match the size of tensor b (3) at non-singleton dimension 0"
        with self.assertRaises(RuntimeError) as info:
            torch.masked_select(self_tensor, mask_tensor, out=self_tensor)
        self.assertEqual(info.exception.args[0], err_msg)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_masked_select_bfloat16(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,)]
        dtype = [torch.bfloat16]
        for type_ in dtype:
            for shape in shapes:
                is_64bit = type_ in (torch.double, torch.long)
                x = torch.rand(shape).to(type_)
                x_mlu = self.to_device(x)
                if type_.is_floating_point:
                    x = torch.nn.Parameter(x)
                    x_mlu = torch.nn.Parameter(x_mlu)
                mask = torch.randn(shape) > 0
                mask_mlu = self.to_device(mask)
                out_cpu = torch.masked_select(x, mask)
                out_mlu = torch.masked_select(x_mlu, mask_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), 0.00 if not is_64bit else 0.003
                )
                if type_.is_floating_point:
                    grad = torch.randn_like(out_cpu)
                    grad_mlu = self.to_device(grad)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        x.grad, x_mlu.grad.cpu(), 0.00 if not is_64bit else 0.003
                    )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_masked_select_large_bfloat16(self):
        dtype = torch.bfloat16
        shape = (5, 64, 1024, 1024)
        a = torch.randn(shape)
        b = torch.randn(shape)
        a_mlu, b_mlu = self.to_mlu_dtype(a, dtype), self.to_mlu_dtype(b, dtype)
        out_cpu = torch.masked_select(a, b > 0)
        out_mlu = torch.masked_select(a_mlu, b_mlu > 0)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
